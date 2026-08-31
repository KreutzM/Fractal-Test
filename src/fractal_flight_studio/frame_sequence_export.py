"""Resumable deterministic frame-sequence export.

This module publishes the frames of an existing offline render plan as a
directory of image files. It reuses ``offline_render`` planning, cadence and
camera contracts unchanged, validates every frame before publication and
publishes each frame atomically so an interrupted run never leaves a
partially written file that looks like a completed frame. A later run can
resume: already-published frames are verified and skipped unless the caller
explicitly asks to overwrite them.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import string
import time
from typing import Callable
from uuid import uuid4

import numpy as np
from PIL import Image, UnidentifiedImageError

from .ffmpeg_mp4 import CancellationCheck
from .flight_plan import FlightSource, flight_plan_fingerprint, surface_lighting_for
from .models import RenderRequest
from .offline_render import OfflineFramePlan, render_offline_frames
from .palettes import PaletteInput
from .surface_lighting import SurfaceLightingSettings
from .temporal_tonemapping import (
    TemporalToneSettings,
    ToneAnalysisCallback,
    ToneStability,
    analyze_offline_tone_states,
    offline_tone_scene_key,
)

ProgressCallback = Callable[["FrameSequenceProgress"], None]

_SUPPORTED_IMAGE_FORMATS = frozenset({"png"})
_PATTERN_FIELD = "index"


class FrameSequenceError(RuntimeError):
    """Raised when a frame cannot be validated or published."""


@dataclass(frozen=True, slots=True)
class FrameSequenceSettings:
    """Output naming and overwrite policy for a frame-sequence export."""

    image_format: str = "png"
    filename_pattern: str = "frame_{index:05d}.png"
    overwrite: bool = False

    def __post_init__(self) -> None:
        image_format = self.image_format.casefold()
        if image_format not in _SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                "unsupported frame-sequence image format "
                f"{self.image_format!r}; supported formats: "
                f"{', '.join(sorted(_SUPPORTED_IMAGE_FORMATS))}"
            )
        object.__setattr__(self, "image_format", image_format)
        _validate_filename_pattern(self.filename_pattern, image_format)

    def frame_name(self, index: int) -> str:
        if index < 0:
            raise ValueError("frame index must not be negative")
        return self.filename_pattern.format(index=index)


def _validate_filename_pattern(pattern: str, image_format: str) -> None:
    if not pattern or pattern != pattern.strip():
        raise ValueError("frame-sequence filename pattern must not be empty")
    separators = {os.sep, os.altsep, "/", "\\"} - {None}
    if any(separator in pattern for separator in separators):
        raise ValueError(
            "frame-sequence filename pattern must be a bare filename without "
            "directory separators"
        )
    fields: list[str | None] = []
    try:
        for _literal, field, _spec, _conversion in string.Formatter().parse(pattern):
            fields.append(field)
    except ValueError as exc:  # malformed braces
        raise ValueError(f"invalid frame-sequence filename pattern: {exc}") from exc
    if fields.count(_PATTERN_FIELD) != 1 or any(
        field not in (None, _PATTERN_FIELD) for field in fields
    ):
        raise ValueError(
            "frame-sequence filename pattern must use exactly one "
            "'{index}' placeholder and no other fields"
        )
    try:
        sample = pattern.format(index=0)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"invalid frame-sequence filename pattern {pattern!r}: {exc}"
        ) from exc
    if Path(sample).suffix.casefold() != f".{image_format}":
        raise ValueError(
            "frame-sequence filename pattern must end with the "
            f".{image_format} extension"
        )


@dataclass(frozen=True, slots=True)
class FrameSequencePlan:
    """One frame-sequence destination derived from an offline render plan."""

    offline_plan: OfflineFramePlan
    output_dir: Path
    settings: FrameSequenceSettings = FrameSequenceSettings()

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser())

    def frame_name(self, index: int) -> str:
        return self.settings.frame_name(index)

    def frame_path(self, index: int) -> Path:
        return self.output_dir / self.settings.frame_name(index)


@dataclass(frozen=True, slots=True)
class FrameSequenceProgress:
    """One progress event after a frame index was rendered or skipped."""

    action: str  # "rendered" or "skipped"
    completed_frames: int
    total_frames: int
    frame_index: int
    time_seconds_text: str


@dataclass(frozen=True, slots=True)
class FrameSequenceResult:
    """Outcome of one (possibly resumed) frame-sequence export run."""

    output_dir: Path
    start_index: int
    stop_index: int
    total_frames: int
    rendered_indices: tuple[int, ...]
    skipped_indices: tuple[int, ...]
    frame_paths: tuple[Path, ...]
    cancelled: bool
    elapsed_seconds: float

    @property
    def completed_indices(self) -> tuple[int, ...]:
        """Every plan index in the run range that now has a published frame."""

        return tuple(sorted(self.rendered_indices + self.skipped_indices))

    def next_start_index(self) -> int | None:
        """First range index without a published frame, or ``None`` if done."""

        completed = set(self.completed_indices)
        for index in range(self.start_index, self.stop_index):
            if index not in completed:
                return index
        return None


def parse_frame_range(
    plan: OfflineFramePlan,
    start_index: int,
    stop_index: int | None,
) -> tuple[int, int]:
    """Clamp and validate an inclusive-start, exclusive-stop frame range."""

    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise ValueError("start_index must be an integer")
    stop = plan.frame_count if stop_index is None else stop_index
    if isinstance(stop, bool) or not isinstance(stop, int):
        raise ValueError("stop_index must be an integer or None")
    if not 0 <= start_index <= stop <= plan.frame_count:
        raise ValueError(
            f"frame range [{start_index}, {stop}) is outside the offline plan "
            f"with {plan.frame_count} frames"
        )
    return start_index, stop


def export_frame_sequence(
    source: FlightSource,
    request_template: RenderRequest,
    renderer,
    offline_plan: OfflineFramePlan,
    output_dir: str | Path,
    settings: FrameSequenceSettings = FrameSequenceSettings(),
    *,
    start_index: int = 0,
    stop_index: int | None = None,
    palette: PaletteInput = "inferno",
    cycles: float = 1.0,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    temporal_tone: TemporalToneSettings = TemporalToneSettings(
        mode=ToneStability.PER_FRAME
    ),
    tone_analysis_progress: ToneAnalysisCallback | None = None,
    progress: ProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    surface_lighting: SurfaceLightingSettings | None = None,
) -> FrameSequenceResult:
    """Render an offline plan into deterministic PNG frames on disk.

    The run covers ``[start_index, stop_index)`` of ``offline_plan``. Frames
    whose destination file already exists are verified against the plan and
    skipped unless ``settings.overwrite`` is set, so an interrupted run can be
    resumed by re-invoking with the same output directory. Each rendered frame
    is validated and published atomically (unique temporary file plus
    ``os.replace``); cancellation stops the run between frames and returns a
    partial result whose ``next_start_index`` marks where to resume.
    """

    start, stop = parse_frame_range(offline_plan, start_index, stop_index)
    sequence_plan = FrameSequencePlan(
        offline_plan=offline_plan,
        output_dir=Path(output_dir).expanduser(),
        settings=settings,
    )
    total = stop - start

    lighting = surface_lighting_for(source, surface_lighting)
    tone_states = None
    tone_state_locked = False
    tone_scene_key = None
    if temporal_tone.mode is ToneStability.TEMPORAL and tone_mapping != "linear":
        if start != 0:
            raise ValueError(
                "temporal tone planning currently covers frames from index 0; "
                "render selected ranges or resumed chunks with per-frame tone "
                "stability, or start from index 0"
            )
        tone_states = analyze_offline_tone_states(
            source,
            request_template,
            renderer,
            offline_plan,
            stop_index=stop,
            settings=temporal_tone,
            palette=palette,
            cycles=cycles,
            phase=phase,
            tone_mapping=tone_mapping,
            progress=tone_analysis_progress,
            cancellation_requested=cancellation_requested,
            surface_lighting=lighting,
        )
        tone_state_locked = any(state is not None for state in tone_states)
        if tone_state_locked:
            tone_scene_key = (
                offline_tone_scene_key(
                    request_template,
                    tone_mapping,
                    palette,
                    cycles,
                    phase,
                ),
                flight_plan_fingerprint(source),
                lighting,
            )

    frames = render_offline_frames(
        source,
        request_template,
        renderer,
        offline_plan,
        start_index=start,
        stop_index=stop,
        palette=palette,
        cycles=cycles,
        phase=phase,
        tone_mapping=tone_mapping,
        tone_states=tone_states,
        tone_scene_key=tone_scene_key,
        tone_state_locked=tone_state_locked,
        surface_lighting=lighting,
    )

    rendered: list[int] = []
    skipped: list[int] = []
    published: list[Path] = []
    cancelled = False
    completed = 0
    started = time.perf_counter()
    sequence_plan.output_dir.mkdir(parents=True, exist_ok=True)

    def report(action: str, index: int, time_text: str) -> None:
        if progress is not None:
            progress(
                FrameSequenceProgress(
                    action, completed, total, index, time_text
                )
            )

    for frame in frames:
        if cancellation_requested is not None and cancellation_requested():
            cancelled = True
            break
        path = sequence_plan.frame_path(frame.index)
        if path.exists() and not settings.overwrite:
            _verify_existing_frame(path, sequence_plan)
            skipped.append(frame.index)
            published.append(path)
            completed += 1
            report("skipped", frame.index, frame.time_seconds_text)
            continue
        _publish_frame(frame.rgb, path, sequence_plan)
        rendered.append(frame.index)
        published.append(path)
        completed += 1
        report("rendered", frame.index, frame.time_seconds_text)

    return FrameSequenceResult(
        output_dir=sequence_plan.output_dir,
        start_index=start,
        stop_index=stop,
        total_frames=total,
        rendered_indices=tuple(rendered),
        skipped_indices=tuple(skipped),
        frame_paths=tuple(published),
        cancelled=cancelled,
        elapsed_seconds=time.perf_counter() - started,
    )


def _verify_existing_frame(path: Path, plan: FrameSequencePlan) -> None:
    """Reject an existing destination that is not a complete plan-sized frame."""

    expected_size = (plan.offline_plan.width, plan.offline_plan.height)
    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise FrameSequenceError(
            f"existing frame {path} could not be read as a PNG image: {exc}; "
            "re-run with overwrite=True to replace it"
        ) from exc
    if mode != "RGB" or size != expected_size:
        raise FrameSequenceError(
            f"existing frame {path} has mode {mode!r} and size {size}, "
            f"expected RGB and {expected_size}; re-run with overwrite=True "
            "to replace it"
        )


def _publish_frame(rgb: np.ndarray, path: Path, plan: FrameSequencePlan) -> None:
    """Validate one RGB frame and publish it atomically at ``path``."""

    array = np.ascontiguousarray(np.asarray(rgb))
    expected_shape = (plan.offline_plan.height, plan.offline_plan.width, 3)
    if array.shape != expected_shape:
        raise FrameSequenceError(
            f"cannot publish frame {path.name}: RGB shape {array.shape} "
            f"does not match expected {expected_shape}"
        )
    if array.dtype != np.uint8:
        raise FrameSequenceError(
            f"cannot publish frame {path.name}: dtype {array.dtype} is not "
            "uint8 RGB"
        )
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.part")
    try:
        image = Image.fromarray(array, mode="RGB")
        if plan.settings.image_format == "png":
            image.save(temporary, format="PNG")
        else:  # pragma: no cover - guarded by FrameSequenceSettings validation
            raise FrameSequenceError(
                f"unsupported frame-sequence image format {plan.settings.image_format!r}"
            )
        with Image.open(temporary) as written:
            written_mode = written.mode
            written_size = written.size
        if written_mode != "RGB" or written_size != image.size:
            raise FrameSequenceError(
                f"temporary frame {temporary.name} re-opened as "
                f"{written_mode!r} {written_size}, expected RGB {image.size}"
            )
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


__all__ = [
    "FrameSequenceError",
    "FrameSequencePlan",
    "FrameSequenceProgress",
    "FrameSequenceResult",
    "FrameSequenceSettings",
    "export_frame_sequence",
    "parse_frame_range",
]
