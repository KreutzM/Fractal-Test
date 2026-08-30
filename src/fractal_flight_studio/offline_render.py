from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Iterator, Protocol, Sequence, TypeAlias

import mpmath as mp
import numpy as np

from .camera import CameraState
from .flight_plan import (
    EvaluatedRenderState,
    FlightSource,
    evaluate_flight_frame,
    flight_path_for,
    surface_lighting_for,
)
from .models import RenderRequest
from .palettes import PaletteInput, palette_cache_key
from .surface_lighting import SurfaceLightingSettings
from .tonemapping import ToneMapState

ScalarDetail: TypeAlias = bool | float | int | str | None


class _FrameRenderer(Protocol):
    name: str

    def render_frame(
        self, request: RenderRequest, *args: Any, **kwargs: Any
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class OfflineRenderSettings:
    width: int = 1920
    height: int = 1080
    fps_numerator: int = 30
    fps_denominator: int = 1
    append_endpoint: bool = True
    max_frames: int = 1_000_000

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("offline render dimensions must be positive")
        if self.fps_numerator <= 0 or self.fps_denominator <= 0:
            raise ValueError("offline render frame rate must be positive")
        if self.max_frames < 1:
            raise ValueError("offline render max_frames must be positive")


@dataclass(frozen=True, slots=True)
class OfflineFramePlan:
    width: int
    height: int
    fps_numerator: int
    fps_denominator: int
    duration_text: str
    digits: int
    frame_count: int
    endpoint_included: bool
    endpoint_appended: bool

    def time_seconds_text(self, index: int) -> str:
        if not 0 <= index < self.frame_count:
            raise IndexError("offline frame index out of range")
        if index == self.frame_count - 1 and self.endpoint_included:
            return self.duration_text
        value = Fraction(index * self.fps_denominator, self.fps_numerator)
        with mp.workdps(self.digits):
            numeric = mp.mpf(value.numerator) / value.denominator
            return mp.nstr(numeric, n=self.digits, min_fixed=-6, max_fixed=6)


@dataclass(frozen=True, slots=True)
class OfflineFrameJob:
    index: int
    time_seconds_text: str
    camera: CameraState
    request: RenderRequest
    palette: PaletteInput = "inferno"
    cycles: float = 1.0
    render_state: EvaluatedRenderState | None = None


@dataclass(frozen=True, slots=True)
class OfflineFrame:
    index: int
    time_seconds_text: str
    camera: CameraState
    rgb: np.ndarray
    backend: str
    elapsed_seconds: float
    details: tuple[tuple[str, ScalarDetail], ...]


class OfflineFrameRenderError(RuntimeError):
    def __init__(self, job: OfflineFrameJob, cause: Exception) -> None:
        self.frame_index = job.index
        self.time_seconds_text = job.time_seconds_text
        self.cause = cause
        super().__init__(
            f"offline frame {job.index} at {job.time_seconds_text}s failed: {cause}"
        )


def build_offline_frame_plan(
    source: FlightSource, settings: OfflineRenderSettings = OfflineRenderSettings()
) -> OfflineFramePlan:
    path = flight_path_for(source)
    duration = _decimal_fraction(path.duration_text, "camera path duration")
    if duration <= 0:
        raise ValueError("camera path duration must be positive")
    frame_rate = Fraction(settings.fps_numerator, settings.fps_denominator)
    cadence_position = duration * frame_rate
    last_cadence_index = cadence_position.numerator // cadence_position.denominator
    cadence_count = last_cadence_index + 1
    endpoint_on_cadence = cadence_position.denominator == 1
    endpoint_appended = settings.append_endpoint and not endpoint_on_cadence
    frame_count = cadence_count + int(endpoint_appended)
    if frame_count > settings.max_frames:
        raise ValueError(
            f"offline render requires {frame_count} frames, exceeding max_frames="
            f"{settings.max_frames}"
        )
    return OfflineFramePlan(
        width=settings.width,
        height=settings.height,
        fps_numerator=settings.fps_numerator,
        fps_denominator=settings.fps_denominator,
        duration_text=path.duration_text,
        digits=path.digits,
        frame_count=frame_count,
        endpoint_included=endpoint_on_cadence or endpoint_appended,
        endpoint_appended=endpoint_appended,
    )


def iter_offline_frame_jobs(
    source: FlightSource,
    request_template: RenderRequest,
    plan: OfflineFramePlan,
    *,
    start_index: int = 0,
    stop_index: int | None = None,
    palette: PaletteInput = "inferno",
    cycles: float = 1.0,
) -> Iterator[OfflineFrameJob]:
    path = flight_path_for(source)
    if plan.duration_text != path.duration_text or plan.digits != path.digits:
        raise ValueError("offline frame plan does not match the camera path")
    stop = plan.frame_count if stop_index is None else stop_index
    if not 0 <= start_index <= stop <= plan.frame_count:
        raise ValueError("offline frame range is outside the plan")
    for index in range(start_index, stop):
        time_text = plan.time_seconds_text(index)
        evaluated = evaluate_flight_frame(
            source,
            time_text,
            request_template,
            palette=palette,
            cycles=cycles,
        )
        request = evaluated.build_request(
            request_template,
            width=plan.width,
            height=plan.height,
        )
        yield OfflineFrameJob(
            index,
            time_text,
            evaluated.camera,
            request,
            evaluated.render.palette,
            evaluated.render.cycles,
            evaluated.render,
        )


def render_offline_frame(
    job: OfflineFrameJob,
    renderer: _FrameRenderer,
    *,
    palette: PaletteInput | None = None,
    cycles: float | None = None,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    tone_state: ToneMapState | None = None,
    tone_scene_key: tuple[object, ...] | None = None,
    tone_state_locked: bool = False,
    surface_lighting: SurfaceLightingSettings | None = None,
) -> OfflineFrame:
    effective_palette = job.palette if palette is None else palette
    effective_cycles = job.cycles if cycles is None else cycles
    scene_key = tone_scene_key or (
        "offline-frame",
        job.request.fractal.value,
        job.request.precision.value,
        job.request.render_mode.value,
        job.request.effective_color_iterations,
        palette_cache_key(effective_palette),
        effective_cycles,
        phase,
        tone_mapping,
        surface_lighting,
    )
    try:
        frame = renderer.render_frame(
            job.request,
            effective_palette,
            effective_cycles,
            phase,
            tone_mapping=tone_mapping,
            tone_state=tone_state,
            tone_scene_key=scene_key,
            tone_smoothing=1.0,
            tone_state_locked=tone_state_locked,
            surface_lighting=surface_lighting,
        )
        rgb = np.asarray(frame.rgb)
        expected_shape = (job.request.height, job.request.width, 3)
        if rgb.shape != expected_shape:
            raise ValueError(
                f"renderer returned RGB shape {rgb.shape}, expected {expected_shape}"
            )
        details = dict(frame.details)
        return OfflineFrame(
            index=job.index,
            time_seconds_text=job.time_seconds_text,
            camera=job.camera,
            rgb=rgb.astype(np.uint8, copy=True),
            backend=frame.backend,
            elapsed_seconds=frame.elapsed_seconds,
            details=_scalar_details(details),
        )
    except Exception as exc:
        if isinstance(exc, OfflineFrameRenderError):
            raise
        raise OfflineFrameRenderError(job, exc) from exc


def render_offline_frames(
    source: FlightSource,
    request_template: RenderRequest,
    renderer: _FrameRenderer,
    plan: OfflineFramePlan,
    *,
    start_index: int = 0,
    stop_index: int | None = None,
    palette: PaletteInput = "inferno",
    cycles: float = 1.0,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    tone_states: Sequence[ToneMapState | None] | None = None,
    tone_scene_key: tuple[object, ...] | None = None,
    tone_state_locked: bool = False,
    surface_lighting: SurfaceLightingSettings | None = None,
) -> Iterator[OfflineFrame]:
    stop = plan.frame_count if stop_index is None else stop_index
    lighting = surface_lighting_for(source, surface_lighting)
    if tone_states is not None and len(tone_states) < stop:
        raise ValueError("tone-state plan is shorter than the requested frame range")
    for job in iter_offline_frame_jobs(
        source,
        request_template,
        plan,
        start_index=start_index,
        stop_index=stop,
        palette=palette,
        cycles=cycles,
    ):
        tone_state = None if tone_states is None else tone_states[job.index]
        yield render_offline_frame(
            job,
            renderer,
            phase=phase,
            tone_mapping=tone_mapping,
            tone_state=tone_state,
            tone_scene_key=tone_scene_key,
            tone_state_locked=tone_state_locked,
            surface_lighting=lighting,
        )


def _decimal_fraction(text: str, label: str) -> Fraction:
    try:
        value = Decimal(text)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a decimal number") from exc
    if not value.is_finite():
        raise ValueError(f"{label} must be finite")
    return Fraction(value)


def _scalar_details(details: dict) -> tuple[tuple[str, ScalarDetail], ...]:
    return tuple(
        sorted(
            (key, value)
            for key, value in details.items()
            if value is None or isinstance(value, (bool, float, int, str))
        )
    )
