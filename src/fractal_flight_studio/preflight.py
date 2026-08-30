from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol, TypeAlias

import mpmath as mp

from .camera import CameraState
from .deep_zoom import PixelGridExhaustedError
from .flight_quality import FrameVisualQuality, analyze_frame_visual_quality
from .flight_plan import (
    FlightSource,
    evaluate_flight_frame,
    flight_path_for,
    flight_plan_fingerprint,
    surface_lighting_for,
)
from .models import RenderRequest
from .palettes import PaletteInput, palette_cache_key
from .surface_lighting import SurfaceLightingSettings

ScalarDetail: TypeAlias = bool | float | int | str | None
PreflightProgressCallback: TypeAlias = Callable[["PreflightSample", int], None]
CancellationCheck: TypeAlias = Callable[[], bool]


class PreflightCancelled(RuntimeError):
    pass


class _FrameRenderer(Protocol):
    name: str

    def render_frame(
        self, request: RenderRequest, *args: Any, **kwargs: Any
    ) -> Any: ...


class PreflightIssueKind(str, Enum):
    NUMERICAL = "numerical"
    VISUAL = "visual"
    RENDER_ERROR = "render_error"


@dataclass(frozen=True, slots=True)
class PreflightSettings:
    width: int = 240
    height: int = 150
    sample_interval_seconds_text: str = "0.5"
    max_samples: int = 240
    stop_on_failure: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("preflight dimensions must be positive")
        if self.max_samples < 2:
            raise ValueError("preflight max_samples must be at least two")
        try:
            interval = mp.mpf(self.sample_interval_seconds_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("preflight sample interval must be a decimal number") from exc
        if not mp.isfinite(interval) or interval <= 0:
            raise ValueError("preflight sample interval must be finite and positive")


@dataclass(frozen=True, slots=True)
class PreflightPlan:
    width: int
    height: int
    sample_times_text: tuple[str, ...]
    decimated: bool


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    sample_index: int
    time_seconds_text: str
    kind: PreflightIssueKind
    reason: str


@dataclass(frozen=True, slots=True)
class PreflightSample:
    index: int
    time_seconds_text: str
    camera: CameraState
    backend: str
    elapsed_seconds: float
    visual_quality: FrameVisualQuality | None
    details: tuple[tuple[str, ScalarDetail], ...]
    safe: bool


@dataclass(frozen=True, slots=True)
class PreflightReport:
    plan: PreflightPlan
    samples: tuple[PreflightSample, ...]
    issues: tuple[PreflightIssue, ...]
    stopped_early: bool
    total_elapsed_seconds: float

    @property
    def complete(self) -> bool:
        return not self.stopped_early and len(self.samples) == len(
            self.plan.sample_times_text
        )

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.kind is PreflightIssueKind.VISUAL
        )

    @property
    def blocking_issues(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.kind is not PreflightIssueKind.VISUAL
        )

    @property
    def safe(self) -> bool:
        return self.complete and not self.issues

    @property
    def exportable(self) -> bool:
        """Whether export may proceed after separately confirming visual warnings."""

        return self.complete and not self.blocking_issues

    @property
    def first_issue(self) -> PreflightIssue | None:
        return self.issues[0] if self.issues else None


def build_preflight_plan(source: FlightSource, settings: PreflightSettings) -> PreflightPlan:
    path = flight_path_for(source)
    with mp.workdps(path.digits):
        duration = mp.mpf(path.duration_text)
        interval = mp.mpf(settings.sample_interval_seconds_text)
        step_count = int(mp.floor(duration / interval))
        final_on_interval = interval * step_count == duration
        nominal_count = step_count + 1 + (0 if final_on_interval else 1)
        decimated = nominal_count > settings.max_samples
        if decimated:
            denominator = settings.max_samples - 1
            times = [duration * index / denominator for index in range(settings.max_samples)]
        else:
            times = [interval * index for index in range(step_count + 1)]
            if not final_on_interval:
                times.append(duration)

        times[0] = mp.mpf("0")
        times[-1] = duration
        text = tuple(
            mp.nstr(value, n=path.digits, min_fixed=-6, max_fixed=6)
            for value in times
        )
    return PreflightPlan(settings.width, settings.height, text, decimated)


def run_path_preflight(
    source: FlightSource,
    request_template: RenderRequest,
    renderer: _FrameRenderer,
    settings: PreflightSettings = PreflightSettings(),
    *,
    palette: PaletteInput = "inferno",
    cycles: float = 1.0,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    tone_smoothing: float = 0.16,
    surface_lighting: SurfaceLightingSettings | None = None,
    progress: PreflightProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> PreflightReport:
    path = flight_path_for(source)
    plan = build_preflight_plan(source, settings)
    samples: list[PreflightSample] = []
    issues: list[PreflightIssue] = []
    tone_state = None
    total_elapsed = 0.0
    stopped_early = False
    lighting = surface_lighting_for(source, surface_lighting)
    scene_key = (
        "path-preflight",
        flight_plan_fingerprint(source),
        request_template.precision.value,
        request_template.render_mode.value,
        palette_cache_key(palette),
        cycles,
        phase,
        tone_mapping,
        lighting,
    )

    for index, time_text in enumerate(plan.sample_times_text):
        if cancellation_requested is not None and cancellation_requested():
            raise PreflightCancelled(
                f"path preflight cancelled after {len(samples)} samples"
            )
        evaluated = evaluate_flight_frame(
            source,
            time_text,
            request_template,
            palette=palette,
            cycles=cycles,
        )
        camera = evaluated.camera
        request = evaluated.build_request(
            request_template,
            width=plan.width,
            height=plan.height,
        )
        frame_palette = evaluated.render.palette
        frame_cycles = evaluated.render.cycles
        sample_issues: list[PreflightIssue] = []
        backend = getattr(renderer, "name", "unknown")
        elapsed = 0.0
        visual_quality = None
        details: tuple[tuple[str, ScalarDetail], ...] = ()

        try:
            frame = renderer.render_frame(
                request,
                frame_palette,
                frame_cycles,
                phase,
                tone_mapping=tone_mapping,
                tone_state=tone_state,
                tone_scene_key=scene_key,
                tone_smoothing=tone_smoothing,
                surface_lighting=lighting,
            )
            backend = frame.backend
            elapsed = frame.elapsed_seconds
            total_elapsed += elapsed
            tone_state = frame.details.get("tone_state")
            details = _scalar_details(frame.details)
            if frame.details.get("pixel_grid_safe") is False:
                sample_issues.append(
                    PreflightIssue(
                        index,
                        time_text,
                        PreflightIssueKind.NUMERICAL,
                        _grid_reason(frame.details),
                    )
                )
            visual_quality = analyze_frame_visual_quality(frame.rgb)
            if not visual_quality.safe:
                sample_issues.append(
                    PreflightIssue(
                        index,
                        time_text,
                        PreflightIssueKind.VISUAL,
                        visual_quality.reason or "visual quality check failed",
                    )
                )
        except PixelGridExhaustedError as exc:
            details = _scalar_details(exc.quality.as_details())
            sample_issues.append(
                PreflightIssue(
                    index,
                    time_text,
                    PreflightIssueKind.NUMERICAL,
                    str(exc),
                )
            )
        except RuntimeError as exc:
            sample_issues.append(
                PreflightIssue(
                    index,
                    time_text,
                    PreflightIssueKind.RENDER_ERROR,
                    str(exc),
                )
            )

        issues.extend(sample_issues)
        sample = PreflightSample(
            index=index,
            time_seconds_text=time_text,
            camera=camera,
            backend=backend,
            elapsed_seconds=elapsed,
            visual_quality=visual_quality,
            details=details,
            safe=not sample_issues,
        )
        samples.append(sample)
        if progress is not None:
            progress(sample, len(plan.sample_times_text))
        if sample_issues and settings.stop_on_failure:
            stopped_early = index + 1 < len(plan.sample_times_text)
            break

    return PreflightReport(
        plan=plan,
        samples=tuple(samples),
        issues=tuple(issues),
        stopped_early=stopped_early,
        total_elapsed_seconds=total_elapsed,
    )


def _scalar_details(details: dict) -> tuple[tuple[str, ScalarDetail], ...]:
    return tuple(
        sorted(
            (key, value)
            for key, value in details.items()
            if value is None or isinstance(value, (bool, float, int, str))
        )
    )


def _grid_reason(details: dict) -> str:
    x_unique = details.get("pixel_grid_x_unique_fraction")
    y_unique = details.get("pixel_grid_y_unique_fraction")
    equal_run = details.get("pixel_grid_maximum_equal_run")
    return (
        "renderer pixel grid is unsafe"
        f" (X unique={x_unique!r}, Y unique={y_unique!r}, "
        f"largest equal run={equal_run!r})"
    )
