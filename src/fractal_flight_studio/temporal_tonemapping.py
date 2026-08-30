from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Sequence

from .ffmpeg_mp4 import CancellationCheck, Mp4ExportCancelled
from .flight_plan import FlightSource, flight_plan_fingerprint, surface_lighting_for
from .models import RenderRequest
from .palettes import PaletteInput, palette_cache_key
from .offline_render import OfflineFramePlan, iter_offline_frame_jobs
from .surface_lighting import SurfaceLightingSettings
from .tonemapping import ToneMapState


class ToneStability(str, Enum):
    """How automatic tone parameters are chosen for an offline video."""

    PER_FRAME = "per_frame"
    TEMPORAL = "temporal"


@dataclass(frozen=True, slots=True)
class TemporalToneSettings:
    mode: ToneStability | str = ToneStability.TEMPORAL
    analysis_width: int = 320
    analysis_height: int = 180
    smoothing: float = 0.18

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ToneStability(self.mode))
        if self.analysis_width <= 0 or self.analysis_height <= 0:
            raise ValueError("tone-analysis dimensions must be positive")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("temporal tone smoothing must be in the interval (0, 1]")


@dataclass(frozen=True, slots=True)
class ToneAnalysisProgress:
    frames_analyzed: int
    total_frames: int
    time_seconds_text: str


ToneAnalysisCallback = Callable[[ToneAnalysisProgress], None]


def offline_tone_scene_key(
    request: RenderRequest,
    tone_mapping: str,
    palette: PaletteInput,
    cycles: float,
    phase: float,
) -> tuple[object, ...]:
    return (
        "offline-video",
        request.fractal.value,
        request.precision.value,
        request.render_mode.value,
        request.effective_color_iterations,
        request.exponent,
        request.julia_c_real,
        request.julia_c_imag,
        palette_cache_key(palette),
        cycles,
        phase,
        tone_mapping,
    )


def analyze_offline_tone_states(
    source: FlightSource,
    request_template: RenderRequest,
    renderer,
    offline_plan: OfflineFramePlan,
    *,
    stop_index: int,
    settings: TemporalToneSettings = TemporalToneSettings(),
    palette: PaletteInput = "inferno",
    cycles: float = 1.0,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    surface_lighting: SurfaceLightingSettings | None = None,
    progress: ToneAnalysisCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
) -> tuple[ToneMapState | None, ...]:
    """Measure per-frame tone targets at low resolution and smooth them zero-phase."""

    if not 0 <= stop_index <= offline_plan.frame_count:
        raise ValueError("tone-analysis range is outside the offline plan")
    if settings.mode is ToneStability.PER_FRAME or tone_mapping == "linear":
        return tuple(None for _ in range(stop_index))

    lighting = surface_lighting_for(source, surface_lighting)
    analysis_plan = replace(
        offline_plan,
        width=settings.analysis_width,
        height=settings.analysis_height,
    )
    scene_key = (
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
    states: list[ToneMapState | None] = []
    for job in iter_offline_frame_jobs(
        source,
        request_template,
        analysis_plan,
        stop_index=stop_index,
        palette=palette,
        cycles=cycles,
    ):
        if cancellation_requested is not None and cancellation_requested():
            raise Mp4ExportCancelled("MP4 export cancelled during tone analysis")
        frame = renderer.render_frame(
            job.request,
            job.palette,
            job.cycles,
            phase,
            tone_mapping=tone_mapping,
            tone_state=None,
            tone_scene_key=scene_key,
            tone_smoothing=1.0,
            tone_state_locked=False,
            surface_lighting=lighting,
        )
        state = frame.details.get("tone_state")
        if state is not None and not isinstance(state, ToneMapState):
            raise TypeError("renderer returned an invalid tone_state during analysis")
        states.append(state)
        if progress is not None:
            progress(ToneAnalysisProgress(job.index + 1, stop_index, job.time_seconds_text))

    return stabilize_tone_states(states, smoothing=settings.smoothing)


def stabilize_tone_states(
    states: Sequence[ToneMapState | None],
    *,
    smoothing: float = 0.18,
) -> tuple[ToneMapState | None, ...]:
    """Apply deterministic forward/backward smoothing without directional exposure lag."""

    if not 0.0 < smoothing <= 1.0:
        raise ValueError("temporal tone smoothing must be in the interval (0, 1]")
    if not states:
        return ()

    filled = _fill_missing_states(states)
    if all(state is None for state in filled):
        return tuple(None for _ in states)

    concrete = tuple(state for state in filled if state is not None)
    first = concrete[0]
    if any(state.mode != first.mode or state.scene_key != first.scene_key for state in concrete):
        raise ValueError("tone states must use one mode and scene key")

    forward = _smooth_pass(filled, smoothing)
    backward = tuple(reversed(_smooth_pass(tuple(reversed(filled)), smoothing)))
    result: list[ToneMapState | None] = []
    for left, right in zip(forward, backward):
        assert left is not None and right is not None
        result.append(_average_state(left, right))
    return tuple(result)


def _fill_missing_states(
    states: Sequence[ToneMapState | None],
) -> tuple[ToneMapState | None, ...]:
    if all(state is None for state in states):
        return tuple(None for _ in states)

    result = list(states)
    previous: ToneMapState | None = None
    for index, state in enumerate(result):
        if state is None and previous is not None:
            result[index] = previous
        elif state is not None:
            previous = state

    following: ToneMapState | None = None
    for index in range(len(result) - 1, -1, -1):
        state = result[index]
        if state is None and following is not None:
            result[index] = following
        elif state is not None:
            following = state
    return tuple(result)


def _smooth_pass(
    states: Sequence[ToneMapState | None], alpha: float
) -> tuple[ToneMapState | None, ...]:
    output: list[ToneMapState | None] = []
    current: ToneMapState | None = None
    for state in states:
        if state is None:
            output.append(current)
            continue
        current = state if current is None else _blend_state(current, state, alpha)
        output.append(current)
    return tuple(output)


def _blend_state(previous: ToneMapState, target: ToneMapState, alpha: float) -> ToneMapState:
    return _make_state(
        target,
        previous.low + alpha * (target.low - previous.low),
        previous.high + alpha * (target.high - previous.high),
        previous.strength + alpha * (target.strength - previous.strength),
        previous.gamma + alpha * (target.gamma - previous.gamma),
    )


def _average_state(left: ToneMapState, right: ToneMapState) -> ToneMapState:
    return _make_state(
        left,
        0.5 * (left.low + right.low),
        0.5 * (left.high + right.high),
        0.5 * (left.strength + right.strength),
        0.5 * (left.gamma + right.gamma),
    )


def _make_state(
    template: ToneMapState,
    low: float,
    high: float,
    strength: float,
    gamma: float,
) -> ToneMapState:
    if high <= low:
        center = 0.5 * (low + high)
        low = center - 5e-10
        high = center + 5e-10
    return ToneMapState(
        mode=template.mode,
        scene_key=template.scene_key,
        low=float(low),
        high=float(high),
        strength=max(float(strength), 1e-9),
        gamma=max(float(gamma), 1e-9),
    )
