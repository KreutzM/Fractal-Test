from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import mpmath as mp

from .camera import CameraState
from .deep_zoom_targets import DeepZoomTarget
from .flight_path import CenterInterpolation, Easing, FlightKeyframe
from .flight_plan import (
    FlightScene,
    PaletteTransition,
    EvaluatedRenderState,
    RenderCue,
    RenderProfile,
)


class TransitionMode(str, Enum):
    """Camera-routing strategy between two flight targets."""

    AUTO = "auto"
    DIRECT = "direct"
    BRIDGE = "bridge"
    OVERVIEW = "overview"
    CUT = "cut"


@dataclass(frozen=True, slots=True)
class TransitionTarget:
    """Portable destination used by the pure transition planner."""

    name: str
    camera: CameraState
    profile: RenderProfile
    scene: FlightScene

    @classmethod
    def from_deep_zoom_target(
        cls,
        target: DeepZoomTarget,
        *,
        scene: FlightScene,
        cycles_text: str = "1",
    ) -> "TransitionTarget":
        if target.fractal != scene.fractal:
            raise ValueError(
                "target fractal does not match the flight-plan scene; use a separate plan for scene changes"
            )
        return cls(
            target.name,
            CameraState(
                target.center_x_text,
                target.center_y_text,
                target.view_width_text,
            ),
            RenderProfile(
                target.recommended_iterations,
                target.reference_bits,
                target.palette,
                cycles_text,
            ),
            scene,
        )


@dataclass(frozen=True, slots=True)
class FreeTargetValues:
    """Portable user-editable values for one free flight target."""

    center_x_text: str
    center_y_text: str
    view_width_text: str
    max_iterations: int
    reference_bits: int
    palette: str
    cycles_text: str


@dataclass(frozen=True, slots=True)
class TransitionSettings:
    """Deterministic heuristics used to route and time one transition."""

    aspect_ratio_text: str = "1.7777777777777778"
    overview_camera: CameraState = CameraState("-0.5", "0", "3.5")
    zoom_stops_per_second_text: str = "3"
    pan_viewports_per_second_text: str = "0.8"
    hold_before_text: str = "0.75"
    hold_after_text: str = "1"
    cut_seconds_text: str = "0.033333333333333333"
    bridge_margin_text: str = "1.35"
    direct_distance_threshold_text: str = "0.85"
    overview_threshold_text: str = "0.78"

    def values(self, *, digits: int) -> tuple[mp.mpf, ...]:
        with mp.workdps(digits):
            values = tuple(
                _positive_decimal(text, label, digits=digits)
                for text, label in (
                    (self.aspect_ratio_text, "aspect ratio"),
                    (self.zoom_stops_per_second_text, "zoom speed"),
                    (self.pan_viewports_per_second_text, "pan speed"),
                    (self.hold_before_text, "hold-before duration"),
                    (self.hold_after_text, "hold-after duration"),
                    (self.cut_seconds_text, "cut duration"),
                    (self.bridge_margin_text, "bridge margin"),
                    (
                        self.direct_distance_threshold_text,
                        "direct transition threshold",
                    ),
                    (self.overview_threshold_text, "overview threshold"),
                )
            )
            if values[-1] > 1:
                raise ValueError("overview threshold must not exceed one")
            self.overview_camera.values(digits=digits)
            return values


@dataclass(frozen=True, slots=True)
class TransitionPlan:
    """Absolute keyframes and render cues that can be appended atomically."""

    source_camera: CameraState
    source_profile: RenderProfile
    scene: FlightScene
    mode: TransitionMode
    requested_mode: TransitionMode
    start_time_text: str
    arrival_time_text: str
    end_time_text: str
    keyframes: tuple[FlightKeyframe, ...]
    render_cues: tuple[RenderCue, ...]
    bridge_width_text: str | None
    target_name: str
    digits: int

    @property
    def duration_text(self) -> str:
        with mp.workdps(self.digits):
            duration = mp.mpf(self.end_time_text) - mp.mpf(self.start_time_text)
            return _format_decimal(duration, digits=self.digits)

    @property
    def intermediate_keyframe_count(self) -> int:
        # Exclude the target arrival and final hold from the route summary.
        return max(0, len(self.keyframes) - 2)

    @property
    def summary(self) -> str:
        bridge = (
            ""
            if self.bridge_width_text is None
            else f"; Brückenbreite {mp.nstr(mp.mpf(self.bridge_width_text), 6)}"
        )
        return (
            f"{self.mode.value}: {self.duration_text} s; "
            f"{self.intermediate_keyframe_count} Zwischen-Keyframes{bridge}"
        )

    def __post_init__(self) -> None:
        if self.digits < 20:
            raise ValueError("transition precision must be at least 20 digits")
        if not self.keyframes:
            raise ValueError("transition plan requires camera keyframes")
        with mp.workdps(self.digits):
            start = mp.mpf(self.start_time_text)
            arrival = mp.mpf(self.arrival_time_text)
            end = mp.mpf(self.end_time_text)
            if not (start < arrival <= end):
                raise ValueError("transition times must satisfy start < arrival <= end")
            key_times = tuple(
                frame.time_seconds(digits=self.digits) for frame in self.keyframes
            )
            if any(value <= start for value in key_times):
                raise ValueError("transition keyframes must follow the start time")
            if any(right <= left for left, right in zip(key_times, key_times[1:])):
                raise ValueError("transition keyframe times must be strictly increasing")
            if key_times[-1] != end:
                raise ValueError("last transition keyframe must end the transition")
            cue_times = tuple(
                cue.time_seconds(digits=self.digits) for cue in self.render_cues
            )
            if cue_times and (cue_times[0] != start or cue_times[-1] != arrival):
                raise ValueError("transition render cues must anchor start and arrival")


def end_render_profile(
    *,
    render_state: EvaluatedRenderState,
) -> RenderProfile:
    """Convert an evaluated end state back into one exact source profile."""

    palette = (
        render_state.palette.target
        if render_state.palette.mix >= 1.0
        else render_state.palette.source
    )
    assert palette is not None
    return RenderProfile(
        render_state.max_iterations,
        render_state.reference_bits,
        palette,
        render_state.cycles_text,
    )


def plan_transition(
    source_camera: CameraState,
    source_profile: RenderProfile,
    target: TransitionTarget,
    *,
    start_time_text: str,
    digits: int = 80,
    requested_mode: TransitionMode | str = TransitionMode.AUTO,
    palette_transition: PaletteTransition | str = PaletteTransition.BLEND,
    settings: TransitionSettings = TransitionSettings(),
) -> TransitionPlan:
    """Build a deterministic route from the current path end to one target.

    The planner uses exact decimal arithmetic for positions, widths and timing.
    It chooses the smallest useful bridge view in ``auto`` mode and only routes
    through the full overview when that bridge is already close to the root view.
    """

    requested = TransitionMode(requested_mode)
    palette_mode = PaletteTransition(palette_transition)
    with mp.workdps(digits):
        start_time = _non_negative_decimal(
            start_time_text, "transition start time", digits=digits
        )
        source_x, source_y, source_width = source_camera.values(digits=digits)
        target_x, target_y, target_width = target.camera.values(digits=digits)
        (
            aspect,
            zoom_speed,
            pan_speed,
            hold_before,
            hold_after,
            cut_seconds,
            bridge_margin,
            direct_threshold,
            overview_threshold,
        ) = settings.values(digits=digits)
        overview_x, overview_y, overview_width = settings.overview_camera.values(
            digits=digits
        )

        dx = abs(target_x - source_x)
        dy = abs(target_y - source_y)
        screen_distance = max(dx, aspect * dy)
        reference_width = max(source_width, target_width)
        normalized_distance = screen_distance / reference_width
        required_bridge = bridge_margin * (
            screen_distance + (source_width + target_width) / 2
        )
        required_bridge = max(
            required_bridge,
            reference_width * mp.mpf("1.2"),
        )
        bridge_width = min(required_bridge, overview_width)

        mode = requested
        if requested is TransitionMode.AUTO:
            if normalized_distance <= direct_threshold:
                mode = TransitionMode.DIRECT
            elif required_bridge >= overview_width * overview_threshold:
                mode = TransitionMode.OVERVIEW
            else:
                mode = TransitionMode.BRIDGE

        current = start_time
        frames: list[FlightKeyframe] = []

        def add_after(
            duration: mp.mpf,
            camera: CameraState,
            *,
            easing: Easing = Easing.SMOOTHERSTEP,
            center: CenterInterpolation = CenterInterpolation.FOCUS,
        ) -> mp.mpf:
            nonlocal current
            current += duration
            frames.append(
                FlightKeyframe(
                    _format_decimal(current, digits=digits),
                    camera,
                    easing,
                    center,
                )
            )
            return current

        # A duplicate first frame creates a short hold without modifying the
        # existing path's last keyframe or its preceding segment.
        add_after(hold_before, source_camera)

        bridge_text: str | None = None
        if mode is TransitionMode.CUT:
            # STEP holds the old camera for the entire tiny outgoing segment and
            # switches exactly at its end, giving the path model a real cut.
            frames[-1] = FlightKeyframe(
                frames[-1].time_seconds_text,
                source_camera,
                Easing.STEP,
                CenterInterpolation.LINEAR,
            )
            arrival = add_after(cut_seconds, target.camera)
        elif mode is TransitionMode.DIRECT:
            direct_seconds = _direct_duration(
                source_width,
                target_width,
                normalized_distance,
                zoom_speed,
            )
            arrival = add_after(direct_seconds, target.camera)
        elif mode is TransitionMode.BRIDGE:
            bridge_text = _format_decimal(bridge_width, digits=digits)
            source_bridge = CameraState.from_values(
                source_x, source_y, bridge_width, digits=digits
            )
            target_bridge = CameraState.from_values(
                target_x, target_y, bridge_width, digits=digits
            )
            add_after(
                _zoom_duration(source_width, bridge_width, zoom_speed),
                source_bridge,
            )
            add_after(
                _pan_duration(screen_distance, bridge_width, pan_speed),
                target_bridge,
                center=CenterInterpolation.LINEAR,
            )
            arrival = add_after(
                _zoom_duration(bridge_width, target_width, zoom_speed),
                target.camera,
            )
        else:
            bridge_text = _format_decimal(overview_width, digits=digits)
            source_overview = CameraState.from_values(
                source_x, source_y, overview_width, digits=digits
            )
            target_overview = CameraState.from_values(
                target_x, target_y, overview_width, digits=digits
            )
            add_after(
                _zoom_duration(source_width, overview_width, zoom_speed),
                source_overview,
            )
            overview_camera = CameraState.from_values(
                overview_x, overview_y, overview_width, digits=digits
            )
            overview_distance = max(
                abs(overview_x - source_x), aspect * abs(overview_y - source_y)
            )
            add_after(
                _pan_duration(overview_distance, overview_width, pan_speed),
                overview_camera,
                center=CenterInterpolation.LINEAR,
            )
            target_overview_distance = max(
                abs(target_x - overview_x), aspect * abs(target_y - overview_y)
            )
            add_after(
                _pan_duration(target_overview_distance, overview_width, pan_speed),
                target_overview,
                center=CenterInterpolation.LINEAR,
            )
            arrival = add_after(
                _zoom_duration(overview_width, target_width, zoom_speed),
                target.camera,
            )

        arrival_text = _format_decimal(arrival, digits=digits)
        add_after(hold_after, target.camera)
        end_text = _format_decimal(current, digits=digits)

        effective_palette_mode = palette_mode
        if mode is TransitionMode.CUT and palette_mode is PaletteTransition.BLEND:
            effective_palette_mode = PaletteTransition.CUT
        source_anchor = RenderCue(
            _format_decimal(start_time, digits=digits),
            source_profile,
            PaletteTransition.CUT,
        )
        target_profile = target.profile
        if effective_palette_mode is PaletteTransition.HOLD:
            # The cue still carries target quality, while HOLD deliberately keeps
            # the current effective palette and cycles.
            target_profile = RenderProfile(
                target.profile.max_iterations,
                target.profile.reference_bits,
                target.profile.palette,
                target.profile.cycles_text,
            )
        target_cue = RenderCue(
            arrival_text,
            target_profile,
            effective_palette_mode,
        )
        return TransitionPlan(
            source_camera=source_camera,
            source_profile=source_profile,
            scene=target.scene,
            mode=mode,
            requested_mode=requested,
            start_time_text=_format_decimal(start_time, digits=digits),
            arrival_time_text=arrival_text,
            end_time_text=end_text,
            keyframes=tuple(frames),
            render_cues=(source_anchor, target_cue),
            bridge_width_text=bridge_text,
            target_name=target.name,
            digits=digits,
        )


def suggested_target_width(
    source_camera: CameraState,
    *,
    zoom_factor_text: str = "10",
    digits: int = 80,
) -> str:
    """Return an exact, moderate default width for a free right-click target."""

    with mp.workdps(digits):
        _x, _y, width = source_camera.values(digits=digits)
        factor = _positive_decimal(zoom_factor_text, "zoom factor", digits=digits)
        if factor <= 1:
            raise ValueError("zoom factor must be greater than one")
        return _format_decimal(width / factor, digits=digits)


def merge_render_cues(
    existing: Iterable[RenderCue],
    additions: Iterable[RenderCue],
    *,
    digits: int,
) -> tuple[RenderCue, ...]:
    """Merge cues by exact time, replacing an existing anchor at the same time."""

    with mp.workdps(digits):
        by_time: dict[mp.mpf, RenderCue] = {
            cue.time_seconds(digits=digits): cue for cue in existing
        }
        for cue in additions:
            by_time[cue.time_seconds(digits=digits)] = cue
        return tuple(by_time[key] for key in sorted(by_time))


def _direct_duration(
    start_width: mp.mpf,
    end_width: mp.mpf,
    normalized_distance: mp.mpf,
    zoom_speed: mp.mpf,
) -> mp.mpf:
    zoom = _zoom_duration(start_width, end_width, zoom_speed)
    steering = min(mp.mpf("2.5"), normalized_distance * mp.mpf("1.25"))
    return min(mp.mpf("12"), max(mp.mpf("1.25"), zoom + steering))


def _zoom_duration(
    start_width: mp.mpf,
    end_width: mp.mpf,
    stops_per_second: mp.mpf,
) -> mp.mpf:
    stops = abs(mp.log(end_width / start_width, 2))
    return min(mp.mpf("12"), max(mp.mpf("0.9"), stops / stops_per_second))


def _pan_duration(
    distance: mp.mpf,
    view_width: mp.mpf,
    viewports_per_second: mp.mpf,
) -> mp.mpf:
    normalized = distance / view_width
    return min(
        mp.mpf("5"),
        max(mp.mpf("1"), normalized / viewports_per_second),
    )


def _positive_decimal(text: str, label: str, *, digits: int) -> mp.mpf:
    value = _non_negative_decimal(text, label, digits=digits)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _non_negative_decimal(text: str, label: str, *, digits: int) -> mp.mpf:
    try:
        with mp.workdps(digits):
            value = mp.mpf(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a decimal number") from exc
    if not mp.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _format_decimal(value: mp.mpf, *, digits: int) -> str:
    return mp.nstr(value, n=digits, min_fixed=-6, max_fixed=12)
