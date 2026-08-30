from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, TypeAlias

import mpmath as mp

from .camera import CameraState
from .flight_path import CameraPath
from .models import FractalKind, RenderRequest
from .palettes import PaletteBlend, PaletteInput, palette_names
from .surface_lighting import SurfaceLightingSettings


def _finite_decimal(text: str, label: str, *, digits: int = 80) -> mp.mpf:
    if not isinstance(text, str):
        raise ValueError(f"{label} must be a decimal string")
    try:
        with mp.workdps(digits):
            value = mp.mpf(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a decimal number") from exc
    if not mp.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _validated_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("flight-plan name must be a string")
    name = value.strip()
    if not name:
        raise ValueError("flight-plan name must not be empty")
    if len(name) > 200:
        raise ValueError("flight-plan name must not exceed 200 characters")
    if any(ord(character) < 32 for character in name):
        raise ValueError("flight-plan name must not contain control characters")
    return name


class PaletteTransition(str, Enum):
    """How a render cue changes from the previous effective coloring."""

    HOLD = "hold"
    BLEND = "blend"
    CUT = "cut"


@dataclass(frozen=True, slots=True)
class FlightScene:
    """Portable fractal-space parameters shared by the complete flight."""

    fractal: FractalKind | str = FractalKind.MANDELBROT
    exponent: int = 3
    julia_c_real_text: str = "-0.8"
    julia_c_imag_text: str = "0.156"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fractal", FractalKind(self.fractal))
        if isinstance(self.exponent, bool) or not isinstance(self.exponent, int):
            raise ValueError("scene exponent must be an integer")
        if not 2 <= self.exponent <= 8:
            raise ValueError("scene exponent must be between 2 and 8")
        self.julia_values()

    def julia_values(self) -> tuple[mp.mpf, mp.mpf]:
        real = _finite_decimal(self.julia_c_real_text, "Julia real component")
        imag = _finite_decimal(self.julia_c_imag_text, "Julia imaginary component")
        return real, imag


@dataclass(frozen=True, slots=True)
class RenderProfile:
    """Portable quality and coloring settings active at one timeline cue."""

    max_iterations: int = 400
    reference_bits: int = 256
    palette: str = "inferno"
    cycles_text: str = "1"

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int):
            raise ValueError("render max_iterations must be an integer")
        if not 1 <= self.max_iterations <= 100_000:
            raise ValueError("render max_iterations must be between 1 and 100000")
        if isinstance(self.reference_bits, bool) or not isinstance(self.reference_bits, int):
            raise ValueError("render reference_bits must be an integer")
        if not 64 <= self.reference_bits <= 16_384:
            raise ValueError("render reference_bits must be between 64 and 16384")
        if self.palette not in palette_names():
            raise ValueError(f"unknown render palette {self.palette!r}")
        cycles = _finite_decimal(self.cycles_text, "render cycles")
        if cycles <= 0:
            raise ValueError("render cycles must be positive")

    @property
    def cycles(self) -> float:
        return float(_finite_decimal(self.cycles_text, "render cycles"))


@dataclass(frozen=True, slots=True)
class RenderCue:
    """One exact render-profile change on the flight timeline."""

    time_seconds_text: str
    profile: RenderProfile = RenderProfile()
    palette_transition: PaletteTransition | str = PaletteTransition.HOLD

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "palette_transition",
            PaletteTransition(self.palette_transition),
        )

    def time_seconds(self, *, digits: int) -> mp.mpf:
        value = _finite_decimal(self.time_seconds_text, "render cue time", digits=digits)
        if value < 0:
            raise ValueError("render cue time must be non-negative")
        return value


@dataclass(frozen=True, slots=True)
class EvaluatedRenderState:
    """Resolved quality and coloring settings for one exact flight time."""

    max_iterations: int
    color_iterations: int
    reference_bits: int
    palette: PaletteBlend
    cycles_text: str
    active_cue_index: int
    next_cue_index: int | None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("evaluated max_iterations must be positive")
        if self.color_iterations < self.max_iterations:
            raise ValueError(
                "evaluated color_iterations must not be below max_iterations"
            )
        if self.reference_bits < 64:
            raise ValueError("evaluated reference_bits must be at least 64")
        cycles = _finite_decimal(self.cycles_text, "evaluated render cycles")
        if cycles <= 0:
            raise ValueError("evaluated render cycles must be positive")

    @property
    def cycles(self) -> float:
        return float(_finite_decimal(self.cycles_text, "evaluated render cycles"))


@dataclass(frozen=True, slots=True)
class RenderTrack:
    """Immutable, ordered render cues for one flight plan."""

    cues: tuple[RenderCue, ...] | Iterable[RenderCue]
    digits: int = 80
    _times: tuple[mp.mpf, ...] = field(init=False, repr=False)
    _effective_color_profiles: tuple[RenderProfile, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 20 <= self.digits <= 16_384:
            raise ValueError("render-track precision must be between 20 and 16384 digits")
        cues = tuple(self.cues)
        if not cues:
            raise ValueError("a render track requires at least one cue")
        times = tuple(cue.time_seconds(digits=self.digits) for cue in cues)
        if times[0] != 0:
            raise ValueError("the first render cue must start at zero seconds")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("render cue times must be strictly increasing")

        effective_colors: list[RenderProfile] = [cues[0].profile]
        for cue in cues[1:]:
            if cue.palette_transition is PaletteTransition.HOLD:
                effective_colors.append(effective_colors[-1])
            else:
                effective_colors.append(cue.profile)

        object.__setattr__(self, "cues", cues)
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_effective_color_profiles", tuple(effective_colors))

    @classmethod
    def default(
        cls,
        profile: RenderProfile = RenderProfile(),
        *,
        digits: int = 80,
    ) -> "RenderTrack":
        return cls((RenderCue("0", profile, PaletteTransition.HOLD),), digits=digits)

    @property
    def first_profile(self) -> RenderProfile:
        return self.cues[0].profile

    @property
    def color_iterations(self) -> int:
        """One stable color scale for the complete render track."""

        return max(cue.profile.max_iterations for cue in self.cues)

    @property
    def duration_text(self) -> str:
        return self.cues[-1].time_seconds_text

    def replace_first_profile(self, profile: RenderProfile) -> "RenderTrack":
        first = self.cues[0]
        cues = (
            RenderCue(first.time_seconds_text, profile, first.palette_transition),
            *self.cues[1:],
        )
        return RenderTrack(cues, digits=self.digits)

    def evaluate(self, time_seconds: str | int | float | mp.mpf) -> EvaluatedRenderState:
        """Resolve conservative quality and deterministic palette behavior."""

        with mp.workdps(self.digits):
            if isinstance(time_seconds, float):
                time_value = mp.mpf(repr(time_seconds))
            else:
                try:
                    time_value = mp.mpf(time_seconds)
                except (TypeError, ValueError) as exc:
                    raise ValueError("render evaluation time must be a decimal number") from exc
            if not mp.isfinite(time_value):
                raise ValueError("render evaluation time must be finite")
            if time_value <= self._times[0]:
                time_value = self._times[0]
            elif time_value >= self._times[-1]:
                time_value = self._times[-1]

            active = max(0, bisect_right(self._times, time_value) - 1)
            next_index = active + 1 if active + 1 < len(self.cues) else None
            active_profile = self.cues[active].profile
            if next_index is None:
                maximum_iterations = active_profile.max_iterations
                reference_bits = active_profile.reference_bits
            else:
                next_profile = self.cues[next_index].profile
                maximum_iterations = max(
                    active_profile.max_iterations,
                    next_profile.max_iterations,
                )
                reference_bits = max(
                    active_profile.reference_bits,
                    next_profile.reference_bits,
                )

            source_color = self._effective_color_profiles[active]
            palette: PaletteBlend = PaletteBlend.solid(source_color.palette)
            cycles = mp.mpf(source_color.cycles_text)
            if next_index is not None:
                target_cue = self.cues[next_index]
                if target_cue.palette_transition is PaletteTransition.BLEND:
                    start = self._times[active]
                    end = self._times[next_index]
                    progress = (time_value - start) / (end - start)
                    progress = min(mp.mpf("1"), max(mp.mpf("0"), progress))
                    target_color = target_cue.profile
                    palette = PaletteBlend(
                        source_color.palette,
                        target_color.palette,
                        float(progress),
                    )
                    cycles = mp.mpf(source_color.cycles_text) + (
                        mp.mpf(target_color.cycles_text)
                        - mp.mpf(source_color.cycles_text)
                    ) * progress

            cycles_text = mp.nstr(
                cycles,
                n=self.digits,
                min_fixed=-6,
                max_fixed=6,
            )
            return EvaluatedRenderState(
                max_iterations=maximum_iterations,
                color_iterations=self.color_iterations,
                reference_bits=reference_bits,
                palette=palette,
                cycles_text=cycles_text,
                active_cue_index=active,
                next_cue_index=next_index,
            )


@dataclass(frozen=True, slots=True)
class FlightPlanDefaults:
    """Defaults used when importing legacy flight-plan schemas."""

    scene: FlightScene = FlightScene()
    render_profile: RenderProfile = RenderProfile()
    surface_lighting: SurfaceLightingSettings = SurfaceLightingSettings()


@dataclass(frozen=True, slots=True)
class EvaluatedFlightFrame:
    """One exact camera and render-state sample from a complete flight plan."""

    time_seconds_text: str
    camera: CameraState
    scene: FlightScene
    render: EvaluatedRenderState
    digits: int

    def build_request(
        self,
        template: RenderRequest,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> RenderRequest:
        julia_real, julia_imag = self.scene.julia_values()
        request = replace(
            template,
            width=template.width if width is None else width,
            height=template.height if height is None else height,
            viewport=self.camera.proxy_viewport(digits=self.digits),
            fractal=self.scene.fractal,
            max_iterations=self.render.max_iterations,
            color_iterations=self.render.color_iterations,
            julia_c_real=float(julia_real),
            julia_c_imag=float(julia_imag),
            exponent=self.scene.exponent,
            reference_bits=self.render.reference_bits,
            center_x_text=self.camera.center_x_text,
            center_y_text=self.camera.center_y_text,
            view_width_text=self.camera.view_width_text,
        )
        request.validate()
        return request


@dataclass(frozen=True, slots=True)
class FlightPlanDocument:
    """Portable camera, scene and render timeline plus document metadata."""

    name: str
    path: CameraPath
    scene: FlightScene = FlightScene()
    render_track: RenderTrack | None = None
    surface_lighting: SurfaceLightingSettings = SurfaceLightingSettings()
    source_schema_version: int = field(default=3, compare=False, repr=False)

    def __post_init__(self) -> None:
        name = _validated_name(self.name)
        object.__setattr__(self, "name", name)
        if isinstance(self.source_schema_version, bool) or not isinstance(
            self.source_schema_version, int
        ):
            raise ValueError("source schema version must be an integer")
        if not isinstance(self.surface_lighting, SurfaceLightingSettings):
            raise ValueError(
                "flight-plan surface_lighting must be SurfaceLightingSettings"
            )
        render_track = self.render_track
        if render_track is None:
            render_track = RenderTrack.default(digits=self.path.digits)
            object.__setattr__(self, "render_track", render_track)
        if render_track.digits != self.path.digits:
            raise ValueError("camera and render tracks must use the same decimal precision")
        with mp.workdps(self.path.digits):
            if render_track._times[-1] > mp.mpf(self.path.duration_text):
                raise ValueError("render cues must not extend beyond the camera path")

    @property
    def digits(self) -> int:
        return self.path.digits

    @property
    def keyframes(self):
        """Compatibility view for path-oriented UI summaries."""
        return self.path.keyframes

    @property
    def duration_text(self) -> str:
        return self.path.duration_text

    def evaluate(self, time_seconds: str | int | float | mp.mpf) -> EvaluatedFlightFrame:
        with mp.workdps(self.digits):
            if isinstance(time_seconds, float):
                value = mp.mpf(repr(time_seconds))
            else:
                value = mp.mpf(time_seconds)
            if not mp.isfinite(value):
                raise ValueError("flight evaluation time must be finite")
            value = min(mp.mpf(self.path.duration_text), max(mp.mpf("0"), value))
            time_text = mp.nstr(value, n=self.digits, min_fixed=-6, max_fixed=6)
        assert self.render_track is not None
        return EvaluatedFlightFrame(
            time_seconds_text=time_text,
            camera=self.path.evaluate(value),
            scene=self.scene,
            render=self.render_track.evaluate(value),
            digits=self.digits,
        )


FlightSource: TypeAlias = CameraPath | FlightPlanDocument


def flight_path_for(source: FlightSource) -> CameraPath:
    return source.path if isinstance(source, FlightPlanDocument) else source


def surface_lighting_for(
    source: FlightSource,
    fallback: SurfaceLightingSettings | None = None,
) -> SurfaceLightingSettings:
    """Resolve the immutable lighting contract for a complete render session."""

    if isinstance(source, FlightPlanDocument):
        return source.surface_lighting
    if fallback is None:
        return SurfaceLightingSettings()
    if not isinstance(fallback, SurfaceLightingSettings):
        raise ValueError("surface_lighting fallback must be SurfaceLightingSettings")
    return fallback


def evaluate_flight_frame(
    source: FlightSource,
    time_seconds: str | int | float | mp.mpf,
    request_template: RenderRequest,
    *,
    palette: PaletteInput = "inferno",
    cycles: float = 1.0,
) -> EvaluatedFlightFrame:
    """Evaluate a complete plan or adapt a legacy camera-only path."""

    if isinstance(source, FlightPlanDocument):
        return source.evaluate(time_seconds)
    camera = source.evaluate(time_seconds)
    scene = FlightScene(
        request_template.fractal,
        request_template.exponent,
        repr(request_template.julia_c_real),
        repr(request_template.julia_c_imag),
    )
    if isinstance(palette, PaletteBlend):
        palette_state = palette
    else:
        palette_state = PaletteBlend.solid(palette)
    return EvaluatedFlightFrame(
        time_seconds_text=str(time_seconds),
        camera=camera,
        scene=scene,
        render=EvaluatedRenderState(
            request_template.max_iterations,
            request_template.effective_color_iterations,
            request_template.reference_bits,
            palette_state,
            format(float(cycles), ".17g"),
            0,
            None,
        ),
        digits=source.digits,
    )


def flight_plan_fingerprint(source: FlightSource) -> tuple[object, ...]:
    path = flight_path_for(source)
    camera = tuple(
        (
            frame.time_seconds_text,
            frame.camera.center_x_text,
            frame.camera.center_y_text,
            frame.camera.view_width_text,
            frame.easing.value,
            frame.center_interpolation.value,
        )
        for frame in path.keyframes
    )
    if not isinstance(source, FlightPlanDocument):
        return ("camera-path", path.digits, camera)
    assert source.render_track is not None
    render = tuple(
        (
            cue.time_seconds_text,
            cue.profile.max_iterations,
            cue.profile.reference_bits,
            cue.profile.palette,
            cue.profile.cycles_text,
            cue.palette_transition.value,
        )
        for cue in source.render_track.cues
    )
    return (
        "flight-plan",
        path.digits,
        camera,
        source.scene.fractal.value,
        source.scene.exponent,
        source.scene.julia_c_real_text,
        source.scene.julia_c_imag_text,
        render,
        (
            source.surface_lighting.enabled,
            source.surface_lighting.strength,
            source.surface_lighting.azimuth_degrees,
            source.surface_lighting.elevation_degrees,
            source.surface_lighting.ambient,
            source.surface_lighting.diffuse,
        ),
    )
