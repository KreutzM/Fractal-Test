from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .tonemapping import ToneMapState, apply_curve


SURFACE_LIGHTING_SLOPE_SCALE = 4.0


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"surface lighting {name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"surface lighting {name} must be a finite number")


@dataclass(frozen=True, slots=True)
class SurfaceLightingSettings:
    """Screen-space Lambert lighting for a normalized fractal height field.

    Azimuth is measured clockwise in image space: 0 degrees points right and
    90 degrees points down. Elevation is measured above the image plane.
    """

    enabled: bool = False
    strength: float = 1.5
    azimuth_degrees: float = 315.0
    elevation_degrees: float = 45.0
    ambient: float = 0.35
    diffuse: float = 0.65

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("surface lighting enabled must be a boolean")
        _require_finite("strength", self.strength)
        _require_finite("azimuth_degrees", self.azimuth_degrees)
        _require_finite("elevation_degrees", self.elevation_degrees)
        _require_finite("ambient", self.ambient)
        _require_finite("diffuse", self.diffuse)
        if self.strength < 0.0:
            raise ValueError("surface lighting strength must be non-negative")
        if not 0.0 < self.elevation_degrees <= 90.0:
            raise ValueError(
                "surface lighting elevation_degrees must be in the interval (0, 90]"
            )
        if not 0.0 <= self.ambient <= 1.0:
            raise ValueError("surface lighting ambient must be in the interval [0, 1]")
        if not 0.0 <= self.diffuse <= 1.0:
            raise ValueError("surface lighting diffuse must be in the interval [0, 1]")


@dataclass(frozen=True, slots=True)
class SurfaceLightingPreset:
    """Named, deterministic collection of complete lighting settings."""

    name: str
    settings: SurfaceLightingSettings


CUSTOM_SURFACE_LIGHTING_PRESET = "Benutzerdefiniert"
SURFACE_LIGHTING_PRESETS: tuple[SurfaceLightingPreset, ...] = (
    SurfaceLightingPreset(
        "Sanft",
        SurfaceLightingSettings(
            enabled=True,
            strength=1.2,
            azimuth_degrees=315.0,
            elevation_degrees=50.0,
            ambient=0.45,
            diffuse=0.55,
        ),
    ),
    SurfaceLightingPreset(
        "Dramatisch",
        SurfaceLightingSettings(
            enabled=True,
            strength=2.8,
            azimuth_degrees=315.0,
            elevation_degrees=28.0,
            ambient=0.20,
            diffuse=0.80,
        ),
    ),
    SurfaceLightingPreset(
        "Seitenlicht",
        SurfaceLightingSettings(
            enabled=True,
            strength=2.2,
            azimuth_degrees=0.0,
            elevation_degrees=25.0,
            ambient=0.25,
            diffuse=0.75,
        ),
    ),
    SurfaceLightingPreset(
        "Gegenlicht",
        SurfaceLightingSettings(
            enabled=True,
            strength=2.2,
            azimuth_degrees=135.0,
            elevation_degrees=35.0,
            ambient=0.25,
            diffuse=0.75,
        ),
    ),
    SurfaceLightingPreset(
        "Toplicht",
        SurfaceLightingSettings(
            enabled=True,
            strength=1.0,
            azimuth_degrees=270.0,
            elevation_degrees=70.0,
            ambient=0.50,
            diffuse=0.50,
        ),
    ),
)
_SURFACE_LIGHTING_PRESETS_BY_NAME = {
    preset.name: preset for preset in SURFACE_LIGHTING_PRESETS
}


def surface_lighting_preset_names(*, include_custom: bool = True) -> tuple[str, ...]:
    names = tuple(preset.name for preset in SURFACE_LIGHTING_PRESETS)
    if include_custom:
        return (CUSTOM_SURFACE_LIGHTING_PRESET, *names)
    return names


def surface_lighting_settings_for_preset(name: str) -> SurfaceLightingSettings:
    try:
        return _SURFACE_LIGHTING_PRESETS_BY_NAME[name].settings
    except KeyError as exc:
        raise ValueError(f"unknown surface-lighting preset {name!r}") from exc


def surface_lighting_preset_for(settings: SurfaceLightingSettings) -> str:
    if not isinstance(settings, SurfaceLightingSettings):
        raise ValueError("surface-lighting preset matching requires SurfaceLightingSettings")
    for preset in SURFACE_LIGHTING_PRESETS:
        if settings == preset.settings:
            return preset.name
    return CUSTOM_SURFACE_LIGHTING_PRESET


def apply_surface_lighting(
    values: np.ndarray,
    inside: np.ndarray,
    rgb: np.ndarray,
    settings: SurfaceLightingSettings = SurfaceLightingSettings(),
    *,
    tone_state: ToneMapState | None = None,
) -> np.ndarray:
    """Apply deterministic screen-space relief shading to an RGB8 frame.

    ``values`` is converted through the same tone curve used for colorization
    before gradients are evaluated. This keeps relief contrast independent of
    the iteration budget. Inside pixels remain byte-identical to the input
    image. The input arrays are never mutated, including when lighting is
    disabled.
    """

    values_array, inside_array, rgb_array = _validated_inputs(values, inside, rgb)
    if not settings.enabled:
        return rgb_array.copy()

    if tone_state is not None and not isinstance(tone_state, ToneMapState):
        raise ValueError("surface lighting tone_state must be ToneMapState or None")

    height = apply_curve(values_array, tone_state).astype(np.float32, copy=True)
    height[inside_array] = np.float32(0.0)
    gradient_x, gradient_y = _height_gradients(height)

    # User-facing strengths are deliberately compact. The fixed slope scale
    # maps them to useful screen-space normal tilts for a [0, 1] tone-mapped
    # height field without coupling the result to image size or max_iterations.
    strength = np.float32(settings.strength * SURFACE_LIGHTING_SLOPE_SCALE)
    normal_x = -strength * gradient_x
    normal_y = -strength * gradient_y
    inverse_length = np.float32(1.0) / np.sqrt(
        normal_x * normal_x + normal_y * normal_y + np.float32(1.0)
    )

    azimuth = math.radians(settings.azimuth_degrees)
    elevation = math.radians(settings.elevation_degrees)
    horizontal = math.cos(elevation)
    light_x = np.float32(horizontal * math.cos(azimuth))
    light_y = np.float32(horizontal * math.sin(azimuth))
    light_z = np.float32(math.sin(elevation))

    lambert = (
        normal_x * light_x + normal_y * light_y + light_z
    ) * inverse_length
    lambert = np.clip(lambert, 0.0, 1.0)
    ambient = np.float32(settings.ambient)
    diffuse = np.float32(settings.diffuse)
    flat_intensity = ambient + diffuse * light_z
    if flat_intensity <= np.float32(1e-6):
        intensity = np.zeros_like(lambert, dtype=np.float32)
    else:
        # Normalize around the response of a flat surface. Toggling relief no
        # longer changes the global exposure; only actual slopes darken or
        # brighten the image. A bounded 2x highlight ceiling avoids clipping
        # explosions for very low light elevations.
        intensity = np.float32(1.0) + (
            diffuse * (lambert - light_z) / flat_intensity
        )
        intensity = np.clip(intensity, 0.0, 2.0)

    shaded = np.rint(rgb_array.astype(np.float32) * intensity[..., np.newaxis])
    result = np.clip(shaded, 0.0, 255.0).astype(np.uint8)
    result[inside_array] = rgb_array[inside_array]
    return result


def _validated_inputs(
    values: np.ndarray,
    inside: np.ndarray,
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_array = np.asarray(values)
    inside_array = np.asarray(inside)
    rgb_array = np.asarray(rgb)

    if values_array.ndim != 2 or values_array.size == 0:
        raise ValueError("surface lighting values must be a non-empty 2D array")
    if not np.issubdtype(values_array.dtype, np.floating):
        raise ValueError("surface lighting values must use a floating-point dtype")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("surface lighting values must be finite")
    if inside_array.shape != values_array.shape or inside_array.dtype != np.bool_:
        raise ValueError(
            "surface lighting inside mask must be boolean and match values shape"
        )
    expected_rgb_shape = values_array.shape + (3,)
    if rgb_array.shape != expected_rgb_shape or rgb_array.dtype != np.uint8:
        raise ValueError(
            "surface lighting rgb must be uint8 with shape values.shape + (3,)"
        )
    return values_array, inside_array, rgb_array


def _height_gradients(height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gradient_x = np.zeros_like(height, dtype=np.float32)
    gradient_y = np.zeros_like(height, dtype=np.float32)
    rows, columns = height.shape

    if columns > 1:
        gradient_x[:, 0] = height[:, 1] - height[:, 0]
        gradient_x[:, -1] = height[:, -1] - height[:, -2]
        if columns > 2:
            gradient_x[:, 1:-1] = (
                height[:, 2:] - height[:, :-2]
            ) * np.float32(0.5)

    if rows > 1:
        gradient_y[0, :] = height[1, :] - height[0, :]
        gradient_y[-1, :] = height[-1, :] - height[-2, :]
        if rows > 2:
            gradient_y[1:-1, :] = (
                height[2:, :] - height[:-2, :]
            ) * np.float32(0.5)

    return gradient_x, gradient_y
