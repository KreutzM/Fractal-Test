from __future__ import annotations

from dataclasses import replace
import math
import time

import numpy as np
from numba import cuda, float32

from ..models import FractalKind, RenderRequest
from ..palettes import PaletteInput
from ..surface_lighting import (
    SURFACE_LIGHTING_SLOPE_SCALE,
    SurfaceLightingSettings,
)
from ..tonemapping import resolve_locked_tone_state, resolve_tone_state, sample_grid
from .base import FrameResult
from .cuda import CudaRenderer as _BaseCudaRenderer

_TONE_MODE_CODES = {"linear": 0, "asinh": 1, "auto": 2}


@cuda.jit
def _sample_values_kernel(values, inside, sample_values, sample_valid, grid_x, grid_y):
    index = cuda.grid(1)
    count = grid_x * grid_y
    if index >= count:
        return
    sx = index % grid_x
    sy = index // grid_x
    width = values.shape[1]
    height = values.shape[0]
    px = int((float32(sx) + float32(0.5)) * float32(width) / float32(grid_x))
    py = int((float32(sy) + float32(0.5)) * float32(height) / float32(grid_y))
    if px >= width:
        px = width - 1
    if py >= height:
        py = height - 1
    value = values[py, px]
    if inside[py, px] or not math.isfinite(value):
        sample_values[index] = float32(0.0)
        sample_valid[index] = False
    else:
        sample_values[index] = value
        sample_valid[index] = True


@cuda.jit(device=True, inline=True)
def _tone_mapped_value(value, mode_code, low, high, strength, gamma):
    window = high - low
    if window <= float32(1e-20):
        mapped = value
    else:
        mapped = (value - low) / window
    if mapped < float32(0.0):
        mapped = float32(0.0)
    elif mapped > float32(1.0):
        mapped = float32(1.0)

    if mode_code != 0:
        denominator = math.log(strength + math.sqrt(strength * strength + float32(1.0)))
        numerator_value = strength * mapped
        numerator = math.log(
            numerator_value + math.sqrt(numerator_value * numerator_value + float32(1.0))
        )
        if denominator > float32(0.0):
            mapped = numerator / denominator
        if mode_code == 2 and gamma != float32(1.0):
            mapped = math.pow(mapped, gamma)
    return mapped


@cuda.jit(device=True, inline=True)
def _surface_height(values, inside, px, py):
    if inside[py, px]:
        return float32(0.0)
    value = values[py, px]
    if value < float32(0.0):
        return float32(0.0)
    if value > float32(1.0):
        return float32(1.0)
    return value


@cuda.jit(device=True, inline=True)
def _round_to_u8(value):
    if value <= float32(0.0):
        return 0
    if value >= float32(255.0):
        return 255
    lower = math.floor(value)
    fraction = value - float32(lower)
    if fraction > float32(0.5):
        return int(lower) + 1
    if fraction < float32(0.5):
        return int(lower)
    lower_int = int(lower)
    return lower_int + (lower_int & 1)


@cuda.jit
def _surface_lighting_kernel(
    values,
    inside,
    rgb,
    lighting_strength,
    light_x,
    light_y,
    light_z,
    ambient,
    diffuse,
):
    px, py = cuda.grid(2)
    height = values.shape[0]
    width = values.shape[1]
    if px >= width or py >= height or inside[py, px]:
        return

    if width <= 1:
        gradient_x = float32(0.0)
    elif px == 0:
        gradient_x = _surface_height(values, inside, 1, py) - _surface_height(
            values, inside, 0, py
        )
    elif px == width - 1:
        gradient_x = _surface_height(
            values, inside, width - 1, py
        ) - _surface_height(values, inside, width - 2, py)
    else:
        gradient_x = float32(0.5) * (
            _surface_height(values, inside, px + 1, py)
            - _surface_height(values, inside, px - 1, py)
        )

    if height <= 1:
        gradient_y = float32(0.0)
    elif py == 0:
        gradient_y = _surface_height(values, inside, px, 1) - _surface_height(
            values, inside, px, 0
        )
    elif py == height - 1:
        gradient_y = _surface_height(
            values, inside, px, height - 1
        ) - _surface_height(values, inside, px, height - 2)
    else:
        gradient_y = float32(0.5) * (
            _surface_height(values, inside, px, py + 1)
            - _surface_height(values, inside, px, py - 1)
        )

    normal_x = -lighting_strength * gradient_x
    normal_y = -lighting_strength * gradient_y
    inverse_length = float32(1.0) / math.sqrt(
        normal_x * normal_x + normal_y * normal_y + float32(1.0)
    )
    lambert = (
        normal_x * light_x + normal_y * light_y + light_z
    ) * inverse_length
    if lambert < float32(0.0):
        lambert = float32(0.0)
    elif lambert > float32(1.0):
        lambert = float32(1.0)
    flat_intensity = ambient + diffuse * light_z
    if flat_intensity <= float32(1e-6):
        intensity = float32(0.0)
    else:
        intensity = float32(1.0) + diffuse * (lambert - light_z) / flat_intensity
        if intensity < float32(0.0):
            intensity = float32(0.0)
        elif intensity > float32(2.0):
            intensity = float32(2.0)

    rgb[py, px, 0] = _round_to_u8(float32(rgb[py, px, 0]) * intensity)
    rgb[py, px, 1] = _round_to_u8(float32(rgb[py, px, 1]) * intensity)
    rgb[py, px, 2] = _round_to_u8(float32(rgb[py, px, 2]) * intensity)


@cuda.jit
def _tone_colorize_kernel(
    values,
    inside,
    rgb,
    lut,
    cycles,
    phase,
    mode_code,
    low,
    high,
    strength,
    gamma,
):
    px, py = cuda.grid(2)
    height = values.shape[0]
    width = values.shape[1]
    if px >= width or py >= height:
        return
    if inside[py, px]:
        values[py, px] = float32(0.0)
        rgb[py, px, 0] = 0
        rgb[py, px, 1] = 0
        rgb[py, px, 2] = 0
        return

    mapped = _tone_mapped_value(
        values[py, px], mode_code, low, high, strength, gamma
    )

    # The fractal kernel overwrites this buffer on the next frame. Reusing it
    # for the tone-mapped height avoids another full-size GPU allocation and
    # lets the lighting kernel sample neighboring heights without repeating the
    # expensive tone curve for every neighbor.
    values[py, px] = mapped

    wrapped = mapped * cycles + phase
    wrapped = wrapped - math.floor(wrapped)
    index = int(wrapped * float32(lut.shape[0] - 1))
    if index < 0:
        index = 0
    elif index >= lut.shape[0]:
        index = lut.shape[0] - 1
    rgb[py, px, 0] = lut[index, 0]
    rgb[py, px, 1] = lut[index, 1]
    rgb[py, px, 2] = lut[index, 2]


class CudaRenderer(_BaseCudaRenderer):
    """CUDA renderer with automatic tone analysis and GPU-side colorization."""

    @staticmethod
    def _stable_color_request(request: RenderRequest) -> RenderRequest:
        """Use the flight-wide color scale as CUDA's conservative loop limit.

        The legacy CUDA kernels normalize escaped values by their loop limit.
        Raising that limit only for frames near a deep target would therefore
        recolor all previously escaped pixels. Until those kernels expose a
        separate color scale, evaluating the global scale is the correct and
        deterministic fallback.
        """

        if request.max_iterations == request.effective_color_iterations:
            return request
        return replace(
            request,
            max_iterations=request.effective_color_iterations,
            color_iterations=request.effective_color_iterations,
        )

    def __init__(self) -> None:
        super().__init__()
        self._sample_values_device = None
        self._sample_valid_device = None
        self._sample_values_host = None
        self._sample_valid_host = None
        self._automatic_tone_state = None
        self._automatic_tone_scene_key = None

    def _ensure_tone_buffers(self) -> None:
        if self._sample_values_device is not None:
            return
        self._sample_values_device = cuda.device_array(
            4096, dtype=np.float32, stream=self._stream
        )
        self._sample_valid_device = cuda.device_array(
            4096, dtype=np.bool_, stream=self._stream
        )
        self._sample_values_host = cuda.pinned_array(4096, dtype=np.float32)
        self._sample_valid_host = cuda.pinned_array(4096, dtype=np.bool_)

    @staticmethod
    def _default_scene_key(request: RenderRequest, tone_mapping: str):
        return (
            request.fractal.value,
            request.precision.value,
            request.render_mode.value,
            request.effective_color_iterations,
            request.exponent,
            request.julia_c_real,
            request.julia_c_imag,
            tone_mapping,
        )

    def render(self, request: RenderRequest):
        return super().render(self._stable_color_request(request))

    def render_frame(
        self,
        request: RenderRequest,
        palette: PaletteInput = "inferno",
        cycles: float = 1.0,
        phase: float = 0.0,
        tone_mapping: str = "auto",
        tone_state=None,
        tone_scene_key=None,
        tone_smoothing: float = 0.16,
        tone_state_locked: bool = False,
        surface_lighting: SurfaceLightingSettings | None = None,
    ) -> FrameResult:
        request = self._stable_color_request(request)
        request.validate()
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if tone_mapping not in _TONE_MODE_CODES:
            raise ValueError(f"unknown tone mapping mode: {tone_mapping}")
        if surface_lighting is not None and not isinstance(
            surface_lighting, SurfaceLightingSettings
        ):
            raise ValueError(
                "surface_lighting must be SurfaceLightingSettings or None"
            )
        if not self.is_available():
            raise RuntimeError("CUDA is not available")

        effective_mode = (
            "linear"
            if tone_mapping == "auto" and request.fractal is FractalKind.NEWTON
            else tone_mapping
        )
        implicit_state = (
            not tone_state_locked and tone_state is None and tone_scene_key is None
        )
        if implicit_state:
            tone_scene_key = self._default_scene_key(request, effective_mode)
            if tone_scene_key == self._automatic_tone_scene_key:
                tone_state = self._automatic_tone_state

        allocation_seconds = self._ensure_buffers(request.height, request.width)
        if not tone_state_locked and effective_mode != "linear":
            self._ensure_tone_buffers()
        palette_seconds = self._ensure_palette(palette)
        threads = (16, 16)
        blocks = (
            (request.width + threads[0] - 1) // threads[0],
            (request.height + threads[1] - 1) // threads[1],
        )
        from ..deep_zoom import should_use_perturbation

        perturb = self._reference_cache.prepare(request) if should_use_perturbation(request) else None

        started = time.perf_counter()
        orbit_upload_seconds = self._launch_fractal(request, blocks, threads, perturb)
        tone_analysis_started = time.perf_counter()

        if tone_state_locked:
            next_state, tone_details = resolve_locked_tone_state(
                effective_mode, tone_state, tone_scene_key
            )
        elif effective_mode == "linear":
            next_state = None
            tone_details = {
                "tone_mapping": "linear",
                "tone_low": 0.0,
                "tone_high": 1.0,
                "tone_strength": 1.0,
                "tone_gamma": 1.0,
                "tone_scene_reset": tone_state is not None,
                "tone_sample_count": 0,
                "tone_state_locked": False,
            }
        else:
            grid_x, grid_y = sample_grid(request.width, request.height, 4096)
            sample_count = grid_x * grid_y
            sample_threads = 128
            sample_blocks = (sample_count + sample_threads - 1) // sample_threads
            _sample_values_kernel[sample_blocks, sample_threads, self._stream](
                self._values_device,
                self._inside_device,
                self._sample_values_device,
                self._sample_valid_device,
                grid_x,
                grid_y,
            )
            self._sample_values_device.copy_to_host(
                self._sample_values_host, stream=self._stream
            )
            self._sample_valid_device.copy_to_host(
                self._sample_valid_host, stream=self._stream
            )
            self._stream.synchronize()
            valid = self._sample_valid_host[:sample_count]
            samples = np.asarray(
                self._sample_values_host[:sample_count][valid], dtype=np.float64
            )
            next_state, tone_details = resolve_tone_state(
                samples,
                effective_mode,
                tone_state,
                tone_scene_key,
                tone_smoothing,
            )
            tone_details["tone_state_locked"] = False

        if implicit_state:
            self._automatic_tone_state = next_state
            self._automatic_tone_scene_key = tone_scene_key

        tone_analysis_seconds = time.perf_counter() - tone_analysis_started
        if next_state is None:
            low, high, strength, gamma = (
                np.float32(x) for x in (0.0, 1.0, 1.0, 1.0)
            )
        else:
            low = np.float32(next_state.low)
            high = np.float32(next_state.high)
            strength = np.float32(next_state.strength)
            gamma = np.float32(next_state.gamma)

        _tone_colorize_kernel[blocks, threads, self._stream](
            self._values_device,
            self._inside_device,
            self._rgb_device,
            self._palette_device,
            np.float32(cycles),
            np.float32(phase),
            _TONE_MODE_CODES[effective_mode],
            low,
            high,
            strength,
            gamma,
        )

        lighting_enabled = bool(
            surface_lighting is not None and surface_lighting.enabled
        )
        lighting_seconds = 0.0
        if lighting_enabled and surface_lighting is not None:
            azimuth = math.radians(surface_lighting.azimuth_degrees)
            elevation = math.radians(surface_lighting.elevation_degrees)
            horizontal = math.cos(elevation)
            lighting_started = time.perf_counter()
            _surface_lighting_kernel[blocks, threads, self._stream](
                self._values_device,
                self._inside_device,
                self._rgb_device,
                np.float32(surface_lighting.strength * SURFACE_LIGHTING_SLOPE_SCALE),
                np.float32(horizontal * math.cos(azimuth)),
                np.float32(horizontal * math.sin(azimuth)),
                np.float32(math.sin(elevation)),
                np.float32(surface_lighting.ambient),
                np.float32(surface_lighting.diffuse),
            )
            lighting_seconds = time.perf_counter() - lighting_started

        self._rgb_device.copy_to_host(self._rgb_host, stream=self._stream)
        self._stream.synchronize()
        rgb = np.array(self._rgb_host, copy=True)
        elapsed = time.perf_counter() - started

        details = {
            "precision": request.precision.value,
            "device": self._device_name(),
            "allocation_seconds": allocation_seconds,
            "palette_upload_seconds": palette_seconds,
            "persistent_buffers": True,
            "optimized_frame_path": True,
            "transfer": (
                "stratified tone sample + single RGB readback"
                if effective_mode != "linear" and not tone_state_locked
                else "single RGB readback"
            ),
            "render_mode": "perturbation" if perturb is not None else "direct",
            "reference_bits": perturb.reference_bits if perturb is not None else 0,
            "reference_rebase_limit": perturb.reference_rebase_limit if perturb is not None else 0,
            "reference_upload_seconds": orbit_upload_seconds,
            "reference_reused": perturb.reference_reused if perturb is not None else False,
            "reference_anchor_x": perturb.reference_anchor_x_text if perturb is not None else "",
            "reference_anchor_y": perturb.reference_anchor_y_text if perturb is not None else "",
            "rebasing_enabled": perturb is not None,
            "glitch_detection_enabled": perturb is not None,
            "tone_state": next_state,
            "tone_mapping_requested": tone_mapping,
            "tone_analysis_seconds": tone_analysis_seconds,
            "surface_lighting_enabled": lighting_enabled,
            "surface_lighting_seconds": lighting_seconds,
        }
        if lighting_enabled and surface_lighting is not None:
            details.update(
                {
                    "surface_lighting_strength": surface_lighting.strength,
                    "surface_lighting_azimuth_degrees": (
                        surface_lighting.azimuth_degrees
                    ),
                    "surface_lighting_elevation_degrees": (
                        surface_lighting.elevation_degrees
                    ),
                    "surface_lighting_ambient": surface_lighting.ambient,
                    "surface_lighting_diffuse": surface_lighting.diffuse,
                    "surface_lighting_height_source": "tone-mapped",
                    "surface_lighting_slope_scale": SURFACE_LIGHTING_SLOPE_SCALE,
                    "surface_lighting_flat_neutral": True,
                    "surface_lighting_timing_scope": "host kernel launch",
                }
            )
        details.update(tone_details)
        return FrameResult(rgb=rgb, backend=self.name, elapsed_seconds=elapsed, details=details)
