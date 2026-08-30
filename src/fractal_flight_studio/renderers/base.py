from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from ..models import RenderRequest
from ..palettes import PaletteInput
from ..surface_lighting import (
    SURFACE_LIGHTING_SLOPE_SCALE,
    SurfaceLightingSettings,
    apply_surface_lighting,
)


@dataclass(slots=True)
class RenderResult:
    values: np.ndarray
    inside: np.ndarray
    backend: str
    elapsed_seconds: float
    details: dict[str, Any]


@dataclass(slots=True)
class FrameResult:
    rgb: np.ndarray
    backend: str
    elapsed_seconds: float
    details: dict[str, Any]


class Renderer(ABC):
    name: str
    _automatic_tone_state = None
    _automatic_tone_scene_key = None

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def render(self, request: RenderRequest) -> RenderResult:
        raise NotImplementedError

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
        """Render a display-ready RGB frame.

        Backends may override this to keep post-processing on the accelerator and
        avoid transferring intermediate arrays to the host.
        """
        from ..palettes import tone_mapped_colorize

        if surface_lighting is not None and not isinstance(
            surface_lighting, SurfaceLightingSettings
        ):
            raise ValueError(
                "surface_lighting must be SurfaceLightingSettings or None"
            )

        effective_tone_mapping = tone_mapping
        if tone_mapping == "auto" and request.fractal.value == "newton":
            effective_tone_mapping = "linear"

        implicit_state = (
            not tone_state_locked and tone_state is None and tone_scene_key is None
        )
        if implicit_state:
            tone_scene_key = (
                request.fractal.value,
                request.precision.value,
                request.render_mode.value,
                request.effective_color_iterations,
                request.exponent,
                request.julia_c_real,
                request.julia_c_imag,
                effective_tone_mapping,
            )
            if tone_scene_key == self._automatic_tone_scene_key:
                tone_state = self._automatic_tone_state

        started = time.perf_counter()
        result = self.render(request)
        color_started = time.perf_counter()
        rgb, next_tone_state, tone_details = tone_mapped_colorize(
            result.values,
            result.inside,
            palette,
            cycles,
            phase,
            effective_tone_mapping,
            tone_state,
            tone_scene_key,
            tone_smoothing,
            tone_state_locked,
        )
        color_seconds = time.perf_counter() - color_started

        lighting_enabled = bool(
            surface_lighting is not None and surface_lighting.enabled
        )
        lighting_seconds = 0.0
        if lighting_enabled:
            lighting_started = time.perf_counter()
            rgb = apply_surface_lighting(
                result.values,
                result.inside,
                rgb,
                surface_lighting,
                tone_state=next_tone_state,
            )
            lighting_seconds = time.perf_counter() - lighting_started

        if implicit_state:
            self._automatic_tone_state = next_tone_state
            self._automatic_tone_scene_key = tone_scene_key

        details = dict(result.details)
        details.update(
            {
                "compute_seconds": result.elapsed_seconds,
                "color_seconds": color_seconds,
                "surface_lighting_seconds": lighting_seconds,
                "surface_lighting_enabled": lighting_enabled,
                "transfer_seconds": details.get("transfer_seconds", 0.0),
                "optimized_frame_path": False,
                "tone_state": next_tone_state,
            }
        )
        details.update(tone_details)
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
                }
            )
        details["tone_mapping_requested"] = tone_mapping
        return FrameResult(
            rgb=rgb,
            backend=result.backend,
            elapsed_seconds=time.perf_counter() - started,
            details=details,
        )
