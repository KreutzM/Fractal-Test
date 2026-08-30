from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .models import RenderRequest
from .renderers import select_renderer
from .surface_lighting import SurfaceLightingSettings


def render_rgb(
    request: RenderRequest,
    backend: str = "auto",
    palette: str = "inferno",
    cycles: float = 1.0,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    tone_state=None,
    tone_scene_key: tuple[Any, ...] | None = None,
    tone_smoothing: float = 0.16,
    surface_lighting: SurfaceLightingSettings | None = None,
):
    renderer = select_renderer(backend)
    result = renderer.render_frame(
        request,
        palette,
        cycles,
        phase,
        tone_mapping,
        tone_state,
        tone_scene_key,
        tone_smoothing,
        surface_lighting=surface_lighting,
    )
    return result.rgb, result


def save_png(
    request: RenderRequest,
    output: str | Path,
    backend: str = "auto",
    palette: str = "inferno",
    cycles: float = 1.0,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    tone_state=None,
    tone_scene_key: tuple[Any, ...] | None = None,
    tone_smoothing: float = 0.16,
    surface_lighting: SurfaceLightingSettings | None = None,
):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb, result = render_rgb(
        request,
        backend,
        palette,
        cycles,
        phase,
        tone_mapping,
        tone_state,
        tone_scene_key,
        tone_smoothing,
        surface_lighting=surface_lighting,
    )
    Image.fromarray(rgb, mode="RGB").save(output_path)
    return result
