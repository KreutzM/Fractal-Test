from __future__ import annotations

import numpy as np
import pytest

from fractal_flight_studio.models import (
    Precision,
    RenderMode,
    RenderRequest,
    Viewport,
)
from fractal_flight_studio.palettes import tone_mapped_colorize
from fractal_flight_studio.renderers import select_renderer
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.surface_lighting import (
    SurfaceLightingSettings,
    apply_surface_lighting,
)


def _enabled(**changes: float | bool) -> SurfaceLightingSettings:
    defaults: dict[str, float | bool] = {
        "enabled": True,
        "strength": 2.0,
        "azimuth_degrees": 315.0,
        "elevation_degrees": 45.0,
        "ambient": 0.25,
        "diffuse": 0.75,
    }
    defaults.update(changes)
    return SurfaceLightingSettings(**defaults)


def test_cpu_frame_lighting_matches_reference_post_process() -> None:
    request = RenderRequest(width=48, height=32, max_iterations=80)
    settings = _enabled(
        strength=2.4,
        azimuth_degrees=210.0,
        elevation_degrees=52.0,
        ambient=0.3,
        diffuse=0.7,
    )
    renderer = CpuRenderer()
    rendered = renderer.render(request)
    colored, _state, _details = tone_mapped_colorize(
        rendered.values,
        rendered.inside,
        palette="inferno",
        tone_mapping="linear",
    )
    expected = apply_surface_lighting(
        rendered.values, rendered.inside, colored, settings
    )

    actual = renderer.render_frame(
        request,
        palette="inferno",
        tone_mapping="linear",
        surface_lighting=settings,
    )

    assert np.array_equal(actual.rgb, expected)
    assert actual.details["surface_lighting_enabled"] is True
    assert actual.details["optimized_frame_path"] is False


def test_realistic_seahorse_auto_tone_produces_directional_relief() -> None:
    request = RenderRequest(
        width=64,
        height=40,
        viewport=Viewport(-0.74364386269, 0.13182590271, 1.3526e-7),
        max_iterations=1200,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.DIRECT,
    )
    renderer = CpuRenderer()
    baseline = renderer.render_frame(
        request,
        palette="inferno",
        tone_mapping="auto",
        tone_smoothing=1.0,
    )
    tone_state = baseline.details["tone_state"]
    assert tone_state is not None
    scene_key = tone_state.scene_key
    dramatic = _enabled(
        strength=2.8,
        azimuth_degrees=315.0,
        elevation_degrees=28.0,
        ambient=0.2,
        diffuse=0.8,
    )
    opposite = _enabled(
        strength=2.8,
        azimuth_degrees=135.0,
        elevation_degrees=28.0,
        ambient=0.2,
        diffuse=0.8,
    )

    lit = renderer.render_frame(
        request,
        palette="inferno",
        tone_mapping="auto",
        tone_state=tone_state,
        tone_scene_key=scene_key,
        tone_state_locked=True,
        surface_lighting=dramatic,
    )
    reversed_light = renderer.render_frame(
        request,
        palette="inferno",
        tone_mapping="auto",
        tone_state=tone_state,
        tone_scene_key=scene_key,
        tone_state_locked=True,
        surface_lighting=opposite,
    )

    visible = ~np.all(baseline.rgb == 0, axis=2)
    baseline_delta = np.abs(
        lit.rgb.astype(np.int16) - baseline.rgb.astype(np.int16)
    )[visible]
    direction_delta = np.abs(
        lit.rgb.astype(np.int16) - reversed_light.rgb.astype(np.int16)
    )[visible]

    assert float(baseline_delta.mean()) > 20.0
    assert float(direction_delta.mean()) > 35.0
    assert 0.65 < float(lit.rgb[visible].mean() / baseline.rgb[visible].mean()) < 1.15


def test_cpu_frame_disabled_lighting_preserves_existing_output() -> None:
    request = RenderRequest(width=40, height=28, max_iterations=60)

    baseline = CpuRenderer().render_frame(request, tone_mapping="linear")
    disabled = CpuRenderer().render_frame(
        request,
        tone_mapping="linear",
        surface_lighting=SurfaceLightingSettings(enabled=False),
    )

    assert np.array_equal(disabled.rgb, baseline.rgb)
    assert baseline.details["surface_lighting_enabled"] is False
    assert disabled.details["surface_lighting_enabled"] is False
    assert baseline.details["surface_lighting_seconds"] == 0.0
    assert disabled.details["surface_lighting_seconds"] == 0.0


def test_cpu_frame_reports_surface_lighting_settings() -> None:
    request = RenderRequest(width=32, height=24, max_iterations=40)
    settings = _enabled(
        strength=1.25,
        azimuth_degrees=17.0,
        elevation_degrees=64.0,
        ambient=0.4,
        diffuse=0.55,
    )

    result = CpuRenderer().render_frame(
        request, tone_mapping="linear", surface_lighting=settings
    )

    assert result.details["surface_lighting_enabled"] is True
    assert result.details["surface_lighting_seconds"] >= 0.0
    assert result.details["surface_lighting_strength"] == 1.25
    assert result.details["surface_lighting_azimuth_degrees"] == 17.0
    assert result.details["surface_lighting_elevation_degrees"] == 64.0
    assert result.details["surface_lighting_ambient"] == 0.4
    assert result.details["surface_lighting_diffuse"] == 0.55
    assert result.details["surface_lighting_height_source"] == "tone-mapped"
    assert result.details["surface_lighting_slope_scale"] == 4.0
    assert result.details["surface_lighting_flat_neutral"] is True


def test_adaptive_cpu_renderer_forwards_surface_lighting() -> None:
    request = RenderRequest(width=36, height=24, max_iterations=50)

    result = select_renderer("cpu").render_frame(
        request,
        tone_mapping="linear",
        surface_lighting=_enabled(),
    )

    assert result.details["surface_lighting_enabled"] is True
    assert result.details["requested_precision"] == request.precision.value


def test_cpu_frame_rejects_invalid_surface_lighting_argument() -> None:
    request = RenderRequest(width=8, height=8, max_iterations=4)

    with pytest.raises(ValueError, match="SurfaceLightingSettings"):
        CpuRenderer().render_frame(
            request,
            tone_mapping="linear",
            surface_lighting=True,  # type: ignore[arg-type]
        )
