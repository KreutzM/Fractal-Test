from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_app import FractalStudioApp
from fractal_flight_studio.flight_path import CameraPath, FlightKeyframe
from fractal_flight_studio.flight_plan import FlightPlanDocument
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.offline_render import (
    OfflineRenderSettings,
    build_offline_frame_plan,
    render_offline_frames,
)
from fractal_flight_studio.preflight import PreflightSettings, run_path_preflight
from fractal_flight_studio.renderers import FrameResult
from fractal_flight_studio.surface_lighting import SurfaceLightingSettings
from fractal_flight_studio.temporal_tonemapping import (
    TemporalToneSettings,
    ToneStability,
    analyze_offline_tone_states,
)


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Renderer:
    name = "fake"

    def __init__(self) -> None:
        self.lighting = []

    def render_frame(self, request, *args, surface_lighting=None, **kwargs):
        self.lighting.append(surface_lighting)
        y, x = np.indices((request.height, request.width))
        rgb = np.stack(
            ((x * 17 + y * 3) % 256, (x * 5 + y * 29) % 256, (x * 11 + y * 7) % 256),
            axis=2,
        ).astype(np.uint8)
        return FrameResult(
            rgb,
            self.name,
            0.001,
            {"pixel_grid_safe": True, "tone_state": None},
        )


def _path() -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "3.5")),
            FlightKeyframe("1", CameraState("-0.75", "0.1", "0.1")),
        )
    )


def _lighting() -> SurfaceLightingSettings:
    return SurfaceLightingSettings(
        enabled=True,
        strength=2.25,
        azimuth_degrees=210.0,
        elevation_degrees=52.0,
        ambient=0.2,
        diffuse=0.8,
    )


def test_flight_app_reads_and_applies_complete_document_lighting() -> None:
    app = SimpleNamespace(
        surface_lighting_enabled_var=_Var(False),
        surface_lighting_strength_var=_Var(1.5),
        surface_lighting_azimuth_var=_Var(315.0),
        surface_lighting_elevation_var=_Var(45.0),
        _surface_lighting_ambient=0.35,
        _surface_lighting_diffuse=0.65,
        _applying_flight_plan_settings=False,
        fractal_var=_Var("mandelbrot"),
        exponent_var=_Var(3),
        julia_real_var=_Var(-0.8),
        julia_imag_var=_Var(0.156),
        iterations_var=_Var(400),
        reference_bits_var=_Var(256),
        palette_var=_Var("inferno"),
        cycles_var=_Var(1.0),
    )
    app._apply_surface_lighting_settings = lambda settings: (
        FractalStudioApp._apply_surface_lighting_settings(app, settings)
    )
    document = FlightPlanDocument("Lit", _path(), surface_lighting=_lighting())

    FractalStudioApp._apply_document_primary_settings(app, document)
    actual = FractalStudioApp._surface_lighting_settings(app)

    assert actual == _lighting()
    assert app._applying_flight_plan_settings is False


def test_preflight_uses_document_lighting_for_every_sample() -> None:
    lighting = _lighting()
    document = FlightPlanDocument("Lit", _path(), surface_lighting=lighting)
    renderer = _Renderer()

    report = run_path_preflight(
        document,
        RenderRequest(),
        renderer,
        PreflightSettings(width=32, height=24, sample_interval_seconds_text="0.5"),
        surface_lighting=SurfaceLightingSettings(enabled=False),
    )

    assert report.safe
    assert renderer.lighting == [lighting, lighting, lighting]


def test_offline_render_uses_one_immutable_lighting_value() -> None:
    lighting = _lighting()
    document = FlightPlanDocument("Lit", _path(), surface_lighting=lighting)
    plan = build_offline_frame_plan(
        document,
        OfflineRenderSettings(width=32, height=24, fps_numerator=2),
    )
    renderer = _Renderer()

    frames = tuple(render_offline_frames(document, RenderRequest(), renderer, plan))

    assert len(frames) == 3
    assert renderer.lighting == [lighting, lighting, lighting]


def test_legacy_offline_path_uses_explicit_lighting_fallback() -> None:
    lighting = _lighting()
    path = _path()
    plan = build_offline_frame_plan(
        path,
        OfflineRenderSettings(width=24, height=16, fps_numerator=1),
    )
    renderer = _Renderer()

    tuple(
        render_offline_frames(
            path,
            RenderRequest(),
            renderer,
            plan,
            surface_lighting=lighting,
        )
    )

    assert renderer.lighting == [lighting, lighting]


def test_temporal_tone_analysis_uses_document_lighting() -> None:
    lighting = _lighting()
    document = FlightPlanDocument("Lit", _path(), surface_lighting=lighting)
    plan = build_offline_frame_plan(
        document,
        OfflineRenderSettings(width=64, height=36, fps_numerator=2, append_endpoint=False),
    )
    renderer = _Renderer()

    states = analyze_offline_tone_states(
        document,
        RenderRequest(),
        renderer,
        plan,
        stop_index=2,
        settings=TemporalToneSettings(
            mode=ToneStability.TEMPORAL,
            analysis_width=20,
            analysis_height=12,
        ),
    )

    assert states == (None, None)
    assert renderer.lighting == [lighting, lighting]


def test_mp4_orchestration_passes_document_lighting_to_analysis_and_final_render(
    monkeypatch, tmp_path: Path
) -> None:
    import fractal_flight_studio.mp4_export as module

    lighting = _lighting()
    document = FlightPlanDocument("Lit", _path(), surface_lighting=lighting)
    plan = build_offline_frame_plan(
        document,
        OfflineRenderSettings(width=32, height=24, fps_numerator=2, append_endpoint=False),
    )
    captured = {}

    def fake_analysis(*args, **kwargs):
        captured["analysis"] = kwargs["surface_lighting"]
        return (None, None)

    def fake_frames(*args, **kwargs):
        captured["render"] = kwargs["surface_lighting"]
        return iter((b"frame-0", b"frame-1"))

    sentinel = object()
    monkeypatch.setattr(module, "analyze_offline_tone_states", fake_analysis)
    monkeypatch.setattr(module, "render_offline_frames", fake_frames)
    monkeypatch.setattr(module, "encode_mp4_frames", lambda *args, **kwargs: sentinel)

    result = module.export_path_to_mp4(
        document,
        RenderRequest(),
        _Renderer(),
        plan,
        tmp_path / "flight.mp4",
        temporal_tone=TemporalToneSettings(mode=ToneStability.TEMPORAL),
        surface_lighting=SurfaceLightingSettings(enabled=False),
    )

    assert result is sentinel
    assert captured == {"analysis": lighting, "render": lighting}
