from __future__ import annotations

import numpy as np

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import CameraPath, FlightKeyframe
from fractal_flight_studio.flight_plan import (
    FlightPlanDocument,
    FlightScene,
    PaletteTransition,
    RenderCue,
    RenderProfile,
    RenderTrack,
)
from fractal_flight_studio.models import FractalKind, RenderRequest
from fractal_flight_studio.offline_render import (
    OfflineRenderSettings,
    build_offline_frame_plan,
    iter_offline_frame_jobs,
)
from fractal_flight_studio.palettes import PaletteBlend, palette_lut


def _path() -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "4")),
            FlightKeyframe("6", CameraState("-0.75", "0.1", "1e-40")),
        ),
        digits=100,
    )


def _document() -> FlightPlanDocument:
    return FlightPlanDocument(
        "Evaluation",
        _path(),
        FlightScene(FractalKind.JULIA, 4, "-0.70176", "-0.3842"),
        RenderTrack(
            (
                RenderCue(
                    "0",
                    RenderProfile(800, 256, "inferno", "1"),
                    PaletteTransition.HOLD,
                ),
                RenderCue(
                    "2",
                    RenderProfile(3000, 768, "ocean", "2"),
                    PaletteTransition.BLEND,
                ),
                RenderCue(
                    "4",
                    RenderProfile(1200, 384, "ember", "1.5"),
                    PaletteTransition.CUT,
                ),
                RenderCue(
                    "6",
                    RenderProfile(900, 256, "electric", "3"),
                    PaletteTransition.HOLD,
                ),
            ),
            digits=100,
        ),
    )


def test_render_track_blends_palette_and_raises_quality_before_deep_target() -> None:
    track = _document().render_track
    assert track is not None

    halfway = track.evaluate("1")
    assert halfway.max_iterations == 3000
    assert halfway.color_iterations == 3000
    assert halfway.reference_bits == 768
    assert halfway.palette == PaletteBlend("inferno", "ocean", 0.5)
    assert halfway.cycles == 1.5

    before_cut = track.evaluate("3")
    assert before_cut.palette == PaletteBlend.solid("ocean")
    assert before_cut.max_iterations == 3000
    assert before_cut.reference_bits == 768

    at_cut = track.evaluate("4")
    assert at_cut.palette == PaletteBlend.solid("ember")
    assert at_cut.max_iterations == 1200
    assert at_cut.color_iterations == 3000
    assert at_cut.reference_bits == 384

    held = track.evaluate("6")
    assert held.palette == PaletteBlend.solid("ember")
    assert held.cycles == 1.5
    assert held.max_iterations == 900
    assert held.color_iterations == 3000
    assert held.reference_bits == 256


def test_document_evaluation_builds_request_from_scene_camera_and_render_track() -> None:
    frame = _document().evaluate("1")
    request = frame.build_request(RenderRequest(width=12, height=8))

    assert request.fractal is FractalKind.JULIA
    assert request.exponent == 4
    assert request.julia_c_real == float("-0.70176")
    assert request.julia_c_imag == float("-0.3842")
    assert request.max_iterations == 3000
    assert request.color_iterations == 3000
    assert request.reference_bits == 768
    assert request.center_x_text == frame.camera.center_x_text
    assert request.view_width_text == frame.camera.view_width_text
    assert request.width == 12 and request.height == 8


def test_offline_jobs_use_exact_time_dependent_render_state() -> None:
    document = _document()
    plan = build_offline_frame_plan(
        document,
        OfflineRenderSettings(width=16, height=10, fps_numerator=1),
    )
    jobs = tuple(
        iter_offline_frame_jobs(
            document,
            RenderRequest(width=99, height=99),
            plan,
            start_index=1,
            stop_index=5,
        )
    )

    assert jobs[0].time_seconds_text == "1.0"
    assert jobs[0].request.max_iterations == 3000
    assert jobs[0].palette == PaletteBlend("inferno", "ocean", 0.5)
    assert jobs[0].cycles == 1.5
    assert jobs[-1].time_seconds_text == "4.0"
    assert jobs[-1].request.max_iterations == 1200
    assert all(job.request.color_iterations == 3000 for job in jobs)
    assert jobs[-1].palette == PaletteBlend.solid("ember")
    assert all(job.request.width == 16 and job.request.height == 10 for job in jobs)


def test_palette_blend_lut_is_deterministic_and_preserves_endpoints() -> None:
    source = palette_lut("inferno", size=32)
    target = palette_lut("ocean", size=32)
    zero = palette_lut(PaletteBlend("inferno", "ocean", 0.0), size=32)
    one = palette_lut(PaletteBlend("inferno", "ocean", 1.0), size=32)
    half = palette_lut(PaletteBlend("inferno", "ocean", 0.5), size=32)

    assert np.array_equal(zero, source)
    assert np.array_equal(one, target)
    expected = np.rint(
        source.astype(np.float32) * 0.5 + target.astype(np.float32) * 0.5
    ).astype(np.uint8)
    assert np.array_equal(half, expected)
    assert np.array_equal(
        half,
        palette_lut(PaletteBlend("inferno", "ocean", 0.5), size=32),
    )


def test_preflight_uses_the_same_per_time_request_and_coloring() -> None:
    from fractal_flight_studio.preflight import PreflightSettings, run_path_preflight
    from fractal_flight_studio.renderers import FrameResult

    class RecordingRenderer:
        name = "recording"

        def __init__(self) -> None:
            self.calls = []

        def render_frame(self, request, palette, cycles, phase, **kwargs):
            self.calls.append((request, palette, cycles, kwargs["tone_scene_key"]))
            y, x = np.indices((request.height, request.width))
            rgb = np.stack(
                ((x * 17 + y) % 256, (x + y * 19) % 256, (x * 7 + y * 3) % 256),
                axis=2,
            ).astype(np.uint8)
            return FrameResult(
                rgb,
                self.name,
                0.01,
                {"pixel_grid_safe": True, "tone_state": object()},
            )

    renderer = RecordingRenderer()
    report = run_path_preflight(
        _document(),
        RenderRequest(),
        renderer,
        PreflightSettings(
            width=18,
            height=12,
            sample_interval_seconds_text="1",
            max_samples=20,
        ),
    )

    assert report.safe
    first_request, first_palette, first_cycles, first_scene_key = renderer.calls[1]
    assert first_request.max_iterations == 3000
    assert first_request.reference_bits == 768
    assert first_palette == PaletteBlend("inferno", "ocean", 0.5)
    assert first_cycles == 1.5
    final_request, final_palette, final_cycles, final_scene_key = renderer.calls[-1]
    assert final_request.max_iterations == 900
    assert final_request.reference_bits == 256
    assert final_palette == PaletteBlend.solid("ember")
    assert final_cycles == 1.5
    assert first_scene_key == final_scene_key


def test_export_fingerprint_includes_scene_and_render_track() -> None:
    from dataclasses import replace

    from fractal_flight_studio.export_controller import (
        FlightExportConfiguration,
        flight_export_fingerprint,
    )

    document = _document()
    base = flight_export_fingerprint(
        document,
        RenderRequest(),
        "fake",
        "inferno",
        1.0,
        FlightExportConfiguration(width=640, height=360),
    )
    changed_track = replace(
        document,
        render_track=RenderTrack(
            (
                *document.render_track.cues[:-1],
                RenderCue(
                    "6",
                    RenderProfile(901, 256, "electric", "3"),
                    PaletteTransition.HOLD,
                ),
            ),
            digits=document.digits,
        ),
    )
    changed = flight_export_fingerprint(
        changed_track,
        RenderRequest(),
        "fake",
        "inferno",
        1.0,
        FlightExportConfiguration(width=640, height=360),
    )

    assert changed != base
