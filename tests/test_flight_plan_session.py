from __future__ import annotations

from pathlib import Path

import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import (
    CameraPath,
    CenterInterpolation,
    FlightKeyframe,
)
from fractal_flight_studio.flight_plan import (
    FlightPlanDocument,
    FlightScene,
    PaletteTransition,
    RenderCue,
    RenderProfile,
    RenderTrack,
)
from fractal_flight_studio.flight_plan_session import FlightPlanSession
from fractal_flight_studio.models import FractalKind
from fractal_flight_studio.surface_lighting import SurfaceLightingSettings


def _path(*, digits: int = 96) -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe(
                "0",
                CameraState("-0.5", "0", "3.5"),
                center_interpolation=CenterInterpolation.FOCUS,
            ),
            FlightKeyframe("5", CameraState("-0.75", "0.1", "1e-30")),
        ),
        digits=digits,
    )


def test_new_session_owns_partial_draft_until_second_keyframe_is_added() -> None:
    session = FlightPlanSession.new(
        CameraState("-0.5", "0", "3.5"),
        digits=96,
        scene=FlightScene(FractalKind.MANDELBROT, 2),
        render_profile=RenderProfile(900, 384, "inferno", "1.5"),
        surface_lighting=SurfaceLightingSettings(enabled=True, strength=1.9),
    )
    notifications: list[bool] = []
    session.add_listener(lambda current: notifications.append(current.dirty))

    assert session.camera_path is None
    assert not session.valid
    assert "zwei" in session.validation_error.lower()
    assert session.render_track.first_profile.max_iterations == 900
    assert session.surface_lighting.enabled

    session.set_camera_draft(
        session.camera_draft.add_keyframe(
            "5",
            CameraState("-0.75", "0.1", "1e-30"),
        )
    )

    assert session.valid
    assert session.camera_path == _path()
    assert session.dirty
    assert notifications == [True]


def test_session_loads_document_and_centralizes_file_dirty_selection_and_playhead() -> None:
    document = FlightPlanDocument(
        "Loaded",
        _path(),
        FlightScene(FractalKind.MULTIBROT, 4, "-0.7", "0.2"),
        RenderTrack(
            (
                RenderCue("0", RenderProfile(600, 256, "inferno", "1")),
                RenderCue(
                    "2.5",
                    RenderProfile(2200, 640, "ocean", "2"),
                    PaletteTransition.BLEND,
                ),
            ),
            digits=96,
        ),
        SurfaceLightingSettings(enabled=True, azimuth_degrees=120.0),
    )
    session = FlightPlanSession.new(CameraState(), digits=96)
    path = Path("loaded.fractal-flight.json")

    session.set_document(document, file_path=path)
    session.set_selected_keyframe(1)
    session.set_playhead("2.5")

    assert session.camera_path == document.path
    assert session.scene == document.scene
    assert session.render_track == document.render_track
    assert session.surface_lighting == document.surface_lighting
    assert session.name == "Loaded"
    assert session.file_path == path
    assert not session.dirty
    assert session.selected_keyframe_index == 1
    assert session.playhead_time_text == "2.5"


def test_sync_primary_settings_updates_scene_and_first_cue_without_losing_later_cues() -> None:
    track = RenderTrack(
        (
            RenderCue("0", RenderProfile(400, 256, "inferno", "1")),
            RenderCue(
                "3",
                RenderProfile(3000, 768, "ocean", "2"),
                PaletteTransition.BLEND,
            ),
        ),
        digits=96,
    )
    session = FlightPlanSession(
        camera_draft=FlightPlanSession.new(CameraState(), digits=96).camera_draft,
        render_track=track,
    )
    session.set_camera_path(_path(), mark_dirty=False)
    new_scene = FlightScene(FractalKind.JULIA, 3, "-0.4", "0.6")
    new_profile = RenderProfile(1200, 512, "ember", "1.75")
    new_lighting = SurfaceLightingSettings(enabled=True, strength=2.1)

    session.sync_primary_settings(new_scene, new_profile, new_lighting)

    assert session.scene == new_scene
    assert session.render_track.cues[0].profile == new_profile
    assert session.render_track.cues[1] == track.cues[1]
    assert session.surface_lighting == new_lighting
    assert session.dirty


def test_mark_saved_uses_valid_document_and_can_rename_atomically() -> None:
    session = FlightPlanSession.new(CameraState(), digits=96)
    session.set_camera_path(_path())

    session.mark_saved(Path("named.fractal-flight.json"), name="Named")

    assert session.name == "Named"
    assert session.file_path == Path("named.fractal-flight.json")
    assert not session.dirty
    assert session.build_document().name == "Named"


def test_render_track_precision_and_duration_participate_in_session_validation() -> None:
    session = FlightPlanSession.new(CameraState(), digits=96)
    session.set_camera_path(_path(), mark_dirty=False)

    with pytest.raises(ValueError, match="same precision"):
        session.set_render_track(RenderTrack.default(digits=80))

    session.set_render_track(
        RenderTrack(
            (
                RenderCue("0", RenderProfile()),
                RenderCue("6", RenderProfile(1000, 512, "ember", "1")),
            ),
            digits=96,
        )
    )

    assert not session.valid
    assert "beyond" in session.validation_error
    with pytest.raises(ValueError, match="beyond"):
        session.build_document()
