from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import tkinter as tk

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_app import FractalStudioApp
from fractal_flight_studio.flight_path import (
    CameraPath,
    CenterInterpolation,
    FlightKeyframe,
)
from fractal_flight_studio.flight_plan import (
    FlightPlanDocument,
    PaletteTransition,
    RenderCue,
    RenderProfile,
    RenderTrack,
)
from fractal_flight_studio.flight_plan_io import FLIGHT_PLAN_SCHEMA_VERSION, load_flight_plan
from fractal_flight_studio.palettes import PaletteBlend
from fractal_flight_studio.surface_lighting import (
    CUSTOM_SURFACE_LIGHTING_PRESET,
    surface_lighting_preset_names,
    surface_lighting_settings_for_preset,
)


def main() -> int:
    root = tk.Tk()
    app = FractalStudioApp(root)
    assert app.flight_plan_session.render_track.first_profile.palette == "inferno"
    assert app.flight_plan_session.scene.fractal.value == "mandelbrot"
    assert len(app.deep_zoom_targets) == 10
    assert app.deep_zoom_target_var.get() == app.deep_zoom_targets[0].name
    assert app.surface_lighting_enabled_var.get() is False
    assert app.surface_lighting_preset_var.get() == CUSTOM_SURFACE_LIGHTING_PRESET
    assert tuple(app.surface_lighting_preset_combo.cget("values")) == (
        surface_lighting_preset_names()
    )
    app.surface_lighting_preset_var.set("Dramatisch")
    app.apply_surface_lighting_preset()
    lighting = surface_lighting_settings_for_preset("Dramatisch")
    assert app._surface_lighting_settings() == lighting
    assert app.surface_lighting_preset_var.get() == "Dramatisch"
    root.update_idletasks()
    assert app.flight_plan_session.surface_lighting == lighting

    app.set_catalog_flight_target()
    catalog_dialog = app.catalog_transition_dialog
    assert catalog_dialog is not None
    root.update_idletasks()
    assert catalog_dialog.current_plan is not None
    catalog_dialog.destroy()

    app.load_catalog_target_view()
    assert app.camera.center_x_text == app.deep_zoom_targets[0].center_x_text
    app.open_target_browser()
    browser = app.target_browser
    assert browser is not None
    root.update_idletasks()
    assert len(browser._visible_targets) == 10
    browser.search_var.set("spiral")
    root.update_idletasks()
    assert browser._visible_targets
    assert all(
        "spiral" in " ".join((target.name, target.description, *target.tags)).casefold()
        for target in browser._visible_targets
    )
    selected = browser.selected_target
    assert selected is not None
    browser._set_selected_target()
    root.update_idletasks()
    assert app.deep_zoom_target_var.get() == selected.name
    assert app.catalog_transition_dialog is not None
    assert app.catalog_transition_dialog.current_plan is not None
    app.catalog_transition_dialog.destroy()
    browser.destroy()
    root.update_idletasks()

    source_camera = app.camera
    click = type("Click", (), {"x": max(1, app.canvas.winfo_width()) // 2, "y": max(1, app.canvas.winfo_height()) // 2})()
    app._set_flight_target(click)
    free_dialog = app.free_target_dialog
    assert free_dialog is not None
    root.update_idletasks()
    assert free_dialog.current_plan is not None
    assert free_dialog.width_var.get() == source_camera.view_width_text
    assert (
        free_dialog.current_plan.keyframes[-1].camera.view_width_text
        == source_camera.view_width_text
    )
    free_dialog._accept(play=False)
    assert app.flight_plan_session.valid
    assert app.camera_path is not None
    assert app.camera_path.keyframes[0].camera == source_camera
    initial_keyframe_count = len(app.camera_path.keyframes)

    app.open_timeline_editor()
    timeline = app.timeline_editor
    assert timeline is not None
    assert timeline.draft.keyframes[0].center_interpolation is CenterInterpolation.FOCUS
    assert len(timeline.draft.keyframes) == initial_keyframe_count
    assert app.camera_path_dirty
    with TemporaryDirectory() as directory:
        plan_path = Path(directory) / "smoke.fractal-flight.json"
        app.camera_path_name = "GUI smoke"
        assert app._save_flight_plan_path(plan_path)
        assert plan_path.exists()
        saved = load_flight_plan(plan_path)
        assert saved.source_schema_version == FLIGHT_PLAN_SCHEMA_VERSION
        assert saved.render_track.first_profile.palette == app.palette_var.get()
        assert saved.surface_lighting == lighting
        assert not app.camera_path_dirty
        timeline._draft = timeline.draft.remove_keyframe(len(timeline.draft.keyframes) - 1)
        timeline._refresh_tree()
        timeline._publish_path_state()
        assert app.camera_path_dirty
        app.surface_lighting_enabled_var.set(False)
        app.surface_lighting_strength_var.set(0.5)
        root.update_idletasks()
        assert app.surface_lighting_preset_var.get() == CUSTOM_SURFACE_LIGHTING_PRESET
        assert app._load_flight_plan_path(plan_path)
        timeline = app.timeline_editor
        assert timeline is not None
        assert len(timeline.draft.keyframes) == initial_keyframe_count
        assert app.camera_path_file == plan_path
        assert app.camera_path_name == "GUI smoke"
        assert app._surface_lighting_settings() == lighting
        assert app.surface_lighting_preset_var.get() == "Dramatisch"
        assert app.flight_plan_session.surface_lighting == lighting
        assert not app.camera_path_dirty
    timeline.destroy()
    root.update_idletasks()

    app.flight_plan_session.set_camera_path(
        CameraPath(
            (
                FlightKeyframe("0", CameraState("-0.5", "0", "3.5")),
                FlightKeyframe("2", CameraState("-0.75", "0.1", "0.1")),
            )
        ),
        mark_dirty=False,
    )
    app.flight_plan_session.set_render_track(
        RenderTrack(
            (
                RenderCue(
                    "0",
                    RenderProfile(400, 256, "inferno", "1"),
                    PaletteTransition.HOLD,
                ),
                RenderCue(
                    "2",
                    RenderProfile(1200, 512, "ocean", "2"),
                    PaletteTransition.BLEND,
                ),
            )
        ),
        mark_dirty=False,
    )
    app._preview_camera_path(app.camera_path.evaluate("1"), "1")
    preview_request = app._request(0.1)
    assert preview_request.max_iterations == 1200
    assert preview_request.color_iterations == 1200
    assert preview_request.reference_bits == 512
    assert app._render_palette() == PaletteBlend("inferno", "ocean", 0.5)
    assert app._render_cycles() == 1.5

    app.open_export_dialog()
    export_dialog = app.export_dialog
    assert export_dialog is not None
    assert "2 Keyframes" in export_dialog.path_summary_var.get()
    assert "60 Frames" in export_dialog.plan_summary_var.get()
    assert export_dialog.tone_stability_var.get() == "Zeitlich stabilisiert"
    assert "Tone Mapping: Zeitlich stabilisiert" in export_dialog.plan_summary_var.get()
    export_source, _config, export_request, *_rest = export_dialog._context()
    assert isinstance(export_source, FlightPlanDocument)
    assert export_source.surface_lighting == lighting
    assert len(export_source.render_track.cues) == 2
    assert export_request.max_iterations == int(app.iterations_var.get())
    export_dialog.destroy()
    root.update_idletasks()

    assert app.flight_plan_playback.loaded
    app._seek_flight_plan(0.5)
    assert float(app.flight_plan_session.playhead_time_text) == 0.5
    assert app.camera == app.flight_plan_session.build_document().evaluate("0.5").camera
    app._play_flight_plan()
    assert app.flight_plan_playback.playing
    app._pause_flight_plan(request_render=False)
    assert app.flight_plan_playback.paused
    app._stop_flight_plan(request_render=False)
    assert app.flight_plan_playback.state.value == "stopped"
    assert app.flight_plan_playback.playhead_seconds == 0.0

    app.open_timeline_editor()
    timeline = app.timeline_editor
    assert timeline is not None
    timeline._open_transition_dialog()
    transition_dialog = timeline._transition_dialog
    assert transition_dialog is not None
    root.update_idletasks()
    assert transition_dialog.current_plan is not None
    transition_dialog.destroy()
    transition = timeline._append_catalog_transition(app.all_deep_zoom_targets[1])
    assert transition.mode.value in {"direct", "bridge", "overview"}
    assert app.flight_plan_session.valid
    assert len(app.flight_plan_session.render_track.cues) >= 3
    assert app.camera_path is not None
    assert app.camera_path.keyframes[-1].camera == transition.keyframes[-1].camera
    with TemporaryDirectory() as directory:
        assert app._save_flight_plan_path(
            Path(directory) / "transition-smoke.fractal-flight.json"
        )
    timeline.destroy()

    app._on_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
