from __future__ import annotations

import mpmath as mp
import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import Easing
from fractal_flight_studio.flight_plan import (
    FlightScene,
    PaletteTransition,
    RenderProfile,
)
from fractal_flight_studio.flight_plan_session import FlightPlanSession
from fractal_flight_studio.flight_transition import (
    TransitionMode,
    TransitionSettings,
    TransitionTarget,
    end_render_profile,
    merge_render_cues,
    plan_transition,
)
from fractal_flight_studio.models import FractalKind


SCENE = FlightScene(FractalKind.MANDELBROT, 2)
SOURCE = CameraState("-0.5", "0", "0.1")
SOURCE_PROFILE = RenderProfile(500, 256, "inferno", "1")


def target(
    x: str,
    y: str,
    width: str = "0.01",
    *,
    palette: str = "ocean",
    iterations: int = 1200,
    bits: int = 512,
) -> TransitionTarget:
    return TransitionTarget(
        "Testziel",
        CameraState(x, y, width),
        RenderProfile(iterations, bits, palette, "1.5"),
        SCENE,
    )


def plan_for(
    destination: TransitionTarget,
    mode: TransitionMode = TransitionMode.AUTO,
    *,
    settings: TransitionSettings = TransitionSettings(),
    palette_transition: PaletteTransition = PaletteTransition.BLEND,
):
    return plan_transition(
        SOURCE,
        SOURCE_PROFILE,
        destination,
        start_time_text="5",
        digits=80,
        requested_mode=mode,
        palette_transition=palette_transition,
        settings=settings,
    )


def test_auto_uses_direct_for_near_or_nested_target():
    plan = plan_for(target("-0.505", "0.001", "0.000001"))

    assert plan.mode is TransitionMode.DIRECT
    assert plan.intermediate_keyframe_count == 1
    assert plan.bridge_width_text is None


def test_auto_uses_smallest_bridge_for_separated_targets():
    plan = plan_for(target("0.6", "0", "0.01"))

    assert plan.mode is TransitionMode.BRIDGE
    assert plan.bridge_width_text is not None
    assert mp.mpf("1") < mp.mpf(plan.bridge_width_text) < mp.mpf("3.5")
    assert plan.intermediate_keyframe_count == 3


def test_auto_uses_overview_when_bridge_is_already_near_root_view():
    plan = plan_for(target("2.6", "0", "0.01"))

    assert plan.mode is TransitionMode.OVERVIEW
    assert mp.mpf(plan.bridge_width_text) == mp.mpf("3.5")
    assert plan.intermediate_keyframe_count == 4


def test_bridge_width_accounts_for_vertical_distance_and_aspect_ratio():
    settings = TransitionSettings(aspect_ratio_text="2")
    plan = plan_for(target("-0.5", "0.5", "0.01"), TransitionMode.BRIDGE, settings=settings)

    assert plan.bridge_width_text is not None
    assert mp.mpf(plan.bridge_width_text) > mp.mpf("1")


def test_explicit_direct_does_not_insert_bridge_keyframes():
    plan = plan_for(target("2.6", "0"), TransitionMode.DIRECT)

    assert plan.mode is TransitionMode.DIRECT
    assert len(plan.keyframes) == 3


def test_cut_uses_step_easing_and_hard_palette_change():
    plan = plan_for(target("1", "0"), TransitionMode.CUT)

    assert plan.keyframes[0].easing is Easing.STEP
    assert plan.render_cues[-1].palette_transition is PaletteTransition.CUT
    with mp.workdps(plan.digits):
        cut_duration = (
            mp.mpf(plan.arrival_time_text)
            - mp.mpf(plan.keyframes[0].time_seconds_text)
        )
        expected = mp.mpf(TransitionSettings().cut_seconds_text)
        assert mp.almosteq(cut_duration, expected)


def test_hold_palette_keeps_color_but_still_carries_target_quality():
    plan = plan_for(
        target("0.6", "0", iterations=2400, bits=768),
        palette_transition=PaletteTransition.HOLD,
    )

    cue = plan.render_cues[-1]
    assert cue.palette_transition is PaletteTransition.HOLD
    assert cue.profile.max_iterations == 2400
    assert cue.profile.reference_bits == 768


def test_render_cues_anchor_transition_start_and_target_arrival():
    plan = plan_for(target("0.6", "0"))

    assert plan.render_cues[0].time_seconds_text == plan.start_time_text
    assert plan.render_cues[0].profile == SOURCE_PROFILE
    assert plan.render_cues[-1].time_seconds_text == plan.arrival_time_text
    assert plan.render_cues[-1].palette_transition is PaletteTransition.BLEND


def test_planning_is_deterministic_and_preserves_exact_target_camera():
    destination = target(
        "-0.7436438870371510000000000001",
        "0.1318259042053300000000000002",
        "1e-24",
    )

    first = plan_for(destination)
    second = plan_for(destination)

    assert first == second
    assert first.keyframes[-1].camera == destination.camera
    assert first.keyframes[-2].camera == destination.camera


def test_merge_render_cues_replaces_equal_time_anchor():
    original = plan_for(target("0.6", "0"), TransitionMode.DIRECT)
    replacement_profile = RenderProfile(800, 384, "electric", "2")
    replacement = original.render_cues[0].__class__(
        original.start_time_text,
        replacement_profile,
        PaletteTransition.CUT,
    )

    merged = merge_render_cues(
        original.render_cues,
        (replacement,),
        digits=80,
    )

    assert len(merged) == 2
    assert merged[0].profile == replacement_profile


def test_end_render_profile_uses_effective_palette_after_hold():
    session = FlightPlanSession.new(
        CameraState(),
        digits=80,
        scene=SCENE,
        render_profile=SOURCE_PROFILE,
    )
    state = session.render_track.evaluate("0")

    profile = end_render_profile(render_state=state)
    assert profile.max_iterations == SOURCE_PROFILE.max_iterations
    assert profile.reference_bits == SOURCE_PROFILE.reference_bits
    assert profile.palette == SOURCE_PROFILE.palette
    assert mp.mpf(profile.cycles_text) == mp.mpf(SOURCE_PROFILE.cycles_text)


def test_session_appends_transition_atomically_and_selects_arrival():
    session = FlightPlanSession.new(
        SOURCE,
        digits=80,
        scene=SCENE,
        render_profile=SOURCE_PROFILE,
    )
    plan = plan_transition(
        SOURCE,
        SOURCE_PROFILE,
        target("0.6", "0"),
        start_time_text="0",
        requested_mode=TransitionMode.BRIDGE,
    )

    session.append_transition(plan)

    assert session.valid
    assert session.camera_path is not None
    assert session.camera_path.keyframes[-1].camera == plan.keyframes[-1].camera
    assert session.render_track.cues[-1] == plan.render_cues[-1]
    assert session.dirty
    assert mp.mpf(session.playhead_time_text) == 0
    selected = session.selected_keyframe_index
    assert selected is not None
    assert (
        session.camera_draft.keyframes[selected].time_seconds_text
        == plan.arrival_time_text
    )


def test_session_rejects_transition_that_does_not_start_at_path_end_without_mutation():
    session = FlightPlanSession.new(
        SOURCE,
        digits=80,
        scene=SCENE,
        render_profile=SOURCE_PROFILE,
    )
    before = session.camera_draft
    plan = plan_for(target("0.6", "0"), TransitionMode.BRIDGE)

    with pytest.raises(ValueError, match="current path end"):
        session.append_transition(plan)

    assert session.camera_draft == before
    assert not session.dirty


def test_transition_settings_validate_overview_threshold():
    with pytest.raises(ValueError, match="must not exceed one"):
        TransitionSettings(overview_threshold_text="1.1").values(digits=80)


def test_suggested_free_target_width_preserves_current_scale_by_default():
    from fractal_flight_studio.flight_transition import suggested_target_width

    camera = CameraState("-0.5", "0", "3.5")
    assert suggested_target_width(camera, digits=80) == "3.5"
    assert suggested_target_width(camera, zoom_factor_text="10", digits=80) == "0.35"


def test_suggested_free_target_width_rejects_zoom_out_factor():
    from fractal_flight_studio.flight_transition import suggested_target_width

    with pytest.raises(ValueError, match="at least one"):
        suggested_target_width(
            CameraState("-0.5", "0", "3.5"),
            zoom_factor_text="0.5",
            digits=80,
        )


def test_session_initializes_fresh_plan_from_accepted_transition_source():
    session = FlightPlanSession.new(
        CameraState("-0.5", "0", "3.5"),
        digits=80,
        scene=SCENE,
        render_profile=SOURCE_PROFILE,
    )
    clicked_from = CameraState("-0.72", "0.12", "0.08")
    plan = plan_transition(
        clicked_from,
        SOURCE_PROFILE,
        target("-0.74", "0.13", "0.008"),
        start_time_text="0",
        requested_mode=TransitionMode.DIRECT,
    )

    session.append_transition(plan, initialize_if_needed=True)

    assert session.valid
    assert session.camera_path is not None
    assert session.camera_path.keyframes[0].camera == clicked_from
    assert session.camera_path.keyframes[-1].camera == plan.keyframes[-1].camera
    assert session.render_track.first_profile.max_iterations == SOURCE_PROFILE.max_iterations


def test_fresh_transition_is_not_applied_until_session_accepts_it():
    session = FlightPlanSession.new(
        SOURCE,
        digits=80,
        scene=SCENE,
        render_profile=SOURCE_PROFILE,
    )
    before = session.camera_draft
    plan_transition(
        CameraState("-0.7", "0.1", "0.1"),
        SOURCE_PROFILE,
        target("-0.74", "0.13", "0.01"),
        start_time_text="0",
    )

    assert session.camera_draft == before
    assert not session.dirty
