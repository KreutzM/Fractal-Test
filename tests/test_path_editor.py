from __future__ import annotations

import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import (
    CameraPath,
    CenterInterpolation,
    Easing,
    FlightKeyframe,
)
from fractal_flight_studio.path_editor import CameraPathDraft


def _camera(x: str, width: str = "1") -> CameraState:
    return CameraState(x, "0", width)


def test_empty_draft_suggests_zero_and_is_not_valid():
    draft = CameraPathDraft()

    assert draft.suggested_time_text() == "0"
    assert not draft.valid
    assert "zwei" in draft.validation_error.lower()


def test_add_keyframes_sorts_exact_times_and_builds_path():
    draft = CameraPathDraft()
    draft = draft.add_keyframe("10", _camera("10"), Easing.LINEAR)
    draft = draft.add_keyframe(
        "0",
        _camera("0"),
        Easing.SMOOTHERSTEP,
        CenterInterpolation.FOCUS,
    )
    draft = draft.add_keyframe("2.5", _camera("2.5"), Easing.SMOOTHSTEP)

    path = draft.build_path()

    assert tuple(frame.time_seconds_text for frame in path.keyframes) == ("0", "2.5", "10")
    assert path.duration_text == "10"
    assert path.keyframes[0].easing is Easing.SMOOTHERSTEP
    assert path.keyframes[0].center_interpolation is CenterInterpolation.FOCUS


def test_duplicate_time_is_rejected_unless_replaced():
    draft = CameraPathDraft().add_keyframe("0", _camera("0"))

    with pytest.raises(ValueError, match="existiert bereits"):
        draft.add_keyframe("0.0", _camera("1"))

    replaced = draft.add_keyframe("0.0", _camera("1"), replace_existing=True)
    assert replaced.keyframes[0].camera.center_x_text == "1"


def test_update_keyframe_reorders_without_losing_exact_camera_text():
    deep = CameraState("-0.7436438870371510000000000001", "0.13182590420533", "1e-420")
    draft = CameraPathDraft().add_keyframe("0", _camera("0"))
    draft = draft.add_keyframe("10", _camera("10"))
    draft = draft.add_keyframe("20", deep, Easing.LINEAR)

    updated = draft.update_keyframe(
        2,
        time_seconds_text="5",
        easing=Easing.SMOOTHERSTEP,
        center_interpolation=CenterInterpolation.FOCUS,
    )

    assert tuple(frame.time_seconds_text for frame in updated.keyframes) == ("0", "5", "10")
    assert updated.keyframes[1].camera is deep
    assert updated.keyframes[1].camera.view_width_text == "1e-420"
    assert updated.keyframes[1].easing is Easing.SMOOTHERSTEP
    assert updated.keyframes[1].center_interpolation is CenterInterpolation.FOCUS


def test_update_rejects_time_collision():
    draft = CameraPathDraft().add_keyframe("0", _camera("0"))
    draft = draft.add_keyframe("5", _camera("5"))

    with pytest.raises(ValueError, match="existiert bereits"):
        draft.update_keyframe(1, time_seconds_text="0.000")


def test_remove_can_leave_partial_draft_but_not_buildable_path():
    draft = CameraPathDraft().add_keyframe("0", _camera("0"))
    draft = draft.add_keyframe("5", _camera("5"))

    partial = draft.remove_keyframe(1)

    assert len(partial.keyframes) == 1
    assert not partial.valid
    with pytest.raises(ValueError, match="zwei"):
        partial.build_path()


def test_first_keyframe_must_start_at_zero_only_when_building():
    draft = CameraPathDraft().add_keyframe("1", _camera("1"))
    draft = draft.add_keyframe("2", _camera("2"))

    assert not draft.valid
    assert "0 Sekunden" in draft.validation_error
    with pytest.raises(ValueError, match="0 Sekunden"):
        draft.build_path()


def test_suggested_time_uses_exact_decimal_arithmetic():
    draft = CameraPathDraft().add_keyframe("0", _camera("0"))
    draft = draft.add_keyframe("0.1", _camera("1"))

    assert draft.suggested_time_text(step_text="0.2") == "0.3"


def test_from_path_preserves_frames_precision_and_interpolation():
    path = CameraPath(
        (
            FlightKeyframe("0", _camera("0", "1"), Easing.LINEAR),
            FlightKeyframe("4", _camera("4", "0.01")),
        ),
        digits=96,
    )

    draft = CameraPathDraft.from_path(path)
    rebuilt = draft.build_path()

    assert draft.digits == 96
    assert rebuilt.keyframes == path.keyframes
    assert rebuilt.evaluate("2") == path.evaluate("2")


def test_clear_preserves_precision():
    draft = CameraPathDraft(digits=120).add_keyframe("0", _camera("0"))

    cleared = draft.clear()

    assert cleared.keyframes == ()
    assert cleared.digits == 120
