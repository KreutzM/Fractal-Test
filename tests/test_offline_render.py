from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import CameraPath, Easing, FlightKeyframe
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.offline_render import (
    OfflineFrameRenderError,
    OfflineRenderSettings,
    build_offline_frame_plan,
    iter_offline_frame_jobs,
    render_offline_frame,
    render_offline_frames,
)


def _path(duration: str = "1") -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "4"), Easing.LINEAR),
            FlightKeyframe(duration, CameraState("-0.75", "0.1", "4e-40")),
        ),
        digits=100,
    )


@dataclass
class _Frame:
    rgb: np.ndarray
    backend: str = "fake"
    elapsed_seconds: float = 0.25
    details: dict | None = None

    def __post_init__(self):
        if self.details is None:
            self.details = {"render_mode": "direct", "nested": {"ignored": True}}


class _Renderer:
    name = "fake"

    def __init__(self):
        self.calls = []
        self.buffer = None

    def render_frame(self, request, *args, **kwargs):
        self.calls.append((request, args, kwargs))
        if self.buffer is None or self.buffer.shape[:2] != (
            request.height,
            request.width,
        ):
            self.buffer = np.full(
                (request.height, request.width, 3), 17, dtype=np.uint8
            )
        return _Frame(
            self.buffer,
            details={"frame": len(self.calls), "tone_state": object()},
        )


def test_plan_uses_exact_cadence_and_includes_aligned_endpoint():
    plan = build_offline_frame_plan(
        _path("1"), OfflineRenderSettings(width=80, height=50, fps_numerator=2)
    )
    assert plan.frame_count == 3
    assert plan.endpoint_included
    assert not plan.endpoint_appended
    assert [plan.time_seconds_text(i) for i in range(3)] == ["0.0", "0.5", "1"]


def test_plan_can_append_or_omit_a_non_cadence_endpoint():
    included = build_offline_frame_plan(
        _path("1.1"),
        OfflineRenderSettings(fps_numerator=2, append_endpoint=True),
    )
    omitted = build_offline_frame_plan(
        _path("1.1"),
        OfflineRenderSettings(fps_numerator=2, append_endpoint=False),
    )
    assert included.frame_count == 4
    assert included.endpoint_appended
    assert included.time_seconds_text(3) == "1.1"
    assert omitted.frame_count == 3
    assert not omitted.endpoint_included
    assert omitted.time_seconds_text(2) == "1.0"


def test_plan_rejects_excessive_work_without_materializing_frame_times():
    with pytest.raises(ValueError, match="exceeding max_frames"):
        build_offline_frame_plan(
            _path("1000000000"),
            OfflineRenderSettings(fps_numerator=60, max_frames=1000),
        )


def test_jobs_preserve_exact_deep_camera_text_and_requested_dimensions():
    path = CameraPath(
        (
            FlightKeyframe(
                "0",
                CameraState("1e-400", "-2e-400", "1e-420"),
                Easing.LINEAR,
            ),
            FlightKeyframe("1", CameraState("2e-400", "-1e-400", "1e-500")),
        ),
        digits=180,
    )
    plan = build_offline_frame_plan(
        path,
        OfflineRenderSettings(width=64, height=40, fps_numerator=2),
    )
    middle = tuple(
        iter_offline_frame_jobs(
            path,
            RenderRequest(),
            plan,
            start_index=1,
            stop_index=2,
        )
    )[0]
    assert middle.index == 1
    assert middle.request.width == 64 and middle.request.height == 40
    assert middle.request.center_x_text == middle.camera.center_x_text
    assert middle.request.view_width_text == middle.camera.view_width_text
    assert middle.request.viewport.width > 0.0


def test_job_ranges_are_deterministic_and_random_accessible():
    path = _path("2")
    plan = build_offline_frame_plan(path, OfflineRenderSettings(fps_numerator=2))
    first = tuple(
        iter_offline_frame_jobs(
            path,
            RenderRequest(),
            plan,
            start_index=2,
            stop_index=3,
        )
    )[0]
    second = tuple(
        iter_offline_frame_jobs(
            path,
            RenderRequest(),
            plan,
            start_index=2,
            stop_index=3,
        )
    )[0]
    assert first == second
    assert first.time_seconds_text == "1.0"
    with pytest.raises(ValueError, match="outside the plan"):
        tuple(
            iter_offline_frame_jobs(
                path,
                RenderRequest(),
                plan,
                start_index=4,
                stop_index=3,
            )
        )


def test_rendered_frames_are_stateless_and_own_their_rgb_storage():
    path = _path("0.5")
    plan = build_offline_frame_plan(
        path, OfflineRenderSettings(width=8, height=6, fps_numerator=2)
    )
    renderer = _Renderer()
    frames = list(render_offline_frames(path, RenderRequest(), renderer, plan))
    assert len(frames) == 2
    assert all(call[2]["tone_state"] is None for call in renderer.calls)
    assert all(call[2]["tone_smoothing"] == 1.0 for call in renderer.calls)
    renderer.buffer.fill(99)
    assert np.all(frames[0].rgb == 17)
    assert frames[0].details == (("frame", 1),)


def test_single_frame_render_is_reproducible_by_job_index():
    path = _path("1")
    plan = build_offline_frame_plan(
        path,
        OfflineRenderSettings(width=5, height=4, fps_numerator=1),
    )
    job = tuple(
        iter_offline_frame_jobs(
            path,
            RenderRequest(),
            plan,
            start_index=1,
            stop_index=2,
        )
    )[0]
    first = render_offline_frame(job, _Renderer())
    second = render_offline_frame(job, _Renderer())
    assert first.time_seconds_text == second.time_seconds_text
    assert first.camera == second.camera
    assert np.array_equal(first.rgb, second.rgb)


def test_invalid_rgb_shape_is_reported_with_frame_context():
    class BadRenderer:
        name = "bad"

        def render_frame(self, request, *args, **kwargs):
            return _Frame(np.zeros((2, 2), dtype=np.uint8))

    path = _path()
    plan = build_offline_frame_plan(
        path,
        OfflineRenderSettings(width=5, height=4, fps_numerator=1),
    )
    job = next(iter_offline_frame_jobs(path, RenderRequest(), plan))
    with pytest.raises(OfflineFrameRenderError, match="offline frame 0") as error:
        render_offline_frame(job, BadRenderer())
    assert isinstance(error.value.cause, ValueError)


def test_renderer_failures_are_wrapped_with_exact_time():
    class FailingRenderer:
        name = "fail"

        def render_frame(self, request, *args, **kwargs):
            raise RuntimeError("device lost")

    path = _path()
    plan = build_offline_frame_plan(path, OfflineRenderSettings(fps_numerator=1))
    job = tuple(
        iter_offline_frame_jobs(
            path,
            RenderRequest(),
            plan,
            start_index=1,
        )
    )[0]
    with pytest.raises(OfflineFrameRenderError, match="at 1s failed") as error:
        render_offline_frame(job, FailingRenderer())
    assert error.value.frame_index == 1
    assert error.value.time_seconds_text == "1"


def test_invalid_settings_and_plan_path_mismatch_are_rejected():
    with pytest.raises(ValueError, match="frame rate"):
        OfflineRenderSettings(fps_numerator=0)
    plan = build_offline_frame_plan(
        _path("1"), OfflineRenderSettings(fps_numerator=1)
    )
    with pytest.raises(ValueError, match="does not match"):
        tuple(iter_offline_frame_jobs(_path("2"), RenderRequest(), plan))
