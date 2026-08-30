from __future__ import annotations

from concurrent.futures import Future

import mpmath as mp
import numpy as np
import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_controller import FlightController
from fractal_flight_studio.render_controller import RenderController


class _InlineExecutor:
    def submit(self, job):
        future = Future()
        try:
            future.set_result(job())
        except Exception as exc:  # pragma: no cover - Future behavior under test
            future.set_exception(exc)
        return future


class _FailingExecutor:
    def submit(self, _job):
        raise RuntimeError("executor unavailable")


def test_camera_zoom_preserves_cursor_anchor_at_deep_precision():
    camera = CameraState(
        "-0.743643887037158704752191506114774",
        "0.131825904205311970493132056385139",
        "1e-40",
    )
    digits = 100
    before = camera.pixel_to_complex(713, 211, 1280, 720, digits=digits)

    zoomed = camera.zoom_at(713, 211, 1280, 720, 1.35, digits=digits)
    after = zoomed.pixel_to_complex(713, 211, 1280, 720, digits=digits)

    assert abs(before[0] - after[0]) < mp.mpf("1e-95")
    assert abs(before[1] - after[1]) < mp.mpf("1e-95")
    assert mp.mpf(zoomed.view_width_text) < mp.mpf(camera.view_width_text)


def test_camera_pan_and_proxy_keep_text_as_source_of_truth():
    camera = CameraState("1e-400", "-2e-400", "1e-420")
    panned = camera.pan_pixels(10, -4, 1000, digits=160)
    proxy = camera.proxy_viewport(digits=160)

    assert panned.center_x_text != camera.center_x_text
    assert panned.center_y_text != camera.center_y_text
    assert camera.center_x_text == "1e-400"
    assert proxy.width > 0.0
    assert proxy.center_x == 0.0


def test_flight_controller_owns_last_good_frame_and_final_gate():
    controller = FlightController()
    controller.set_target("-0.75", "0.1")
    camera = CameraState("-0.5", "0.0", "3.5")
    rgb = np.full((4, 5, 3), 17, dtype=np.uint8)

    assert controller.start(camera, rgb)
    rgb.fill(99)
    assert np.all(controller.last_good_rgb == 17)
    assert controller.should_check_result(2, 9)

    controller.stop(numerical_limit=True)
    assert controller.should_check_result(9, 9)
    assert not controller.should_check_result(8, 9)

    accepted = np.full((4, 5, 3), 31, dtype=np.uint8)
    controller.accept(CameraState("-0.7", "0.08", "1e-9"), accepted)
    assert controller.pending_final_quality_check is False
    assert np.all(controller.last_good_rgb == 31)


def test_render_controller_coalesces_invalidations():
    controller = RenderController(executor=_InlineExecutor())
    first_generation = controller.invalidate()
    first = controller.submit(lambda: "first")

    assert first is not None
    assert first[0] == first_generation
    assert controller.busy
    controller.invalidate()
    assert controller.submit(lambda: "ignored") is None
    assert controller.complete(first_generation) is True
    assert not controller.busy

    second = controller.submit(lambda: "second")
    assert second is not None
    assert second[1].result() == "second"
    assert controller.complete(second[0]) is False


def test_render_controller_recovers_when_executor_rejects_submission():
    controller = RenderController(executor=_FailingExecutor())
    controller.invalidate()

    with pytest.raises(RuntimeError, match="executor unavailable"):
        controller.submit(lambda: None)

    assert controller.busy is False
