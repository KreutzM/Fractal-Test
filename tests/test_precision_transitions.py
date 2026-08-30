from __future__ import annotations

from types import SimpleNamespace

import mpmath as mp
import numpy as np

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_app import FractalStudioApp
from fractal_flight_studio.flight_controller import FlightController
from fractal_flight_studio.deep_zoom import (
    PixelGridExhaustedError,
    digits_for_bits,
    effective_direct_precision,
    minimum_safe_view_width,
    pixel_grid_quality,
    should_use_perturbation,
)
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.render_controller import RenderController
from fractal_flight_studio.renderers import select_renderer

DEEP_CENTER_X = "-0.743643887037158704752191506114774"
DEEP_CENTER_Y = "0.131825904205311970493132056385139"


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Root:
    def __init__(self) -> None:
        self.after_calls = []

    def after(self, *args):
        self.after_calls.append(args)


def test_auto_promotes_float32_before_coordinate_quantization():
    request = RenderRequest(
        width=800,
        height=600,
        viewport=Viewport(-0.5, 0.0, 1e-4),
        precision=Precision.FLOAT32,
        render_mode=RenderMode.AUTO,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-4",
    )
    assert effective_direct_precision(request) is Precision.FLOAT64
    assert should_use_perturbation(request) is False


def test_auto_uses_perturbation_after_float64_limit():
    request = RenderRequest(
        width=800,
        height=600,
        viewport=Viewport(-0.5, 0.0, 1e-13),
        precision=Precision.FLOAT32,
        render_mode=RenderMode.AUTO,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-13",
    )
    assert effective_direct_precision(request) is Precision.FLOAT64
    assert should_use_perturbation(request) is True


def test_direct_mode_never_promotes_or_switches_implicitly():
    request = RenderRequest(
        width=800,
        height=600,
        viewport=Viewport(-0.5, 0.0, 1e-13),
        precision=Precision.FLOAT32,
        render_mode=RenderMode.DIRECT,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-13",
    )
    assert effective_direct_precision(request) is Precision.FLOAT32
    assert should_use_perturbation(request) is False


def test_reference_bits_define_a_finite_flight_floor():
    request_256 = RenderRequest(
        width=1000,
        height=700,
        precision=Precision.FLOAT32,
        render_mode=RenderMode.AUTO,
        reference_bits=256,
        center_x_text=DEEP_CENTER_X,
        center_y_text=DEEP_CENTER_Y,
        view_width_text="1e-20",
    )
    request_512 = RenderRequest(
        width=request_256.width,
        height=request_256.height,
        precision=request_256.precision,
        render_mode=request_256.render_mode,
        reference_bits=512,
        center_x_text=request_256.center_x_text,
        center_y_text=request_256.center_y_text,
        view_width_text=request_256.view_width_text,
    )
    floor_256 = minimum_safe_view_width(request_256)
    floor_512 = minimum_safe_view_width(request_512)
    assert floor_256 > 0
    assert floor_512 > 0
    assert floor_512 < floor_256
    assert floor_256 > mp.mpf("1e-80")


def test_cpu_auto_reports_float32_to_float64_promotion():
    renderer = select_renderer("cpu")
    request = RenderRequest(
        width=40,
        height=30,
        max_iterations=40,
        viewport=Viewport(-0.5, 0.0, 1e-6),
        precision=Precision.FLOAT32,
        render_mode=RenderMode.AUTO,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-6",
    )
    result = renderer.render(request)
    assert result.details["render_mode"] == "direct"
    assert result.details["requested_precision"] == "float32"
    assert result.details["precision"] == "float64"
    assert result.details["precision_promoted"] is True
    assert result.details["pixel_grid_safe"] is True


class _UnusedExecutor:
    def submit(self, _job):
        raise AssertionError("this test must not submit work")


def _flight_adapter(
    *,
    generation: int,
    running: bool,
    pending_final: bool,
    current_camera: CameraState,
    current_rgb: np.ndarray,
    good_camera: CameraState,
    good_rgb: np.ndarray,
):
    render_controller = RenderController(executor=_UnusedExecutor())
    for _ in range(generation):
        render_controller.invalidate()
    flight_controller = FlightController()
    flight_controller.set_target(DEEP_CENTER_X, DEEP_CENTER_Y)
    flight_controller.running = running
    flight_controller.pending_final_quality_check = pending_final
    flight_controller.last_good_camera = good_camera
    flight_controller.last_good_rgb = good_rgb
    app = SimpleNamespace(
        render_controller=render_controller,
        flight_controller=flight_controller,
        camera=current_camera,
        last_rgb=current_rgb,
        stop_message=None,
        request_count=0,
        status_var=SimpleNamespace(set=lambda _value: None),
    )

    def stop_flight(message=None):
        flight_controller.stop()
        app.stop_message = message

    app._stop_flight = stop_flight
    app._restore_last_good_flight_frame = (
        lambda error=None, visual=None: FractalStudioApp._restore_last_good_flight_frame(
            app, error, visual
        )
    )
    app._should_check_flight_result = (
        lambda value: FractalStudioApp._should_check_flight_result(app, value)
    )
    app.request_render = lambda: setattr(app, "request_count", app.request_count + 1)
    return app


def test_flight_rejects_bad_grid_and_restores_last_good_frame():
    request = RenderRequest(
        width=64,
        height=48,
        center_x_text=DEEP_CENTER_X,
        center_y_text=DEEP_CENTER_Y,
        view_width_text="1e-20",
    )
    good_rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    quality = pixel_grid_quality(1.0, 0.0, 1e-18, 1e-18, 64, 48)
    future = SimpleNamespace(
        result=lambda: (_ for _ in ()).throw(PixelGridExhaustedError(quality))
    )
    good_camera = CameraState("-0.75", "0.1", "1e-15")
    app = _flight_adapter(
        generation=3,
        running=True,
        pending_final=False,
        current_camera=CameraState(DEEP_CENTER_X, DEEP_CENTER_Y, "1e-20"),
        current_rgb=np.full((48, 64, 3), 255, dtype=np.uint8),
        good_camera=good_camera,
        good_rgb=good_rgb,
    )

    FractalStudioApp._finish_render(app, future, 3, request)

    assert app.flight_controller.running is False
    assert app.camera == good_camera
    assert app.last_rgb is good_rgb
    assert "nicht mehr sinnvoll aufgelöst" in app.stop_message
    assert "größter Block" in app.stop_message
    assert app.request_count == 1


def test_flight_controller_clamps_at_numerical_precision_floor():
    bits = 64
    base_request = RenderRequest(
        width=100,
        height=80,
        viewport=Viewport(-0.5, 0.0, 1e-10),
        precision=Precision.FLOAT32,
        render_mode=RenderMode.AUTO,
        reference_bits=bits,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="1e-10",
    )
    floor = minimum_safe_view_width(base_request)
    current_width = floor * mp.mpf("1.01")
    camera = CameraState("-0.5", "0.0", mp.nstr(current_width, 40))
    controller = FlightController()
    controller.set_target("-0.5", "0.0")
    assert controller.start(camera, None)

    step = controller.step(
        camera,
        zoom_rate=1.035,
        minimum_width=floor,
        digits=digits_for_bits(bits),
    )

    assert step.reached_limit is True
    assert mp.mpf(step.camera.view_width_text) == floor
    controller.stop(numerical_limit=True)
    assert controller.running is False
    assert controller.pending_final_quality_check is True


def test_flight_rejects_blocky_rgb_candidate_before_display():
    request = RenderRequest(
        width=64,
        height=48,
        center_x_text=DEEP_CENTER_X,
        center_y_text=DEEP_CENTER_Y,
        view_width_text="1e-20",
    )
    source = np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3)
    blocky_rgb = np.repeat(np.repeat(source, 4, axis=0), 4, axis=1)
    good_rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    future = SimpleNamespace(
        result=lambda: SimpleNamespace(
            rgb=blocky_rgb,
            details={},
            backend="cpu-numba",
            elapsed_seconds=0.01,
        )
    )
    app = _flight_adapter(
        generation=3,
        running=True,
        pending_final=False,
        current_camera=CameraState(DEEP_CENTER_X, DEEP_CENTER_Y, "1e-20"),
        current_rgb=blocky_rgb,
        good_camera=CameraState("-0.75", "0.1", "1e-15"),
        good_rgb=good_rgb,
    )

    FractalStudioApp._finish_render(app, future, 3, request)

    assert app.flight_controller.running is False
    assert app.last_rgb is good_rgb
    assert "wiederholtes zweidimensionales Pixelraster" in app.stop_message
    assert app.request_count == 1


def test_numerically_clamped_final_frame_still_passes_visual_gate():
    request = RenderRequest(width=32, height=24, view_width_text="1e-70")
    block = np.zeros((6, 8, 3), dtype=np.uint8)
    block[..., 0] = np.arange(8, dtype=np.uint8)
    blocky_rgb = np.repeat(np.repeat(block, 4, axis=0), 4, axis=1)
    good_rgb = np.full((24, 32, 3), 17, dtype=np.uint8)
    future = SimpleNamespace(
        result=lambda: SimpleNamespace(
            rgb=blocky_rgb,
            details={},
            backend="cpu-numba",
            elapsed_seconds=0.01,
        )
    )
    app = _flight_adapter(
        generation=9,
        running=False,
        pending_final=True,
        current_camera=CameraState("-0.75", "0.1", "1e-70"),
        current_rgb=blocky_rgb,
        good_camera=CameraState("-0.75", "0.1", "1e-15"),
        good_rgb=good_rgb,
    )

    FractalStudioApp._finish_render(app, future, 9, request)

    assert app.flight_controller.pending_final_quality_check is False
    assert app.last_rgb is good_rgb
    assert "nicht mehr sinnvoll aufgelöst" in app.stop_message
    assert app.request_count == 1
