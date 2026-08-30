from __future__ import annotations

import numpy as np
import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.deep_zoom import PixelGridExhaustedError, PixelGridQuality
from fractal_flight_studio.flight_path import CameraPath, Easing, FlightKeyframe
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.preflight import (
    PreflightCancelled,
    PreflightIssueKind,
    PreflightSettings,
    build_preflight_plan,
    run_path_preflight,
)
from fractal_flight_studio.renderers import FrameResult


def _path(duration: str = "2.5") -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "4"), Easing.LINEAR),
            FlightKeyframe(duration, CameraState("-0.75", "0.125", "4e-40")),
        ),
        digits=100,
    )


def _structured_rgb(width: int, height: int) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        (
            (x * 17 + y * 3) % 256,
            (x * 5 + y * 29) % 256,
            (x * 11 + y * 7) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


class _Renderer:
    name = "fake"

    def __init__(self, *, mode: str = "safe") -> None:
        self.mode = mode
        self.requests = []
        self.tone_states = []

    def render_frame(self, request, *args, tone_state=None, **kwargs):
        self.requests.append(request)
        self.tone_states.append(tone_state)
        if self.mode == "runtime-error":
            raise RuntimeError("backend unavailable")
        if self.mode == "grid-exhausted":
            raise PixelGridExhaustedError(
                PixelGridQuality(0.0, 0.0, 0.0, 2, False)
            )
        rgb = (
            np.zeros((request.height, request.width, 3), dtype=np.uint8)
            if self.mode == "uniform"
            else _structured_rgb(request.width, request.height)
        )
        details = {
            "pixel_grid_safe": self.mode != "unsafe-grid",
            "tone_state": object(),
        }
        if self.mode == "unsafe-grid":
            details.update(
                pixel_grid_x_unique_fraction=0.8,
                pixel_grid_y_unique_fraction=1.0,
                pixel_grid_maximum_equal_run=3,
            )
        return FrameResult(rgb, self.name, 0.01, details)


def test_plan_uses_exact_interval_and_includes_final_endpoint():
    plan = build_preflight_plan(
        _path(), PreflightSettings(sample_interval_seconds_text="1", max_samples=10)
    )
    assert plan.sample_times_text == ("0.0", "1.0", "2.0", "2.5")
    assert plan.decimated is False


def test_plan_decimates_evenly_when_sample_cap_is_exceeded():
    plan = build_preflight_plan(
        _path(), PreflightSettings(sample_interval_seconds_text="0.1", max_samples=3)
    )
    assert plan.sample_times_text == ("0.0", "1.25", "2.5")
    assert plan.decimated is True


def test_plan_caps_extreme_nominal_sample_counts_without_expanding_them():
    plan = build_preflight_plan(
        _path("1e100"),
        PreflightSettings(sample_interval_seconds_text="1e-100", max_samples=3),
    )
    assert plan.sample_times_text == ("0.0", "5.0e+99", "1.0e+100")
    assert plan.decimated is True


def test_preflight_builds_low_resolution_exact_camera_requests_and_reuses_local_tone_state():
    renderer = _Renderer()
    report = run_path_preflight(
        _path(),
        RenderRequest(width=1920, height=1080),
        renderer,
        PreflightSettings(width=96, height=60, sample_interval_seconds_text="1"),
    )
    assert report.safe
    assert len(report.samples) == 4
    assert all(
        request.width == 96 and request.height == 60 for request in renderer.requests
    )
    assert renderer.requests[-1].center_x_text == "-0.75"
    assert renderer.requests[-1].view_width_text == "4e-40"
    assert renderer.requests[-1].viewport.width > 0.0
    assert renderer.tone_states[0] is None
    assert renderer.tone_states[1] is not None
    assert dict(report.samples[0].details) == {"pixel_grid_safe": True}


def test_preflight_preserves_sub_float64_camera_text():
    path = CameraPath(
        (
            FlightKeyframe(
                "0", CameraState("1e-400", "-2e-400", "1e-420"), Easing.LINEAR
            ),
            FlightKeyframe("1", CameraState("2e-400", "-3e-400", "1e-500")),
        ),
        digits=180,
    )
    renderer = _Renderer()
    run_path_preflight(
        path,
        RenderRequest(),
        renderer,
        PreflightSettings(sample_interval_seconds_text="0.5"),
    )
    assert renderer.requests[1].center_x_text != "0.0"
    assert renderer.requests[1].view_width_text != "0.0"
    assert renderer.requests[1].viewport.center_x == 0.0
    assert renderer.requests[1].viewport.width > 0.0


@pytest.mark.parametrize(
    "mode, kind",
    [
        ("uniform", PreflightIssueKind.VISUAL),
        ("unsafe-grid", PreflightIssueKind.NUMERICAL),
        ("grid-exhausted", PreflightIssueKind.NUMERICAL),
        ("runtime-error", PreflightIssueKind.RENDER_ERROR),
    ],
)
def test_preflight_reports_quality_and_render_failures(mode, kind):
    report = run_path_preflight(
        _path("1"),
        RenderRequest(),
        _Renderer(mode=mode),
        PreflightSettings(sample_interval_seconds_text="1"),
    )
    assert not report.safe
    assert report.first_issue is not None
    assert report.first_issue.kind is kind


def test_stop_on_failure_avoids_rendering_remaining_samples():
    renderer = _Renderer(mode="uniform")
    report = run_path_preflight(
        _path(),
        RenderRequest(),
        renderer,
        PreflightSettings(sample_interval_seconds_text="0.5", stop_on_failure=True),
    )
    assert report.stopped_early
    assert len(report.samples) == 1
    assert len(renderer.requests) == 1


def test_settings_reject_invalid_workloads():
    with pytest.raises(ValueError, match="dimensions"):
        PreflightSettings(width=0)
    with pytest.raises(ValueError, match="at least two"):
        PreflightSettings(max_samples=1)
    with pytest.raises(ValueError, match="finite and positive"):
        PreflightSettings(sample_interval_seconds_text="0")


def test_preflight_reports_progress_for_each_completed_sample():
    progress = []
    report = run_path_preflight(
        _path("1"),
        RenderRequest(),
        _Renderer(),
        PreflightSettings(sample_interval_seconds_text="0.5"),
        progress=lambda sample, total: progress.append(
            (sample.index, sample.time_seconds_text, total)
        ),
    )

    assert report.safe
    assert progress == [(0, "0.0", 3), (1, "0.5", 3), (2, "1.0", 3)]


def test_preflight_cancellation_stops_before_next_sample():
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 2

    renderer = _Renderer()
    with pytest.raises(PreflightCancelled, match="after 1 samples"):
        run_path_preflight(
            _path("2"),
            RenderRequest(),
            renderer,
            PreflightSettings(sample_interval_seconds_text="0.5"),
            cancellation_requested=cancelled,
        )

    assert len(renderer.requests) == 1
