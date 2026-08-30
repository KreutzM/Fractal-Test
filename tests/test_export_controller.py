from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import fractal_flight_studio.export_controller as workflow
from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.ffmpeg_mp4 import FFmpegInfo, Mp4ExportResult
from fractal_flight_studio.flight_path import (
    CameraPath,
    CenterInterpolation,
    FlightKeyframe,
)
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.preflight import PreflightCancelled, PreflightSettings
from fractal_flight_studio.renderers import FrameResult


def _path() -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "3")),
            FlightKeyframe("1", CameraState("-0.75", "0.1", "0.1")),
        )
    )


class _Renderer:
    name = "fake"

    def render_frame(self, request, *args, **kwargs):
        y, x = np.indices((request.height, request.width))
        rgb = np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8)
        return FrameResult(rgb, self.name, 0.01, {"pixel_grid_safe": True})


def test_parse_frame_rate_accepts_exact_decimal_and_fraction():
    assert workflow.parse_frame_rate("29.97") == (2997, 100)
    assert workflow.parse_frame_rate("30000/1001") == (30000, 1001)

    with pytest.raises(ValueError, match="positive"):
        workflow.parse_frame_rate("0")
    with pytest.raises(ValueError, match="decimal or fraction"):
        workflow.parse_frame_rate("abc")


def test_configuration_builds_cadence_only_video_plan():
    config = workflow.FlightExportConfiguration(
        width=640,
        height=360,
        frame_rate_text="30",
    )

    offline = config.build_offline_plan(_path())

    assert offline.endpoint_appended is False
    assert config.export_frame_count(_path()) == 30


def test_configuration_rejects_odd_yuv420_dimensions():
    with pytest.raises(ValueError, match="even video dimensions"):
        workflow.FlightExportConfiguration(width=641, height=360)


def test_fingerprint_changes_with_path_or_render_configuration():
    path = _path()
    request = RenderRequest(max_iterations=400)
    config = workflow.FlightExportConfiguration(width=640, height=360)
    first = workflow.flight_export_fingerprint(
        path, request, "fake", "inferno", 1.0, config
    )
    changed_request = workflow.flight_export_fingerprint(
        path, RenderRequest(max_iterations=800), "fake", "inferno", 1.0, config
    )
    changed_config = workflow.flight_export_fingerprint(
        path,
        request,
        "fake",
        "inferno",
        1.0,
        workflow.FlightExportConfiguration(width=1280, height=720),
    )
    changed_center_mode = workflow.flight_export_fingerprint(
        CameraPath(
            (
                FlightKeyframe(
                    "0",
                    CameraState("-0.5", "0", "3"),
                    center_interpolation=CenterInterpolation.FOCUS,
                ),
                FlightKeyframe("1", CameraState("-0.75", "0.1", "0.1")),
            )
        ),
        request,
        "fake",
        "inferno",
        1.0,
        config,
    )

    assert first != changed_request
    assert first != changed_config
    assert first != changed_center_mode


def test_controller_runs_preflight_and_publishes_progress():
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller = workflow.FlightExportController(executor)
        future = controller.start_preflight(
            _path(),
            RenderRequest(),
            _Renderer(),
            PreflightSettings(width=32, height=24, sample_interval_seconds_text="0.5"),
            palette="inferno",
            cycles=1.0,
        )
        report = future.result(timeout=5)
        progress = controller.progress
        controller.complete(future)

    assert report.safe
    assert progress is not None
    assert progress.kind is workflow.FlightExportJobKind.PREFLIGHT
    assert progress.completed == progress.total == 3
    assert not controller.busy


def test_controller_probe_and_mp4_jobs_use_worker_callbacks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        workflow,
        "probe_ffmpeg",
        lambda executable: FFmpegInfo(executable, "ffmpeg version fake"),
    )
    output = tmp_path / "flight.mp4"

    def fake_export(
        path,
        request,
        renderer,
        offline_plan,
        output_path,
        settings,
        **kwargs,
    ):
        total = workflow.build_mp4_export_plan(offline_plan).frame_count
        for index in range(total):
            kwargs["progress"](
                type(
                    "Progress",
                    (),
                    {"frames_written": index + 1, "total_frames": total},
                )()
            )
        Path(output_path).write_bytes(b"mp4")
        return Mp4ExportResult(
            Path(output_path),
            total,
            total * offline_plan.width * offline_plan.height * 3,
            0.1,
            FFmpegInfo("ffmpeg", "ffmpeg version fake"),
            ("ffmpeg",),
        )

    monkeypatch.setattr(workflow, "export_path_to_mp4", fake_export)

    with ThreadPoolExecutor(max_workers=1) as executor:
        controller = workflow.FlightExportController(executor)
        probe = controller.start_probe("ffmpeg-test")
        assert probe.result(timeout=5).executable == "ffmpeg-test"
        controller.complete(probe)

        config = workflow.FlightExportConfiguration(
            width=64,
            height=36,
            frame_rate_text="2",
        )
        export = controller.start_mp4(
            _path(),
            RenderRequest(),
            _Renderer(),
            config.build_offline_plan(_path()),
            output,
            config.mp4_settings(),
            palette="inferno",
            cycles=1.0,
        )
        result = export.result(timeout=5)
        progress = controller.progress
        controller.complete(export)

    assert result.output_path == output
    assert progress is not None
    assert progress.kind is workflow.FlightExportJobKind.MP4
    assert progress.completed == progress.total == 2


def test_controller_cancel_stops_preflight_between_samples():
    from threading import Event

    started = Event()
    release = Event()

    class SlowRenderer(_Renderer):
        def render_frame(self, request, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().render_frame(request, *args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        controller = workflow.FlightExportController(executor)
        future = controller.start_preflight(
            _path(),
            RenderRequest(),
            SlowRenderer(),
            PreflightSettings(width=32, height=24, sample_interval_seconds_text="0.25"),
            palette="inferno",
            cycles=1.0,
        )
        assert started.wait(timeout=5)
        controller.cancel()
        release.set()
        with pytest.raises(PreflightCancelled):
            future.result(timeout=5)
        progress = controller.progress
        controller.complete(future)

    assert progress is not None
    assert progress.message == "Abbruch angefordert …"


def test_temporal_export_progress_covers_analysis_and_encoding(monkeypatch, tmp_path):
    output = tmp_path / "temporal.mp4"

    def fake_export(path, request, renderer, offline_plan, output_path, settings, **kwargs):
        total = workflow.build_mp4_export_plan(offline_plan).frame_count
        for index in range(total):
            kwargs["tone_analysis_progress"](
                type(
                    "AnalysisProgress",
                    (),
                    {
                        "frames_analyzed": index + 1,
                        "total_frames": total,
                        "time_seconds_text": str(index),
                    },
                )()
            )
        for index in range(total):
            kwargs["progress"](
                type(
                    "Progress",
                    (),
                    {"frames_written": index + 1, "total_frames": total},
                )()
            )
        return Mp4ExportResult(
            Path(output_path),
            total,
            0,
            0.0,
            FFmpegInfo("ffmpeg", "ffmpeg version fake"),
            ("ffmpeg",),
        )

    monkeypatch.setattr(workflow, "export_path_to_mp4", fake_export)
    config = workflow.FlightExportConfiguration(width=64, height=36, frame_rate_text="2")

    with ThreadPoolExecutor(max_workers=1) as executor:
        controller = workflow.FlightExportController(executor)
        future = controller.start_mp4(
            _path(),
            RenderRequest(),
            _Renderer(),
            config.build_offline_plan(_path()),
            output,
            config.mp4_settings(),
            palette="inferno",
            cycles=1.0,
            temporal_tone=config.temporal_tone_settings(),
        )
        future.result(timeout=5)
        progress = controller.progress
        controller.complete(future)

    assert progress is not None
    assert progress.completed == progress.total == 4
    assert progress.message == "Rendere und kodiere Frame 2/2"


def test_tone_stability_is_validated_and_part_of_preflight_fingerprint():
    temporal = workflow.FlightExportConfiguration(width=64, height=36)
    per_frame = workflow.FlightExportConfiguration(
        width=64,
        height=36,
        tone_stability="per_frame",
    )

    assert temporal.tone_stability.value == "temporal"
    assert temporal.preflight_fingerprint() != per_frame.preflight_fingerprint()
    with pytest.raises(ValueError, match="ToneStability"):
        workflow.FlightExportConfiguration(
            width=64,
            height=36,
            tone_stability="unknown",
        )
