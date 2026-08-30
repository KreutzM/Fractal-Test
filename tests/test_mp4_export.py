from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import fractal_flight_studio.ffmpeg_mp4 as mp4
import fractal_flight_studio.mp4_export as path_export
from fractal_flight_studio.offline_render import OfflineFrame, OfflineFramePlan


def _offline_plan(*, duration="1", fps=2, frame_count=3, included=True, appended=False):
    return OfflineFramePlan(4, 2, fps, 1, duration, 50, frame_count, included, appended)


def _frames(plan: mp4.Mp4ExportPlan):
    for index in range(plan.frame_count):
        yield OfflineFrame(
            index,
            plan.time_seconds_text(index),
            None,
            np.full((plan.height, plan.width, 3), index, dtype=np.uint8),
            "fake",
            0.0,
            (),
        )


class _Input:
    def __init__(self, owner):
        self.owner = owner
        self.closed = False

    def write(self, data):
        if self.owner.break_pipe:
            raise BrokenPipeError("closed")
        self.owner.payload.extend(data)
        return len(data)

    def close(self):
        self.closed = True


class _Process:
    def __init__(self, command, *, returncode=0, stderr=b"", break_pipe=False):
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = io.BytesIO(stderr)
        self.payload = bytearray()
        self.break_pipe = break_pipe
        self.stdin = _Input(self)
        self.terminated = False
        self.killed = False
        self._done = False

    def wait(self, timeout=None):
        self._done = True
        if self.returncode == 0 and not self.terminated and not self.killed:
            Path(self.command[-1]).write_bytes(b"fake-mp4")
        return self.returncode

    def poll(self):
        return self.returncode if self._done else None

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self._done = True

    def kill(self):
        self.killed = True
        self.returncode = -9
        self._done = True


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    info = mp4.FFmpegInfo("/fake/ffmpeg", "ffmpeg version test")
    monkeypatch.setattr(mp4, "probe_ffmpeg", lambda executable: info)
    made = []

    def install(*, returncode=0, stderr=b"", break_pipe=False):
        def factory(command, **kwargs):
            process = _Process(
                command,
                returncode=returncode,
                stderr=stderr,
                break_pipe=break_pipe,
            )
            made.append(process)
            return process

        monkeypatch.setattr(mp4.subprocess, "Popen", factory)
        return made

    return install


def test_mp4_plan_drops_endpoint_on_exact_cadence():
    plan = path_export.build_mp4_export_plan(_offline_plan())
    assert plan.frame_count == 2
    assert [plan.time_seconds_text(i) for i in range(2)] == ["0.0", "0.5"]


def test_mp4_plan_keeps_all_regular_frames_for_off_cadence_duration():
    source = _offline_plan(duration="1.25", fps=2, frame_count=3, included=False)
    plan = path_export.build_mp4_export_plan(source)
    assert plan.frame_count == 3
    assert plan.time_seconds_text(2) == "1.0"


def test_mp4_plan_rejects_appended_endpoint():
    with pytest.raises(ValueError, match="append_endpoint=False"):
        path_export.build_mp4_export_plan(_offline_plan(frame_count=4, appended=True))


def test_command_uses_raw_rgb_and_exact_fractional_rate(tmp_path):
    source = OfflineFramePlan(1920, 1080, 30000, 1001, "1", 80, 30, False, False)
    plan = path_export.build_mp4_export_plan(source)
    command = mp4.build_ffmpeg_mp4_command(
        mp4.FFmpegInfo("ffmpeg", "version"),
        plan,
        tmp_path / "out.part",
        mp4.Mp4ExportSettings(),
    )
    assert command[command.index("-f") + 1] == "rawvideo"
    assert command[command.index("-pixel_format") + 1] == "rgb24"
    assert command[command.index("-framerate") + 1] == "30000/1001"
    assert command[-3:] == ("-f", "mp4", str(tmp_path / "out.part"))


def test_yuv420p_rejects_odd_dimensions(tmp_path):
    plan = mp4.Mp4ExportPlan(3, 2, 30, 1, "1", 50, 1, 1)
    with pytest.raises(ValueError, match="even video dimensions"):
        mp4.build_ffmpeg_mp4_command(
            mp4.FFmpegInfo("ffmpeg", "version"),
            plan,
            tmp_path / "x",
            mp4.Mp4ExportSettings(),
        )


def test_encoder_streams_exact_bytes_and_atomically_publishes(tmp_path, fake_ffmpeg):
    made = fake_ffmpeg()
    plan = path_export.build_mp4_export_plan(_offline_plan())
    progress = []
    output = tmp_path / "flight.mp4"
    result = mp4.encode_mp4_frames(_frames(plan), plan, output, progress=progress.append)
    assert output.read_bytes() == b"fake-mp4"
    assert len(made[0].payload) == plan.frame_count * plan.width * plan.height * 3
    assert result.frame_count == 2
    assert progress[-1].frames_written == 2
    assert not list(tmp_path.glob("*.part"))


def test_encoder_refuses_existing_output_without_overwrite(tmp_path, fake_ffmpeg):
    fake_ffmpeg()
    output = tmp_path / "flight.mp4"
    output.write_bytes(b"original")
    plan = path_export.build_mp4_export_plan(_offline_plan())
    with pytest.raises(FileExistsError):
        mp4.encode_mp4_frames(_frames(plan), plan, output)
    assert output.read_bytes() == b"original"


def test_encoder_rejects_short_stream_and_removes_partial(tmp_path, fake_ffmpeg):
    made = fake_ffmpeg()
    plan = path_export.build_mp4_export_plan(_offline_plan())
    with pytest.raises(mp4.Mp4ExportError, match="ended after 1"):
        mp4.encode_mp4_frames(list(_frames(plan))[:1], plan, tmp_path / "flight.mp4")
    assert made[0].terminated
    assert not any(tmp_path.iterdir())


def test_encoder_reports_ffmpeg_failure_and_stderr(tmp_path, fake_ffmpeg):
    fake_ffmpeg(returncode=1, stderr=b"encoder unavailable")
    plan = path_export.build_mp4_export_plan(_offline_plan())
    with pytest.raises(mp4.Mp4ExportError, match="encoder unavailable") as error:
        mp4.encode_mp4_frames(_frames(plan), plan, tmp_path / "flight.mp4")
    assert error.value.return_code == 1
    assert not any(tmp_path.iterdir())


def test_encoder_reports_broken_pipe_with_frame_context(tmp_path, fake_ffmpeg):
    fake_ffmpeg(returncode=1, stderr=b"bad codec", break_pipe=True)
    plan = path_export.build_mp4_export_plan(_offline_plan())
    with pytest.raises(mp4.Mp4ExportError, match="frame 0"):
        mp4.encode_mp4_frames(_frames(plan), plan, tmp_path / "flight.mp4")


def test_encoder_cancellation_terminates_process_and_cleans_partial(tmp_path, fake_ffmpeg):
    made = fake_ffmpeg()
    plan = path_export.build_mp4_export_plan(_offline_plan())
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(mp4.Mp4ExportCancelled):
        mp4.encode_mp4_frames(
            _frames(plan), plan, tmp_path / "flight.mp4", cancellation_requested=cancelled
        )
    assert made[0].terminated
    assert not any(tmp_path.iterdir())


def test_encoder_validates_frame_identity_and_dtype(tmp_path, fake_ffmpeg):
    fake_ffmpeg()
    plan = path_export.build_mp4_export_plan(_offline_plan())
    bad = OfflineFrame(
        4, "0.0", None, np.zeros((2, 4, 3), dtype=np.uint8), "fake", 0, ()
    )
    with pytest.raises(mp4.Mp4ExportError, match="expected frame index 0"):
        mp4.encode_mp4_frames([bad], plan, tmp_path / "flight.mp4")


def test_resolve_ffmpeg_reports_missing_binary(monkeypatch):
    monkeypatch.setattr(mp4.shutil, "which", lambda executable: None)
    with pytest.raises(mp4.FFmpegNotFoundError, match="not found on PATH"):
        mp4.resolve_ffmpeg("definitely-missing-ffmpeg")


def test_high_level_export_drops_aligned_endpoint(monkeypatch, tmp_path):
    source = _offline_plan()
    captured = {}
    sentinel = object()

    def fake_render(*args, **kwargs):
        captured["stop_index"] = kwargs["stop_index"]
        return iter(())

    def fake_encode(frames, plan, output_path, settings, **kwargs):
        captured["plan"] = plan
        captured["output"] = output_path
        return sentinel

    monkeypatch.setattr(path_export, "render_offline_frames", fake_render)
    monkeypatch.setattr(path_export, "encode_mp4_frames", fake_encode)
    result = path_export.export_path_to_mp4(
        object(), object(), object(), source, tmp_path / "flight.mp4"
    )
    assert result is sentinel
    assert captured["stop_index"] == 2
    assert captured["plan"].frame_count == 2
