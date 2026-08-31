from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import CameraPath, Easing, FlightKeyframe
from fractal_flight_studio.frame_sequence_export import (
    FrameSequenceError,
    FrameSequencePlan,
    FrameSequenceSettings,
    _publish_frame,
    export_frame_sequence,
    parse_frame_range,
)
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.offline_render import (
    OfflineFrameRenderError,
    OfflineRenderSettings,
    build_offline_frame_plan,
    render_offline_frames,
)
from fractal_flight_studio.renderers import FrameResult
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.temporal_tonemapping import (
    TemporalToneSettings,
    ToneStability,
)
from fractal_flight_studio.tonemapping import ToneMapState


def _path(duration: str = "1") -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "4"), Easing.LINEAR),
            FlightKeyframe(duration, CameraState("-0.75", "0.1", "4e-40")),
        ),
        digits=100,
    )


def _request(width: int = 6, height: int = 4) -> RenderRequest:
    return RenderRequest(width=width, height=height, max_iterations=32)


def _plan(duration: str = "1", fps_numerator: int = 2, **kwargs):
    settings = OfflineRenderSettings(
        width=6, height=4, fps_numerator=fps_numerator, **kwargs
    )
    return build_offline_frame_plan(_path(duration), settings)


class _FakeRenderer:
    name = "fake"

    def __init__(self, pattern="gradient"):
        self.calls: list[RenderRequest] = []
        self.raw_calls: list[tuple] = []
        self.pattern = pattern

    def render_frame(self, request, *args, **kwargs):
        self.calls.append(request)
        self.raw_calls.append((request, args, kwargs))
        if self.pattern == "raise":
            raise RuntimeError("renderer failure")
        y, x = np.indices((request.height, request.width))
        rgb = np.stack(
            (
                (x * 17 + y * 3) % 256,
                (x * 5 + y * 29) % 256,
                (x * 11 + y * 7) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        if self.pattern == "bad-shape":
            rgb = rgb[:, :, :2]
        details: dict = {"tone_state": None}
        if self.pattern == "tone-state":
            index = len(self.calls)
            details = {
                "tone_state": ToneMapState(
                    mode="asinh",
                    scene_key=None,
                    low=1.0 + index,
                    high=100.0 + index,
                    strength=1.0,
                    gamma=0.75,
                )
            }
        return FrameResult(rgb, self.name, 0.001, details)


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        assert image.mode == "RGB"
        return np.asarray(image)


def test_settings_validate_format_and_pattern():
    settings = FrameSequenceSettings()
    assert settings.frame_name(7) == "frame_00007.png"
    custom = FrameSequenceSettings(filename_pattern="f{index:03d}.png")
    assert custom.frame_name(3) == "f003.png"
    with pytest.raises(ValueError, match="image format"):
        FrameSequenceSettings(image_format="tiff")
    with pytest.raises(ValueError, match="index"):
        FrameSequenceSettings(filename_pattern="frame_05d.png")
    with pytest.raises(ValueError, match="index"):
        FrameSequenceSettings(filename_pattern="frame_{other}.png")
    with pytest.raises(ValueError, match="separator"):
        FrameSequenceSettings(filename_pattern="../frame_{index}.png")
    with pytest.raises(ValueError, match="extension"):
        FrameSequenceSettings(filename_pattern="frame_{index}.jpg")


def test_parse_frame_range_bounds_and_defaults():
    plan = _plan()
    assert plan.frame_count == 3
    assert parse_frame_range(plan, 0, None) == (0, 3)
    assert parse_frame_range(plan, 1, 2) == (1, 2)
    with pytest.raises(ValueError, match="outside"):
        parse_frame_range(plan, -1, 2)
    with pytest.raises(ValueError, match="outside"):
        parse_frame_range(plan, 0, 4)
    with pytest.raises(ValueError, match="outside"):
        parse_frame_range(plan, 2, 1)
    with pytest.raises(ValueError, match="integer"):
        parse_frame_range(plan, True, None)


def test_export_writes_deterministic_numbered_frames(tmp_path: Path):
    renderer = _FakeRenderer()
    result = export_frame_sequence(
        _path(), _request(), renderer, _plan(), tmp_path / "frames"
    )
    assert not result.cancelled
    assert result.rendered_indices == (0, 1, 2)
    assert result.skipped_indices == ()
    names = [p.name for p in result.frame_paths]
    assert names == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    assert all(p.parent == (tmp_path / "frames").expanduser() for p in result.frame_paths)
    assert len(renderer.calls) == 3
    # re-exporting the same plan produces byte-identical files
    before = [p.read_bytes() for p in result.frame_paths]
    again = export_frame_sequence(
        _path(), _request(), _FakeRenderer(), _plan(), tmp_path / "frames"
    )
    assert again.rendered_indices == ()
    assert again.skipped_indices == (0, 1, 2)
    assert [p.read_bytes() for p in again.frame_paths] == before


def test_export_validates_output_rgb(tmp_path: Path):
    export_frame_sequence(
        _path(), _request(), _FakeRenderer(), _plan(), tmp_path / "good"
    )
    for index in range(3):
        array = _read(tmp_path / "good" / f"frame_{index:05d}.png")
        assert array.shape == (4, 6, 3)
        assert array.dtype == np.uint8


def test_range_selection_renders_only_the_requested_slice(tmp_path: Path):
    renderer = _FakeRenderer()
    result = export_frame_sequence(
        _path(),
        _request(),
        renderer,
        _plan(),
        tmp_path,
        start_index=1,
        stop_index=2,
    )
    assert result.start_index == 1
    assert result.stop_index == 2
    assert result.rendered_indices == (1,)
    assert [p.name for p in result.frame_paths] == ["frame_00001.png"]
    assert list((tmp_path).glob("*.png")) == [tmp_path / "frame_00001.png"]


def test_resume_skips_completed_frames(tmp_path: Path):
    plan = _plan()
    first = export_frame_sequence(
        _path(), _request(), _FakeRenderer(), plan, tmp_path, stop_index=2
    )
    assert first.rendered_indices == (0, 1)
    assert first.next_start_index() is None  # the [0, 2) range is complete
    second = export_frame_sequence(
        _path(), _request(), _FakeRenderer(), plan, tmp_path
    )
    assert second.skipped_indices == (0, 1)
    assert second.rendered_indices == (2,)
    assert second.next_start_index() is None
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "frame_00000.png",
        "frame_00001.png",
        "frame_00002.png",
    ]


def test_resume_rejects_corrupt_existing_frame(tmp_path: Path):
    (tmp_path / "frame_00001.png").write_bytes(b"not a png")
    with pytest.raises(FrameSequenceError, match="frame_00001.png"):
        export_frame_sequence(_path(), _request(), _FakeRenderer(), _plan(), tmp_path)
    # overwrite explicitly replaces the corrupt frame instead
    result = export_frame_sequence(
        _path(),
        _request(),
        _FakeRenderer(),
        _plan(),
        tmp_path,
        FrameSequenceSettings(overwrite=True),
    )
    assert result.rendered_indices == (0, 1, 2)
    assert _read(tmp_path / "frame_00001.png").shape == (4, 6, 3)


def test_resume_rejects_wrong_sized_existing_frame(tmp_path: Path):
    Image.new("RGB", (9, 9), (1, 2, 3)).save(tmp_path / "frame_00002.png")
    with pytest.raises(FrameSequenceError, match="expected RGB"):
        export_frame_sequence(
            _path(),
            _request(),
            _FakeRenderer(),
            _plan(fps_numerator=2),
            tmp_path,
            start_index=2,
        )


def test_invalid_frames_never_publish_and_clean_temporaries(tmp_path: Path):
    # offline_render rejects renderer shape/type violations before publication
    with pytest.raises(OfflineFrameRenderError):
        export_frame_sequence(
            _path(),
            _request(),
            _FakeRenderer("bad-shape"),
            _plan(),
            tmp_path,
            start_index=0,
            stop_index=1,
        )
    assert list(tmp_path.iterdir()) == []
    # the publication layer itself validates defensively
    plan = FrameSequencePlan(_plan(), tmp_path)
    with pytest.raises(FrameSequenceError, match="RGB shape"):
        _publish_frame(np.zeros((4, 6, 2), dtype=np.uint8), plan.frame_path(0), plan)
    with pytest.raises(FrameSequenceError, match="dtype"):
        _publish_frame(
            np.zeros((4, 6, 3), dtype=np.float32), plan.frame_path(0), plan
        )
    with pytest.raises(FrameSequenceError, match="RGB shape"):
        _publish_frame(
            np.full((8, 6, 3), 7, dtype=np.uint8), plan.frame_path(0), plan
        )
    assert list(tmp_path.iterdir()) == []
    # a successful publish leaves exactly one file, no temporary residue
    _publish_frame(
        np.full((4, 6, 3), 7, dtype=np.uint8), plan.frame_path(0), plan
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["frame_00000.png"]


def test_renderer_failure_leaves_no_partial_output(tmp_path: Path):
    with pytest.raises(Exception):
        export_frame_sequence(
            _path(),
            _request(),
            _FakeRenderer("raise"),
            _plan(),
            tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_cancellation_stops_between_frames_and_reports_resume_index(tmp_path: Path):
    seen = 0

    def cancel_now() -> bool:
        return seen >= 2

    progress_seen = []

    def progress(event) -> None:
        nonlocal seen
        seen += 1
        progress_seen.append((event.action, event.frame_index, event.completed_frames))

    result = export_frame_sequence(
        _path(),
        _request(),
        _FakeRenderer(),
        _plan(),
        tmp_path,
        progress=progress,
        cancellation_requested=cancel_now,
    )
    assert result.cancelled
    assert result.rendered_indices == (0, 1)
    assert result.next_start_index() == 2
    assert progress_seen == [("rendered", 0, 1), ("rendered", 1, 2)]
    # no stray temporary files remain
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "frame_00000.png",
        "frame_00001.png",
    ]


def test_rational_cadence_times_are_exact(tmp_path: Path):
    plan = _plan(duration="1", fps_numerator=30000, fps_denominator=1001)
    renderer = _FakeRenderer()
    result = export_frame_sequence(_path(), _request(), renderer, plan, tmp_path)
    assert result.rendered_indices == tuple(range(plan.frame_count))
    assert plan.fps_numerator == 30000 and plan.fps_denominator == 1001
    expected_times = [plan.time_seconds_text(i) for i in range(plan.frame_count)]
    assert expected_times[1] == "0.03336666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666667"
    # appended endpoint frame exists at the exact duration text
    assert plan.endpoint_appended
    assert expected_times[-1] == plan.duration_text


def test_temporal_tone_states_are_planned_and_locked(tmp_path: Path):
    plan = _plan()
    renderer = _FakeRenderer("tone-state")
    analysis_events = []
    result = export_frame_sequence(
        _path(),
        _request(),
        renderer,
        plan,
        tmp_path,
        temporal_tone=TemporalToneSettings(
            mode=ToneStability.TEMPORAL, analysis_width=8, analysis_height=6
        ),
        tone_analysis_progress=lambda progress: analysis_events.append(progress),
    )
    assert result.rendered_indices == (0, 1, 2)
    # analysis covered every planned frame before the full-resolution pass
    assert [e.frames_analyzed for e in analysis_events] == [1, 2, 3]
    # 3 low-res analysis calls + 3 full-res publication calls
    assert len(renderer.calls) == 6
    assert [c.width for c in renderer.calls[:3]] == [8, 8, 8]
    assert [c.width for c in renderer.calls[3:]] == [6, 6, 6]
    # the final pass receives smoothed, locked tone states (not the raw ramp)
    tone_kwargs = _tone_kwargs(renderer)
    final_states = [kwargs["tone_state"] for kwargs in tone_kwargs[3:]]
    assert all(state is not None and state.mode == "asinh" for state in final_states)
    locked = [kwargs["tone_state_locked"] for kwargs in tone_kwargs]
    assert locked[3:] == [True, True, True]
    smoothed = [state.low for state in final_states]
    assert smoothed[1] < smoothed[2]  # temporal ordering preserved
    assert smoothed != [1.0, 2.0, 3.0]  # actual smoothing, not the raw ramp


def test_temporal_tone_rejects_non_zero_start(tmp_path: Path):
    output = tmp_path / "seq"
    with pytest.raises(ValueError, match="index 0"):
        export_frame_sequence(
            _path(),
            _request(),
            _FakeRenderer("tone-state"),
            _plan(),
            output,
            start_index=1,
            temporal_tone=TemporalToneSettings(
                mode=ToneStability.TEMPORAL, analysis_width=8, analysis_height=6
            ),
        )
    # validation failures happen before the output directory is created
    assert not output.exists()


def _tone_kwargs(renderer):
    # render_frame(self, request, *args, **kwargs): offline_render passes the
    # tone arguments as keywords.
    return [kwargs for _request_, _args, kwargs in renderer.raw_calls]


def test_inputs_are_not_mutated(tmp_path: Path):
    path = _path()
    request = _request()
    plan = _plan()
    path_snapshot = copy.deepcopy(path)
    request_snapshot = copy.deepcopy(request)
    plan_snapshot = copy.deepcopy(plan)
    export_frame_sequence(path, request, _FakeRenderer(), plan, tmp_path)
    assert path == path_snapshot
    assert request == request_snapshot
    assert plan == plan_snapshot
    assert request.viewport == request_snapshot.viewport
    assert request.center_x_text == request_snapshot.center_x_text


def test_frames_match_render_offline_frames_output(tmp_path: Path):
    plan = _plan()
    renderer = _FakeRenderer()
    result = export_frame_sequence(_path(), _request(), renderer, plan, tmp_path)
    reference = _FakeRenderer()
    frames = list(
        render_offline_frames(_path(), _request(), reference, plan)
    )
    assert len(frames) == len(result.frame_paths)
    for frame, path in zip(frames, result.frame_paths):
        assert path.name == f"frame_{frame.index:05d}.png"
        assert np.array_equal(_read(path), frame.rgb)


def test_cpu_renderer_end_to_end_without_gpu(tmp_path: Path):
    from fractal_flight_studio.models import Precision

    plan = build_offline_frame_plan(
        _path(),
        OfflineRenderSettings(width=16, height=12, fps_numerator=2),
    )
    request = RenderRequest(
        width=16,
        height=12,
        max_iterations=64,
        precision=Precision.FLOAT64,
    )
    result = export_frame_sequence(
        _path(), request, CpuRenderer(), plan, tmp_path / "cpu"
    )
    assert result.rendered_indices == (0, 1, 2)
    arrays = [_read(tmp_path / "cpu" / f"frame_{i:05d}.png") for i in range(3)]
    assert all(a.shape == (12, 16, 3) and a.dtype == np.uint8 for a in arrays)
    # deep-zoom frames are not all-identical and not all-black
    assert not np.array_equal(arrays[0], arrays[2])
    assert arrays[0].max() > 0


def test_progress_and_result_counts(tmp_path: Path):
    events = []
    result = export_frame_sequence(
        _path(),
        _request(),
        _FakeRenderer(),
        _plan(),
        tmp_path,
        stop_index=2,
        progress=lambda event: events.append(event),
    )
    assert [e.total_frames for e in events] == [2, 2]
    assert [e.completed_frames for e in events] == [1, 2]
    assert result.completed_indices == (0, 1)
    assert result.total_frames == 2
    assert result.elapsed_seconds >= 0.0
