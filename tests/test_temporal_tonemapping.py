from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fractal_flight_studio.mp4_export as mp4_export
from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.ffmpeg_mp4 import (
    FFmpegInfo,
    Mp4ExportCancelled,
    Mp4ExportResult,
)
from fractal_flight_studio.flight_path import CameraPath, FlightKeyframe
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.offline_render import OfflineRenderSettings, build_offline_frame_plan
from fractal_flight_studio.renderers import FrameResult
from fractal_flight_studio.temporal_tonemapping import (
    TemporalToneSettings,
    ToneStability,
    analyze_offline_tone_states,
    stabilize_tone_states,
)
from fractal_flight_studio.tonemapping import ToneMapState, apply_tone_mapping


def _state(value: float, *, scene=("scene",)) -> ToneMapState:
    return ToneMapState("auto", scene, value, value + 2.0, 3.0 + value, 0.8 + value * 0.01)


def _path() -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "3")),
            FlightKeyframe("1", CameraState("-0.75", "0.1", "0.1")),
        )
    )


def test_locked_tone_state_is_applied_without_reanalysis():
    values = np.linspace(0.0, 10.0, 16, dtype=np.float32).reshape(4, 4)
    inside = np.zeros_like(values, dtype=bool)
    state = ToneMapState("auto", ("locked",), 2.0, 8.0, 4.0, 0.9)

    mapped, resolved, details = apply_tone_mapping(
        values,
        inside,
        mode="auto",
        state=state,
        scene_key=("locked",),
        locked=True,
    )

    assert resolved is state
    assert details["tone_state_locked"] is True
    assert details["tone_sample_count"] == 0
    assert mapped[0, 0] == 0.0
    assert mapped[-1, -1] == 1.0

    with pytest.raises(ValueError, match="scene"):
        apply_tone_mapping(
            values,
            inside,
            mode="auto",
            state=state,
            scene_key=("different",),
            locked=True,
        )


def test_zero_phase_smoothing_is_deterministic_and_reduces_parameter_jumps():
    raw = (_state(0.0), _state(0.0), _state(10.0), _state(10.0))

    first = stabilize_tone_states(raw, smoothing=0.25)
    second = stabilize_tone_states(raw, smoothing=0.25)

    assert first == second
    assert all(state is not None for state in first)
    raw_jump = max(abs(right.low - left.low) for left, right in zip(raw, raw[1:]))
    stable_jump = max(
        abs(right.low - left.low)
        for left, right in zip(first, first[1:])
        if left is not None and right is not None
    )
    assert stable_jump < raw_jump
    assert 0.0 < first[0].low < first[-1].low < 10.0


def test_smoothing_fills_frames_without_outside_samples_from_neighbors():
    states = (None, _state(2.0), None, _state(4.0), None)

    stable = stabilize_tone_states(states, smoothing=0.5)

    assert len(stable) == len(states)
    assert all(state is not None for state in stable)
    assert stable[0].scene_key == ("scene",)
    assert stable[-1].scene_key == ("scene",)
    assert stabilize_tone_states((None, None)) == (None, None)


def test_temporal_mp4_export_runs_low_resolution_prepass_and_locks_final_states(
    monkeypatch, tmp_path
):
    path = _path()
    plan = build_offline_frame_plan(
        path,
        OfflineRenderSettings(
            width=8,
            height=6,
            fps_numerator=2,
            append_endpoint=False,
        ),
    )

    class Renderer:
        name = "fake"

        def __init__(self):
            self.analysis_calls = []
            self.final_calls = []

        def render_frame(self, request, *args, **kwargs):
            if kwargs.get("tone_state_locked"):
                self.final_calls.append((request, kwargs["tone_state"]))
                state = kwargs["tone_state"]
            else:
                index = len(self.analysis_calls)
                state = ToneMapState(
                    "auto",
                    kwargs["tone_scene_key"],
                    float(index * 10),
                    float(index * 10 + 2),
                    3.0,
                    1.0,
                )
                self.analysis_calls.append((request, state))
            rgb = np.zeros((request.height, request.width, 3), dtype=np.uint8)
            return FrameResult(rgb, self.name, 0.0, {"tone_state": state})

    captured = {}

    def fake_encode(frames, export_plan, output_path, settings, **kwargs):
        captured["frames"] = list(frames)
        return Mp4ExportResult(
            Path(output_path),
            export_plan.frame_count,
            0,
            0.0,
            FFmpegInfo("ffmpeg", "ffmpeg version fake"),
            ("ffmpeg",),
        )

    monkeypatch.setattr(mp4_export, "encode_mp4_frames", fake_encode)
    renderer = Renderer()
    analysis_progress = []

    result = mp4_export.export_path_to_mp4(
        path,
        RenderRequest(),
        renderer,
        plan,
        tmp_path / "flight.mp4",
        temporal_tone=TemporalToneSettings(
            mode=ToneStability.TEMPORAL,
            analysis_width=4,
            analysis_height=2,
            smoothing=0.5,
        ),
        tone_analysis_progress=analysis_progress.append,
    )

    assert result.frame_count == 2
    assert len(renderer.analysis_calls) == 2
    assert all(call[0].width == 4 and call[0].height == 2 for call in renderer.analysis_calls)
    assert len(renderer.final_calls) == 2
    assert all(call[0].width == 8 and call[0].height == 6 for call in renderer.final_calls)
    assert [progress.frames_analyzed for progress in analysis_progress] == [1, 2]
    final_lows = [state.low for _, state in renderer.final_calls]
    assert final_lows == pytest.approx([2.5, 7.5])
    assert len(captured["frames"]) == 2


def test_tone_analysis_cancellation_stops_before_the_next_frame():
    path = _path()
    plan = build_offline_frame_plan(
        path,
        OfflineRenderSettings(
            width=8,
            height=6,
            fps_numerator=4,
            append_endpoint=False,
        ),
    )

    class Renderer:
        name = "fake"

        def __init__(self):
            self.calls = 0

        def render_frame(self, request, *args, **kwargs):
            self.calls += 1
            state = ToneMapState(
                "auto",
                kwargs["tone_scene_key"],
                0.0,
                2.0,
                3.0,
                1.0,
            )
            rgb = np.zeros((request.height, request.width, 3), dtype=np.uint8)
            return FrameResult(rgb, self.name, 0.0, {"tone_state": state})

    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 2

    renderer = Renderer()
    with pytest.raises(Mp4ExportCancelled, match="tone analysis"):
        analyze_offline_tone_states(
            path,
            RenderRequest(),
            renderer,
            plan,
            stop_index=4,
            settings=TemporalToneSettings(analysis_width=4, analysis_height=2),
            cancellation_requested=cancelled,
        )

    assert renderer.calls == 1
