from __future__ import annotations

import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import CameraPath, FlightKeyframe
from fractal_flight_studio.flight_plan import (
    FlightPlanDocument,
    PaletteTransition,
    RenderCue,
    RenderProfile,
    RenderTrack,
)
from fractal_flight_studio.flight_plan_playback import (
    FlightPlanPlaybackController,
    PlaybackState,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _document() -> FlightPlanDocument:
    path = CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "4")),
            FlightKeyframe("2", CameraState("-0.7", "0.1", "0.25")),
            FlightKeyframe("5", CameraState("-0.75", "0.12", "1e-12")),
        ),
        digits=96,
    )
    return FlightPlanDocument(
        "Playback",
        path,
        render_track=RenderTrack(
            (
                RenderCue("0", RenderProfile(400, 256, "inferno", "1")),
                RenderCue(
                    "5",
                    RenderProfile(1200, 512, "ocean", "2"),
                    PaletteTransition.BLEND,
                ),
            ),
            digits=96,
        ),
    )


def test_playback_uses_monotonic_elapsed_time_instead_of_frame_steps() -> None:
    clock = FakeClock()
    controller = FlightPlanPlaybackController(clock=clock)
    controller.load(_document())

    controller.play()
    clock.advance(2.4)
    sample = controller.sample()

    assert sample.state is PlaybackState.PLAYING
    assert sample.playhead_seconds == pytest.approx(2.4)
    assert float(sample.frame.time_seconds_text) == pytest.approx(2.4)
    assert sample.frame.render.max_iterations == 1200
    assert sample.frame.render.reference_bits == 512


def test_pause_and_resume_preserve_the_playhead_exactly() -> None:
    clock = FakeClock()
    controller = FlightPlanPlaybackController(clock=clock)
    controller.load(_document())
    controller.play()
    clock.advance(1.25)

    paused = controller.pause()
    clock.advance(10.0)
    still_paused = controller.sample()

    assert paused.playhead_seconds == pytest.approx(1.25)
    assert still_paused.playhead_seconds == pytest.approx(1.25)
    assert still_paused.state is PlaybackState.PAUSED

    controller.play()
    clock.advance(0.75)
    resumed = controller.sample()
    assert resumed.playhead_seconds == pytest.approx(2.0)


def test_seek_while_playing_reanchors_the_wall_clock() -> None:
    clock = FakeClock()
    controller = FlightPlanPlaybackController(clock=clock)
    controller.load(_document())
    controller.play()
    clock.advance(1.0)

    sought = controller.seek(3.5)
    clock.advance(0.5)
    after = controller.sample()

    assert sought.state is PlaybackState.PLAYING
    assert sought.playhead_seconds == pytest.approx(3.5)
    assert after.playhead_seconds == pytest.approx(4.0)


def test_rate_change_is_continuous_and_applies_only_to_future_time() -> None:
    clock = FakeClock()
    controller = FlightPlanPlaybackController(clock=clock)
    controller.load(_document())
    controller.play()
    clock.advance(1.0)

    changed = controller.set_rate(2.0)
    clock.advance(0.5)
    after = controller.sample()

    assert changed.playhead_seconds == pytest.approx(1.0)
    assert after.playhead_seconds == pytest.approx(2.0)
    assert after.playback_rate == 2.0


def test_natural_end_clamps_once_and_play_restarts_from_zero() -> None:
    clock = FakeClock()
    controller = FlightPlanPlaybackController(clock=clock)
    controller.load(_document(), playhead_seconds=4.5)
    controller.play()
    clock.advance(1.0)

    finished = controller.sample()

    assert finished.reached_end
    assert finished.state is PlaybackState.STOPPED
    assert finished.playhead_seconds == pytest.approx(5.0)
    assert finished.frame.camera == _document().path.keyframes[-1].camera

    restarted = controller.play()
    assert restarted.state is PlaybackState.PLAYING
    assert restarted.playhead_seconds == pytest.approx(0.0)


def test_stop_resets_and_seek_clamps_to_plan_bounds() -> None:
    controller = FlightPlanPlaybackController(clock=FakeClock())
    controller.load(_document())

    assert controller.seek(-4).playhead_seconds == 0.0
    assert controller.seek(99).playhead_seconds == 5.0
    assert controller.stop().playhead_seconds == 0.0


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_playback_rate_validation(value: float) -> None:
    controller = FlightPlanPlaybackController(clock=FakeClock())
    controller.load(_document())

    with pytest.raises(ValueError, match="playback rate"):
        controller.set_rate(value)


def test_controller_requires_a_loaded_document() -> None:
    controller = FlightPlanPlaybackController(clock=FakeClock())

    with pytest.raises(RuntimeError, match="no flight plan"):
        controller.play()
