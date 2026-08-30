from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Callable

from .flight_plan import EvaluatedFlightFrame, FlightPlanDocument


Clock = Callable[[], float]


class PlaybackState(str, Enum):
    """Lifecycle of deterministic wall-clock flight-plan playback."""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class PlaybackSample:
    """One evaluated playback position and its exact flight-plan frame."""

    state: PlaybackState
    playhead_seconds: float
    duration_seconds: float
    playback_rate: float
    frame: EvaluatedFlightFrame
    reached_end: bool = False


class FlightPlanPlaybackController:
    """Tk-independent wall-clock controller for one immutable flight plan.

    The playhead is derived from a monotonic clock rather than advanced by a
    fixed frame delta. Rendering code may therefore skip obsolete frames while
    the timeline itself remains temporally correct.
    """

    def __init__(self, *, clock: Clock = time.monotonic) -> None:
        self._clock = clock
        self._document: FlightPlanDocument | None = None
        self._state = PlaybackState.STOPPED
        self._playhead_seconds = 0.0
        self._playback_rate = 1.0
        self._anchor_clock_seconds: float | None = None
        self._anchor_playhead_seconds = 0.0

    @property
    def document(self) -> FlightPlanDocument | None:
        return self._document

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def playing(self) -> bool:
        return self._state is PlaybackState.PLAYING

    @property
    def paused(self) -> bool:
        return self._state is PlaybackState.PAUSED

    @property
    def loaded(self) -> bool:
        return self._document is not None

    @property
    def duration_seconds(self) -> float:
        document = self._document
        return 0.0 if document is None else float(document.duration_text)

    @property
    def playhead_seconds(self) -> float:
        return self._resolved_playhead(self._clock(), commit=False)

    @property
    def playback_rate(self) -> float:
        return self._playback_rate

    def load(
        self,
        document: FlightPlanDocument,
        *,
        playhead_seconds: float = 0.0,
    ) -> PlaybackSample:
        self._document = document
        self._state = PlaybackState.STOPPED
        self._anchor_clock_seconds = None
        self._playhead_seconds = self._clamp_time(playhead_seconds)
        self._anchor_playhead_seconds = self._playhead_seconds
        return self.sample()

    def unload(self) -> None:
        self._document = None
        self._state = PlaybackState.STOPPED
        self._playhead_seconds = 0.0
        self._anchor_playhead_seconds = 0.0
        self._anchor_clock_seconds = None

    def set_rate(self, playback_rate: float) -> PlaybackSample:
        rate = float(playback_rate)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("playback rate must be a positive finite number")
        now = self._clock()
        self._playhead_seconds = self._resolved_playhead(now, commit=True)
        self._playback_rate = rate
        if self.playing:
            self._anchor_clock_seconds = now
            self._anchor_playhead_seconds = self._playhead_seconds
        return self.sample(now=now)

    def play(self) -> PlaybackSample:
        self._require_document()
        now = self._clock()
        current = self._resolved_playhead(now, commit=True)
        if current >= self.duration_seconds:
            current = 0.0
            self._playhead_seconds = 0.0
        self._state = PlaybackState.PLAYING
        self._anchor_clock_seconds = now
        self._anchor_playhead_seconds = current
        return self.sample(now=now)

    def pause(self) -> PlaybackSample:
        self._require_document()
        now = self._clock()
        self._playhead_seconds = self._resolved_playhead(now, commit=True)
        self._anchor_playhead_seconds = self._playhead_seconds
        self._anchor_clock_seconds = None
        self._state = PlaybackState.PAUSED
        return self.sample(now=now)

    def stop(self, *, reset: bool = True) -> PlaybackSample:
        self._require_document()
        now = self._clock()
        current = self._resolved_playhead(now, commit=True)
        self._state = PlaybackState.STOPPED
        self._anchor_clock_seconds = None
        self._playhead_seconds = 0.0 if reset else current
        self._anchor_playhead_seconds = self._playhead_seconds
        return self.sample(now=now)

    def seek(self, playhead_seconds: float) -> PlaybackSample:
        self._require_document()
        target = self._clamp_time(playhead_seconds)
        now = self._clock()
        self._playhead_seconds = target
        self._anchor_playhead_seconds = target
        self._anchor_clock_seconds = now if self.playing else None
        return self.sample(now=now)

    def sample(self, *, now: float | None = None) -> PlaybackSample:
        document = self._require_document()
        current_now = self._clock() if now is None else float(now)
        playhead = self._resolved_playhead(current_now, commit=True)
        reached_end = self.playing and playhead >= self.duration_seconds
        if reached_end:
            self._state = PlaybackState.STOPPED
            self._anchor_clock_seconds = None
            self._playhead_seconds = self.duration_seconds
            self._anchor_playhead_seconds = self._playhead_seconds
            playhead = self._playhead_seconds
        frame = document.evaluate(playhead)
        return PlaybackSample(
            state=self._state,
            playhead_seconds=playhead,
            duration_seconds=self.duration_seconds,
            playback_rate=self._playback_rate,
            frame=frame,
            reached_end=reached_end,
        )

    def _resolved_playhead(self, now: float, *, commit: bool) -> float:
        if not self.playing or self._anchor_clock_seconds is None:
            return self._playhead_seconds
        elapsed = max(0.0, float(now) - self._anchor_clock_seconds)
        value = self._clamp_time(
            self._anchor_playhead_seconds + elapsed * self._playback_rate
        )
        if commit:
            self._playhead_seconds = value
        return value

    def _clamp_time(self, value: float) -> float:
        seconds = float(value)
        if not math.isfinite(seconds):
            raise ValueError("playhead time must be finite")
        return min(self.duration_seconds, max(0.0, seconds))

    def _require_document(self) -> FlightPlanDocument:
        if self._document is None:
            raise RuntimeError("no flight plan is loaded for playback")
        return self._document
