from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import mpmath as mp

from .camera import CameraState


class CenterInterpolation(str, Enum):
    """How the camera center moves through an outgoing keyframe segment."""

    LINEAR = "linear"
    FOCUS = "focus"


class Easing(str, Enum):
    """Interpolation curve used by a keyframe's outgoing segment."""

    LINEAR = "linear"
    SMOOTHSTEP = "smoothstep"
    SMOOTHERSTEP = "smootherstep"
    STEP = "step"

    def apply(self, progress: mp.mpf) -> mp.mpf:
        if progress <= 0:
            return mp.mpf("0")
        if progress >= 1:
            return mp.mpf("1")
        if self is Easing.LINEAR:
            return progress
        if self is Easing.STEP:
            return mp.mpf("0")
        if self is Easing.SMOOTHSTEP:
            return progress * progress * (3 - 2 * progress)
        return progress * progress * progress * (
            progress * (progress * 6 - 15) + 10
        )


@dataclass(frozen=True, slots=True)
class FlightKeyframe:
    """One exact camera sample on a deterministic flight timeline."""

    time_seconds_text: str
    camera: CameraState
    easing: Easing | str = Easing.SMOOTHSTEP
    center_interpolation: CenterInterpolation | str = CenterInterpolation.LINEAR

    def __post_init__(self) -> None:
        object.__setattr__(self, "easing", Easing(self.easing))
        object.__setattr__(
            self,
            "center_interpolation",
            CenterInterpolation(self.center_interpolation),
        )

    def time_seconds(self, *, digits: int) -> mp.mpf:
        try:
            with mp.workdps(digits):
                value = mp.mpf(self.time_seconds_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("keyframe time must be a decimal number") from exc
        if not mp.isfinite(value) or value < 0:
            raise ValueError("keyframe time must be finite and non-negative")
        self.camera.values(digits=digits)
        return value


@dataclass(frozen=True, slots=True)
class CameraPath:
    """Immutable, deterministic interpolation through exact camera keyframes."""

    keyframes: tuple[FlightKeyframe, ...] | Iterable[FlightKeyframe]
    digits: int = 80
    _times: tuple[mp.mpf, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.digits < 20:
            raise ValueError("path precision must be at least 20 decimal digits")
        frames = tuple(self.keyframes)
        if len(frames) < 2:
            raise ValueError("a camera path requires at least two keyframes")
        times = tuple(frame.time_seconds(digits=self.digits) for frame in frames)
        if times[0] != 0:
            raise ValueError("the first keyframe must start at zero seconds")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("keyframe times must be strictly increasing")
        object.__setattr__(self, "keyframes", frames)
        object.__setattr__(self, "_times", times)

    @property
    def duration_text(self) -> str:
        return self.keyframes[-1].time_seconds_text

    def evaluate(self, time_seconds: str | int | float | mp.mpf) -> CameraState:
        with mp.workdps(self.digits):
            time_value = _to_mpf(time_seconds)
            if not mp.isfinite(time_value):
                raise ValueError("evaluation time must be finite")
            if time_value <= self._times[0]:
                return self.keyframes[0].camera
            if time_value >= self._times[-1]:
                return self.keyframes[-1].camera

            segment_index = next(
                index
                for index, end_time in enumerate(self._times[1:])
                if time_value <= end_time
            )
            start = self.keyframes[segment_index]
            end = self.keyframes[segment_index + 1]
            start_time = self._times[segment_index]
            end_time = self._times[segment_index + 1]
            progress = (time_value - start_time) / (end_time - start_time)
            eased = start.easing.apply(progress)

            start_x, start_y, start_width = start.camera.values(digits=self.digits)
            end_x, end_y, end_width = end.camera.values(digits=self.digits)
            log_width = mp.log(start_width) + (
                mp.log(end_width) - mp.log(start_width)
            ) * eased
            view_width = mp.exp(log_width)
            center_progress = eased
            if (
                start.center_interpolation is CenterInterpolation.FOCUS
                and start_width != end_width
            ):
                # Tie center movement to the current zoom scale. During a deep
                # zoom-in this moves the camera toward the destination early,
                # keeping that destination at an approximately stable screen
                # position instead of zooming into arbitrary points between
                # the two centers. The symmetric zoom-out behavior keeps the
                # starting detail stable until enough context is visible.
                center_progress = (start_width - view_width) / (
                    start_width - end_width
                )
                center_progress = min(mp.mpf("1"), max(mp.mpf("0"), center_progress))
            center_x = start_x + (end_x - start_x) * center_progress
            center_y = start_y + (end_y - start_y) * center_progress
            return CameraState.from_values(
                center_x,
                center_y,
                view_width,
                digits=self.digits,
            )


def _to_mpf(value: str | int | float | mp.mpf) -> mp.mpf:
    try:
        if isinstance(value, float):
            return mp.mpf(repr(value))
        return mp.mpf(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation time must be a decimal number") from exc
