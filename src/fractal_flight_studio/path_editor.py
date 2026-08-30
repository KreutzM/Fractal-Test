from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import mpmath as mp

from .camera import CameraState
from .flight_path import (
    CameraPath,
    CenterInterpolation,
    Easing,
    FlightKeyframe,
)


@dataclass(frozen=True, slots=True)
class CameraPathDraft:
    """Immutable editable collection of exact camera keyframes.

    Unlike :class:`CameraPath`, a draft may temporarily contain fewer than two
    keyframes or start after zero seconds. Every individual keyframe remains
    numerically valid and timeline positions remain unique.
    """

    keyframes: tuple[FlightKeyframe, ...] | Iterable[FlightKeyframe] = ()
    digits: int = 80
    _times: tuple[mp.mpf, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.digits < 20:
            raise ValueError("draft precision must be at least 20 decimal digits")
        frames = tuple(self.keyframes)
        timed_frames: list[tuple[mp.mpf, FlightKeyframe]] = []
        for frame in frames:
            timed_frames.append((frame.time_seconds(digits=self.digits), frame))
        timed_frames.sort(key=lambda item: item[0])
        times = tuple(item[0] for item in timed_frames)
        if any(right == left for left, right in zip(times, times[1:])):
            raise ValueError("keyframe times must be unique")
        object.__setattr__(self, "keyframes", tuple(item[1] for item in timed_frames))
        object.__setattr__(self, "_times", times)

    @classmethod
    def from_path(cls, path: CameraPath) -> "CameraPathDraft":
        return cls(path.keyframes, digits=path.digits)

    @property
    def duration_text(self) -> str | None:
        if not self.keyframes:
            return None
        return self.keyframes[-1].time_seconds_text

    @property
    def validation_error(self) -> str | None:
        if len(self.keyframes) < 2:
            return "Mindestens zwei Keyframes sind erforderlich."
        if self._times[0] != 0:
            return "Der erste Keyframe muss bei 0 Sekunden beginnen."
        try:
            CameraPath(self.keyframes, digits=self.digits)
        except ValueError as exc:
            return str(exc)
        return None

    @property
    def valid(self) -> bool:
        return self.validation_error is None

    def build_path(self) -> CameraPath:
        error = self.validation_error
        if error is not None:
            raise ValueError(error)
        return CameraPath(self.keyframes, digits=self.digits)

    def suggested_time_text(self, *, step_text: str = "5") -> str:
        with mp.workdps(self.digits):
            try:
                step = mp.mpf(step_text)
            except (TypeError, ValueError) as exc:
                raise ValueError("suggested timeline step must be a decimal number") from exc
            if not mp.isfinite(step) or step <= 0:
                raise ValueError("suggested timeline step must be finite and positive")
            if not self._times:
                return "0"
            return _format_decimal(self._times[-1] + step, digits=self.digits)

    def add_keyframe(
        self,
        time_seconds_text: str,
        camera: CameraState,
        easing: Easing | str = Easing.SMOOTHSTEP,
        center_interpolation: CenterInterpolation | str = CenterInterpolation.LINEAR,
        *,
        replace_existing: bool = False,
    ) -> "CameraPathDraft":
        candidate = FlightKeyframe(
            time_seconds_text, camera, easing, center_interpolation
        )
        candidate_time = candidate.time_seconds(digits=self.digits)
        frames = list(self.keyframes)
        matching_index = next(
            (index for index, value in enumerate(self._times) if value == candidate_time),
            None,
        )
        if matching_index is not None:
            if not replace_existing:
                raise ValueError(f"Bei {time_seconds_text} Sekunden existiert bereits ein Keyframe.")
            frames[matching_index] = candidate
        else:
            frames.append(candidate)
        return CameraPathDraft(frames, digits=self.digits)

    def update_keyframe(
        self,
        index: int,
        *,
        time_seconds_text: str | None = None,
        camera: CameraState | None = None,
        easing: Easing | str | None = None,
        center_interpolation: CenterInterpolation | str | None = None,
    ) -> "CameraPathDraft":
        current = self.keyframes[index]
        candidate = FlightKeyframe(
            current.time_seconds_text if time_seconds_text is None else time_seconds_text,
            current.camera if camera is None else camera,
            current.easing if easing is None else easing,
            (
                current.center_interpolation
                if center_interpolation is None
                else center_interpolation
            ),
        )
        frames = list(self.keyframes)
        del frames[index]
        candidate_time = candidate.time_seconds(digits=self.digits)
        for frame in frames:
            if frame.time_seconds(digits=self.digits) == candidate_time:
                raise ValueError(
                    f"Bei {candidate.time_seconds_text} Sekunden existiert bereits ein Keyframe."
                )
        frames.append(candidate)
        return CameraPathDraft(frames, digits=self.digits)

    def remove_keyframe(self, index: int) -> "CameraPathDraft":
        frames = list(self.keyframes)
        del frames[index]
        return CameraPathDraft(frames, digits=self.digits)

    def clear(self) -> "CameraPathDraft":
        return CameraPathDraft(digits=self.digits)


def _format_decimal(value: mp.mpf, *, digits: int) -> str:
    return mp.nstr(value, n=digits, min_fixed=-6, max_fixed=12)
