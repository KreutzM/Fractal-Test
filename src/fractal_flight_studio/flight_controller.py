from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp
import numpy as np

from .camera import CameraState


@dataclass(frozen=True, slots=True)
class FlightStep:
    camera: CameraState
    reached_limit: bool


class FlightController:
    """Own flight state independently of Tk widgets and render backends."""

    def __init__(self) -> None:
        self.target_text: tuple[str, str] | None = None
        self.running = False
        self.pending_final_quality_check = False
        self.last_good_camera: CameraState | None = None
        self.last_good_rgb: np.ndarray | None = None

    def set_target(self, center_x_text: str, center_y_text: str) -> None:
        self.target_text = (center_x_text, center_y_text)

    def start(self, camera: CameraState, rgb: np.ndarray | None) -> bool:
        if self.target_text is None:
            return False
        self.running = True
        self.pending_final_quality_check = False
        self.last_good_camera = camera
        self.last_good_rgb = rgb.copy() if rgb is not None else None
        return True

    def stop(self, *, numerical_limit: bool = False) -> None:
        if self.running and numerical_limit:
            self.pending_final_quality_check = True
        elif not numerical_limit:
            self.pending_final_quality_check = False
        self.running = False

    def should_check_result(self, generation: int, current_generation: int) -> bool:
        return self.running or (
            self.pending_final_quality_check and generation == current_generation
        )

    def accept(self, camera: CameraState, rgb: np.ndarray) -> None:
        self.last_good_camera = camera
        self.last_good_rgb = rgb.copy()
        self.pending_final_quality_check = False

    def reject(self) -> tuple[CameraState | None, np.ndarray | None]:
        self.running = False
        self.pending_final_quality_check = False
        return self.last_good_camera, self.last_good_rgb

    def step(
        self,
        camera: CameraState,
        *,
        zoom_rate: float,
        minimum_width: mp.mpf,
        digits: int,
        attraction: str | float = "0.08",
    ) -> FlightStep:
        if not self.running or self.target_text is None:
            raise RuntimeError("flight is not running")
        next_camera, reached_limit = camera.approach(
            self.target_text,
            zoom_rate=zoom_rate,
            attraction=attraction,
            minimum_width=minimum_width,
            digits=digits,
        )
        return FlightStep(next_camera, reached_limit)
