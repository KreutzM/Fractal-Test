from __future__ import annotations

from dataclasses import dataclass
import math

from .models import Viewport


@dataclass(frozen=True, slots=True)
class FlightPath:
    start: Viewport
    end: Viewport

    def viewport_at(self, t: float) -> Viewport:
        if t <= 0.0:
            return self.start
        if t >= 1.0:
            return self.end
        eased = t * t * (3.0 - 2.0 * t)
        if self.start.width <= 0 or self.end.width <= 0:
            raise ValueError("viewport widths must be positive")
        log_width = math.log(self.start.width) * (1.0 - eased) + math.log(self.end.width) * eased
        return Viewport(
            center_x=self.start.center_x * (1.0 - eased) + self.end.center_x * eased,
            center_y=self.start.center_y * (1.0 - eased) + self.end.center_y * eased,
            width=math.exp(log_width),
        )
