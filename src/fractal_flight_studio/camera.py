from __future__ import annotations

from dataclasses import dataclass
import math

import mpmath as mp

from .models import FractalKind, RenderRequest, Viewport


@dataclass(frozen=True, slots=True)
class CameraState:
    """High-precision camera state stored as exact decimal text."""

    center_x_text: str = "-0.5"
    center_y_text: str = "0.0"
    view_width_text: str = "3.5"

    @classmethod
    def for_fractal(cls, fractal: FractalKind) -> "CameraState":
        if fractal is FractalKind.NEWTON:
            return cls("0.0", "0.0", "4.0")
        if fractal is FractalKind.BURNING_SHIP:
            return cls("-0.5", "-0.5", "3.5")
        return cls()

    @classmethod
    def from_request(cls, request: RenderRequest) -> "CameraState":
        return cls(
            request.center_x_text or repr(request.viewport.center_x),
            request.center_y_text or repr(request.viewport.center_y),
            request.view_width_text or repr(request.viewport.width),
        )

    @classmethod
    def from_values(
        cls,
        center_x: mp.mpf,
        center_y: mp.mpf,
        width: mp.mpf,
        *,
        digits: int,
    ) -> "CameraState":
        if width <= 0:
            raise ValueError("camera width must be positive")
        return cls(
            mp.nstr(center_x, n=digits),
            mp.nstr(center_y, n=digits),
            mp.nstr(width, n=digits),
        )

    def values(self, *, digits: int) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
        with mp.workdps(digits):
            center_x = mp.mpf(self.center_x_text)
            center_y = mp.mpf(self.center_y_text)
            width = mp.mpf(self.view_width_text)
        if width <= 0:
            raise ValueError("camera width must be positive")
        return center_x, center_y, width

    @staticmethod
    def _proxy_float(text: str, *, digits: int, minimum_positive: float | None = None) -> float:
        with mp.workdps(digits):
            value = mp.mpf(text)
        try:
            converted = float(value)
        except OverflowError:
            converted = math.copysign(float("inf"), -1.0 if value < 0 else 1.0)
        if minimum_positive is not None and converted == 0.0 and value > 0:
            return minimum_positive
        return converted

    def proxy_viewport(self, *, digits: int, minimum_positive: float = 1e-300) -> Viewport:
        return Viewport(
            center_x=self._proxy_float(self.center_x_text, digits=digits),
            center_y=self._proxy_float(self.center_y_text, digits=digits),
            width=self._proxy_float(
                self.view_width_text,
                digits=digits,
                minimum_positive=minimum_positive,
            ),
        )

    def pixel_to_complex(
        self,
        px: float,
        py: float,
        image_width: int,
        image_height: int,
        *,
        digits: int,
    ) -> tuple[mp.mpf, mp.mpf]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        with mp.workdps(digits):
            center_x, center_y, width = self.values(digits=digits)
            aspect = mp.mpf(image_height) / mp.mpf(image_width)
            x = center_x + (mp.mpf(px) / image_width - mp.mpf("0.5")) * width
            y = center_y + (mp.mpf(py) / image_height - mp.mpf("0.5")) * width * aspect
            return x, y

    def zoom_at(
        self,
        px: float,
        py: float,
        image_width: int,
        image_height: int,
        factor: float,
        *,
        digits: int,
    ) -> "CameraState":
        if factor <= 0:
            raise ValueError("zoom factor must be positive")
        with mp.workdps(digits):
            before_x, before_y = self.pixel_to_complex(
                px,
                py,
                image_width,
                image_height,
                digits=digits,
            )
            center_x, center_y, width = self.values(digits=digits)
            zoomed = self.from_values(
                center_x,
                center_y,
                width / mp.mpf(factor),
                digits=digits,
            )
            after_x, after_y = zoomed.pixel_to_complex(
                px,
                py,
                image_width,
                image_height,
                digits=digits,
            )
            next_x, next_y, next_width = zoomed.values(digits=digits)
            return self.from_values(
                next_x + before_x - after_x,
                next_y + before_y - after_y,
                next_width,
                digits=digits,
            )

    def pan_pixels(
        self,
        dx_pixels: float,
        dy_pixels: float,
        image_width: int,
        *,
        digits: int,
    ) -> "CameraState":
        if image_width <= 0:
            raise ValueError("image width must be positive")
        with mp.workdps(digits):
            center_x, center_y, width = self.values(digits=digits)
            units_per_pixel = width / image_width
            return self.from_values(
                center_x - mp.mpf(dx_pixels) * units_per_pixel,
                center_y - mp.mpf(dy_pixels) * units_per_pixel,
                width,
                digits=digits,
            )

    def approach(
        self,
        target_text: tuple[str, str],
        *,
        zoom_rate: float,
        attraction: str | float,
        minimum_width: mp.mpf,
        digits: int,
    ) -> tuple["CameraState", bool]:
        if zoom_rate <= 1.0:
            raise ValueError("zoom rate must be greater than one")
        if minimum_width <= 0:
            raise ValueError("minimum width must be positive")
        with mp.workdps(digits):
            center_x, center_y, width = self.values(digits=digits)
            target_x = mp.mpf(target_text[0])
            target_y = mp.mpf(target_text[1])
            attraction_value = mp.mpf(attraction)
            next_width = width / mp.mpf(zoom_rate)
            reached_limit = next_width <= minimum_width
            next_camera = self.from_values(
                center_x + (target_x - center_x) * attraction_value,
                center_y + (target_y - center_y) * attraction_value,
                minimum_width if reached_limit else next_width,
                digits=digits,
            )
            return next_camera, reached_limit
