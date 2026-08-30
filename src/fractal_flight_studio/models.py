from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class FractalKind(str, Enum):
    MANDELBROT = "mandelbrot"
    JULIA = "julia"
    BURNING_SHIP = "burning_ship"
    MULTIBROT = "multibrot"
    NEWTON = "newton"


class Precision(str, Enum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class RenderMode(str, Enum):
    AUTO = "auto"
    DIRECT = "direct"
    PERTURBATION = "perturbation"


@dataclass(frozen=True, slots=True)
class Viewport:
    center_x: float = -0.5
    center_y: float = 0.0
    width: float = 3.5

    def pixel_to_complex(
        self, px: float, py: float, image_width: int, image_height: int
    ) -> tuple[float, float]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        aspect = image_height / image_width
        x = self.center_x + (px / image_width - 0.5) * self.width
        y = self.center_y + (py / image_height - 0.5) * self.width * aspect
        return x, y

    def zoom_at(
        self,
        px: float,
        py: float,
        image_width: int,
        image_height: int,
        factor: float,
    ) -> "Viewport":
        if factor <= 0:
            raise ValueError("zoom factor must be positive")
        before_x, before_y = self.pixel_to_complex(px, py, image_width, image_height)
        updated = replace(self, width=self.width / factor)
        after_x, after_y = updated.pixel_to_complex(px, py, image_width, image_height)
        return replace(
            updated,
            center_x=updated.center_x + before_x - after_x,
            center_y=updated.center_y + before_y - after_y,
        )

    def pan_pixels(
        self, dx: float, dy: float, image_width: int, image_height: int
    ) -> "Viewport":
        if image_width <= 0 or image_height <= 0:
            return self
        units_per_pixel = self.width / image_width
        return replace(
            self,
            center_x=self.center_x - dx * units_per_pixel,
            center_y=self.center_y - dy * units_per_pixel,
        )


@dataclass(frozen=True, slots=True)
class RenderRequest:
    width: int = 960
    height: int = 640
    viewport: Viewport = Viewport()
    fractal: FractalKind = FractalKind.MANDELBROT
    max_iterations: int = 400
    color_iterations: int | None = None
    escape_radius: float = 4.0
    julia_c_real: float = -0.8
    julia_c_imag: float = 0.156
    exponent: int = 3
    precision: Precision = Precision.FLOAT32
    render_mode: RenderMode = RenderMode.AUTO
    reference_bits: int = 256
    center_x_text: str | None = None
    center_y_text: str | None = None
    view_width_text: str | None = None

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not 1 <= self.max_iterations <= 100_000:
            raise ValueError("max_iterations must be between 1 and 100000")
        if self.color_iterations is not None:
            if isinstance(self.color_iterations, bool) or not isinstance(
                self.color_iterations, int
            ):
                raise ValueError("color_iterations must be an integer or None")
            if not self.max_iterations <= self.color_iterations <= 100_000:
                raise ValueError(
                    "color_iterations must be between max_iterations and 100000"
                )
        if self.escape_radius <= 1.0:
            raise ValueError("escape_radius must be greater than 1")
        if not 2 <= self.exponent <= 8:
            raise ValueError("exponent must be between 2 and 8")
        if self.viewport.width <= 0 and not self.view_width_text:
            raise ValueError("viewport width must be positive")
        if not 64 <= self.reference_bits <= 16384:
            raise ValueError("reference_bits must be between 64 and 16384")

    @property
    def effective_color_iterations(self) -> int:
        """Stable iteration scale used only for coloring escaped pixels."""

        return (
            self.max_iterations
            if self.color_iterations is None
            else self.color_iterations
        )
