from __future__ import annotations

from dataclasses import dataclass
import math

import mpmath as mp
import numpy as np

from .models import FractalKind, Precision, RenderMode, RenderRequest

_LOG10_2 = math.log10(2.0)

# The Pauldelbrot-style glitch criterion flags catastrophic cancellation when
# the actual orbit becomes many orders of magnitude smaller than its reference
# orbit. The pixel is repaired by rebasing its *representation* to Z_0 = 0;
# no absolute deep-zoom coordinate is converted to float64.
GLITCH_TOLERANCE = 1e-12
MIN_REFERENCE_MAGNITUDE_SQUARED = 1e-300

# A reference remains fixed across ordinary pan/zoom interaction. A flight
# target selected inside the viewport therefore keeps the same reference orbit
# throughout the flight. Re-anchoring happens only after moving several source
# view widths away or zooming substantially outward.
REFERENCE_PAN_RADIUS = 4.0
REFERENCE_ZOOM_OUT_LIMIT = 4.0

# Keep a small numerical safety margin before coordinate quantization becomes
# visible as blocky or single-colour frames. Automatic rendering may promote a
# requested float32 direct kernel to float64 before perturbation takes over.
DIRECT_ULP_GUARD = 8.0
REFERENCE_GUARD_BITS = 16
PERTURBATION_NORMAL_GUARD = 64.0
PIXEL_GRID_ULP_GUARD = 4.0


class PixelGridExhaustedError(RuntimeError):
    """Raised when a fresh renderer pixel grid can no longer resolve neighbours."""

    def __init__(self, quality: "PixelGridQuality") -> None:
        self.quality = quality
        super().__init__(
            "renderer pixel grid no longer resolves neighbouring coordinates "
            f"(X {quality.x_unique_fraction:.1%}, Y {quality.y_unique_fraction:.1%}, "
            f"largest equal run {quality.maximum_equal_run})"
        )


@dataclass(frozen=True, slots=True)
class PixelGridQuality:
    x_unique_fraction: float
    y_unique_fraction: float
    minimum_ulp_margin: float
    maximum_equal_run: int
    safe: bool

    def as_details(self) -> dict[str, float | int | bool]:
        return {
            "pixel_grid_safe": self.safe,
            "pixel_grid_x_unique_fraction": self.x_unique_fraction,
            "pixel_grid_y_unique_fraction": self.y_unique_fraction,
            "pixel_grid_minimum_ulp_margin": self.minimum_ulp_margin,
            "pixel_grid_maximum_equal_run": self.maximum_equal_run,
        }


@dataclass(slots=True)
class ReferenceOrbit:
    anchor_x_text: str
    anchor_y_text: str
    anchor_width_text: str
    orbit_real: np.ndarray
    orbit_imag: np.ndarray
    reference_bits: int
    max_iterations: int
    rebase_limit: int
    escape_radius: float

    @property
    def key(self) -> tuple[str, str, int, int, float]:
        return (
            self.anchor_x_text,
            self.anchor_y_text,
            self.reference_bits,
            self.max_iterations,
            self.escape_radius,
        )


@dataclass(slots=True)
class PerturbationData:
    center_x_text: str
    center_y_text: str
    view_width_text: str
    reference_anchor_x_text: str
    reference_anchor_y_text: str
    x0_rel: float
    y0_rel: float
    dx: float
    dy: float
    orbit_real: np.ndarray
    orbit_imag: np.ndarray
    reference_bits: int
    reference_key: tuple[str, str, int, int, float]
    reference_reused: bool
    reference_rebase_limit: int
    grid_quality: PixelGridQuality
    reference_reanchored_for_grid: bool = False


@dataclass(slots=True)
class ViewText:
    center_x: str
    center_y: str
    width: str


def digits_for_bits(bits: int) -> int:
    return max(30, int(math.ceil(bits * _LOG10_2)) + 8)


def request_view_text(request: RenderRequest) -> ViewText:
    return ViewText(
        center_x=request.center_x_text if request.center_x_text is not None else repr(request.viewport.center_x),
        center_y=request.center_y_text if request.center_y_text is not None else repr(request.viewport.center_y),
        width=request.view_width_text if request.view_width_text is not None else repr(request.viewport.width),
    )


def direct_pixel_size(request: RenderRequest) -> float:
    view = request_view_text(request)
    try:
        width = abs(float(view.width))
    except Exception:
        width = abs(request.viewport.width)
    return width / max(1, request.width)


def _view_center_as_float(request: RenderRequest) -> tuple[float, float]:
    view = request_view_text(request)
    try:
        cx = float(view.center_x)
    except Exception:
        cx = request.viewport.center_x
    try:
        cy = float(view.center_y)
    except Exception:
        cy = request.viewport.center_y
    return cx, cy


def _coordinate_ulp(value: float, precision: Precision) -> float:
    if precision is Precision.FLOAT32:
        scalar = np.float32(value)
        if not np.isfinite(scalar):
            return float("inf")
        return abs(float(np.spacing(scalar)))
    if not math.isfinite(value):
        return float("inf")
    return math.ulp(value)


def _maximum_equal_run(differences: np.ndarray) -> int:
    maximum = 1
    current = 1
    for difference in differences:
        if difference == 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 1
    return maximum


def _axis_grid_quality(origin, step, count: int) -> tuple[float, float, int]:
    if count <= 1:
        return 1.0, float("inf"), 1

    dtype = np.result_type(origin, step)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        dtype = np.dtype(np.float64)
    scalar = dtype.type
    indices = np.arange(count, dtype=dtype)
    values = scalar(origin) + indices * scalar(step)
    differences = np.diff(values)
    distinct = differences != scalar(0)
    unique_fraction = (int(np.count_nonzero(distinct)) + 1) / count
    maximum_equal_run = _maximum_equal_run(differences)

    spacing = np.maximum(np.abs(np.spacing(values[:-1])), np.abs(np.spacing(values[1:])))
    ratios = np.zeros_like(differences, dtype=np.float64)
    valid = np.isfinite(differences) & np.isfinite(spacing) & (spacing > 0)
    ratios[valid] = np.abs(differences[valid].astype(np.float64)) / spacing[valid].astype(np.float64)
    minimum_ulp_margin = float(np.min(ratios)) if ratios.size else float("inf")
    return unique_fraction, minimum_ulp_margin, maximum_equal_run


def pixel_grid_quality(
    x0,
    y0,
    dx,
    dy,
    width: int,
    height: int,
    *,
    ulp_guard: float = PIXEL_GRID_ULP_GUARD,
) -> PixelGridQuality:
    """Measure whether the renderer can still distinguish neighbouring pixels."""

    x_unique, x_margin, x_run = _axis_grid_quality(x0, dx, width)
    y_unique, y_margin, y_run = _axis_grid_quality(y0, dy, height)
    minimum_margin = min(x_margin, y_margin)
    safe = (
        x_unique == 1.0
        and y_unique == 1.0
        and minimum_margin >= ulp_guard
        and math.isfinite(minimum_margin)
    )
    return PixelGridQuality(
        x_unique_fraction=x_unique,
        y_unique_fraction=y_unique,
        minimum_ulp_margin=minimum_margin,
        maximum_equal_run=max(x_run, y_run),
        safe=safe,
    )


def direct_pixel_grid_quality(request: RenderRequest, precision: Precision) -> PixelGridQuality:
    """Measure the exact direct-kernel coordinate grid for one request."""

    view = request_view_text(request)
    dps = digits_for_bits(request.reference_bits)
    with mp.workdps(dps):
        width = mp.mpf(view.width)
        height = width * mp.mpf(request.height) / mp.mpf(request.width)
        dx = width / request.width
        dy = height / request.height
        x0 = mp.mpf(view.center_x) - width / 2 + dx / 2
        y0 = mp.mpf(view.center_y) - height / 2 + dy / 2
    scalar = np.float32 if precision is Precision.FLOAT32 else np.float64
    return pixel_grid_quality(
        scalar(x0),
        scalar(y0),
        scalar(dx),
        scalar(dy),
        request.width,
        request.height,
    )


def direct_precision_pixel_limit(request: RenderRequest, precision: Precision) -> float:
    """Smallest safe direct-kernel pixel spacing near the current center."""
    cx, cy = _view_center_as_float(request)
    ulp = max(_coordinate_ulp(cx, precision), _coordinate_ulp(cy, precision))
    return ulp * DIRECT_ULP_GUARD


def effective_direct_precision(request: RenderRequest) -> Precision:
    """Promote float32 to float64 before float32 coordinates quantize."""
    if request.render_mode is not RenderMode.AUTO or request.precision is Precision.FLOAT64:
        return request.precision
    if direct_pixel_size(request) <= direct_precision_pixel_limit(request, Precision.FLOAT32):
        return Precision.FLOAT64
    return Precision.FLOAT32


def should_use_perturbation(request: RenderRequest) -> bool:
    if request.fractal is not FractalKind.MANDELBROT:
        return False
    if request.render_mode is RenderMode.PERTURBATION:
        return True
    if request.render_mode is RenderMode.DIRECT:
        return False
    precision = effective_direct_precision(request)
    return direct_pixel_size(request) <= direct_precision_pixel_limit(request, precision)


def minimum_safe_view_width(request: RenderRequest) -> mp.mpf:
    """Return a conservative viewport-width floor for interactive navigation.

    Mandelbrot auto/perturbation mode is limited by both the configured
    high-precision coordinate budget and normal float64 perturbation deltas.
    Other modes are limited by the safest available direct kernel.
    """
    view = request_view_text(request)
    if request.fractal is FractalKind.MANDELBROT and request.render_mode is not RenderMode.DIRECT:
        dps = digits_for_bits(request.reference_bits)
        with mp.workdps(dps):
            cx = abs(mp.mpf(view.center_x))
            cy = abs(mp.mpf(view.center_y))
            coordinate_scale = max(mp.mpf(1), cx, cy)
            usable_bits = max(1, int(request.reference_bits) - REFERENCE_GUARD_BITS)
            reference_pixel_floor = coordinate_scale * mp.power(2, -usable_bits)
            float64_pixel_floor = mp.mpf(str(np.finfo(np.float64).tiny)) * PERTURBATION_NORMAL_GUARD
            return max(reference_pixel_floor, float64_pixel_floor) * max(1, request.width)

    precision = Precision.FLOAT64 if request.render_mode is RenderMode.AUTO else request.precision
    pixel_floor = direct_precision_pixel_limit(request, precision)
    return mp.mpf(repr(pixel_floor)) * max(1, request.width)


def _candidate_offsets() -> tuple[tuple[str, str], ...]:
    # Center first so ties prefer the visually natural reference.
    return (
        ("0", "0"),
        ("-0.35", "0"),
        ("0.35", "0"),
        ("0", "-0.35"),
        ("0", "0.35"),
        ("-0.35", "-0.35"),
        ("-0.35", "0.35"),
        ("0.35", "-0.35"),
        ("0.35", "0.35"),
    )


def _escape_lifetime(cr: mp.mpf, ci: mp.mpf, max_iterations: int, escape_squared: mp.mpf) -> int:
    zr = mp.mpf("0")
    zi = mp.mpf("0")
    for iteration in range(max_iterations):
        next_zr = zr * zr - zi * zi + cr
        next_zi = 2 * zr * zi + ci
        zr, zi = next_zr, next_zi
        if zr * zr + zi * zi > escape_squared:
            return iteration + 1
    return max_iterations


def _select_reference_anchor(request: RenderRequest) -> tuple[str, str]:
    view = request_view_text(request)
    bits = int(request.reference_bits)
    dps = digits_for_bits(bits)
    with mp.workdps(dps):
        center_x = mp.mpf(view.center_x)
        center_y = mp.mpf(view.center_y)
        width = mp.mpf(view.width)
        height = width * mp.mpf(request.height) / mp.mpf(request.width)
        escape_squared = mp.mpf(request.escape_radius) ** 2
        best_x = center_x
        best_y = center_y
        best_lifetime = -1
        for ox_text, oy_text in _candidate_offsets():
            candidate_x = center_x + mp.mpf(ox_text) * width
            candidate_y = center_y + mp.mpf(oy_text) * height
            lifetime = _escape_lifetime(
                candidate_x, candidate_y, request.max_iterations, escape_squared
            )
            if lifetime > best_lifetime:
                best_lifetime = lifetime
                best_x = candidate_x
                best_y = candidate_y
            if lifetime == request.max_iterations:
                break
        digits = digits_for_bits(bits)
        return mp.nstr(best_x, digits), mp.nstr(best_y, digits)


def _build_reference_orbit(request: RenderRequest) -> ReferenceOrbit:
    view = request_view_text(request)
    bits = int(request.reference_bits)
    anchor_x_text, anchor_y_text = _select_reference_anchor(request)
    dps = digits_for_bits(bits)
    with mp.workdps(dps):
        cr = mp.mpf(anchor_x_text)
        ci = mp.mpf(anchor_y_text)
        zr = mp.mpf("0")
        zi = mp.mpf("0")
        escape_squared = mp.mpf(request.escape_radius) ** 2
        orbit_real = np.zeros(request.max_iterations + 1, dtype=np.float64)
        orbit_imag = np.zeros(request.max_iterations + 1, dtype=np.float64)
        rebase_limit = request.max_iterations
        for iteration in range(request.max_iterations):
            next_zr = zr * zr - zi * zi + cr
            next_zi = 2 * zr * zi + ci
            zr, zi = next_zr, next_zi
            orbit_real[iteration + 1] = float(zr)
            orbit_imag[iteration + 1] = float(zi)
            if zr * zr + zi * zi > escape_squared:
                # Z_(iteration+1) is still useful for the current escape/glitch
                # test, but it must not be used as the base for another step.
                rebase_limit = iteration + 1
                break
    return ReferenceOrbit(
        anchor_x_text=anchor_x_text,
        anchor_y_text=anchor_y_text,
        anchor_width_text=view.width,
        orbit_real=orbit_real,
        orbit_imag=orbit_imag,
        reference_bits=bits,
        max_iterations=request.max_iterations,
        rebase_limit=rebase_limit,
        escape_radius=float(request.escape_radius),
    )


def _reference_is_suitable(request: RenderRequest, reference: ReferenceOrbit) -> bool:
    if reference.reference_bits != request.reference_bits:
        return False
    if reference.max_iterations < request.max_iterations:
        return False
    if reference.escape_radius != float(request.escape_radius):
        return False
    view = request_view_text(request)
    dps = digits_for_bits(request.reference_bits)
    with mp.workdps(dps):
        current_width = abs(mp.mpf(view.width))
        anchor_width = abs(mp.mpf(reference.anchor_width_text))
        if current_width > anchor_width * REFERENCE_ZOOM_OUT_LIMIT:
            return False
        dx = mp.mpf(view.center_x) - mp.mpf(reference.anchor_x_text)
        dy = mp.mpf(view.center_y) - mp.mpf(reference.anchor_y_text)
        distance = mp.sqrt(dx * dx + dy * dy)
        return distance <= anchor_width * REFERENCE_PAN_RADIUS


def _prepare_with_reference(
    request: RenderRequest,
    reference: ReferenceOrbit,
    *,
    reused: bool,
) -> PerturbationData:
    view = request_view_text(request)
    dps = digits_for_bits(request.reference_bits)
    with mp.workdps(dps):
        width = mp.mpf(view.width)
        height = width * mp.mpf(request.height) / mp.mpf(request.width)
        dx = width / request.width
        dy = height / request.height
        center_offset_x = mp.mpf(view.center_x) - mp.mpf(reference.anchor_x_text)
        center_offset_y = mp.mpf(view.center_y) - mp.mpf(reference.anchor_y_text)
        x0_rel = center_offset_x - width / 2 + dx / 2
        y0_rel = center_offset_y - height / 2 + dy / 2
        grid_quality = pixel_grid_quality(
            np.float64(x0_rel),
            np.float64(y0_rel),
            np.float64(dx),
            np.float64(dy),
            request.width,
            request.height,
        )
        return PerturbationData(
            center_x_text=view.center_x,
            center_y_text=view.center_y,
            view_width_text=view.width,
            reference_anchor_x_text=reference.anchor_x_text,
            reference_anchor_y_text=reference.anchor_y_text,
            x0_rel=float(x0_rel),
            y0_rel=float(y0_rel),
            dx=float(dx),
            dy=float(dy),
            orbit_real=reference.orbit_real[: request.max_iterations + 1],
            orbit_imag=reference.orbit_imag[: request.max_iterations + 1],
            reference_bits=reference.reference_bits,
            reference_key=reference.key,
            reference_reused=reused,
            reference_rebase_limit=min(reference.rebase_limit, request.max_iterations),
            grid_quality=grid_quality,
        )


class PerturbationReferenceCache:
    """Keeps a stable reference orbit across related frames."""

    def __init__(self) -> None:
        self._reference: ReferenceOrbit | None = None

    @property
    def reference(self) -> ReferenceOrbit | None:
        return self._reference

    def clear(self) -> None:
        self._reference = None

    def prepare(self, request: RenderRequest) -> PerturbationData:
        reused = self._reference is not None and _reference_is_suitable(request, self._reference)
        if not reused:
            self._reference = _build_reference_orbit(request)
        assert self._reference is not None
        perturb = _prepare_with_reference(request, self._reference, reused=reused)
        if reused and not perturb.grid_quality.safe:
            self._reference = _build_reference_orbit(request)
            perturb = _prepare_with_reference(request, self._reference, reused=False)
            perturb.reference_reanchored_for_grid = True
        if not perturb.grid_quality.safe:
            raise PixelGridExhaustedError(perturb.grid_quality)
        return perturb


def prepare_perturbation(request: RenderRequest) -> PerturbationData:
    """Prepare one standalone perturbation frame with a fresh reference."""
    return PerturbationReferenceCache().prepare(request)
