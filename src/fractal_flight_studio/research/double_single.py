from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import mpmath as mp
import numpy as np
from numba import cuda, float32, float64

F32_EPSILON = float(np.finfo(np.float32).eps)
F32_TINY = float(np.finfo(np.float32).tiny)

ESCAPE_HIGH_ONLY = 0
ESCAPE_FULL = 1
ESCAPE_ADAPTIVE = 2


@dataclass(frozen=True, slots=True)
class DoubleSingle:
    hi: np.float32
    lo: np.float32

    def as_float(self) -> float:
        return float(self.hi) + float(self.lo)


@dataclass(frozen=True, slots=True)
class ReferencePointResult:
    escaped: bool
    escape_iteration: int
    smooth_iteration: float
    orbit: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DoubleSinglePointResult:
    escaped: bool
    escape_iteration: int
    smooth_iteration: float
    orbit: tuple[tuple[DoubleSingle, DoubleSingle], ...]


def _f32(value: float | np.floating) -> np.float32:
    return np.float32(value)


def split_mpf(value: mp.mpf | str | float) -> DoubleSingle:
    """Split an arbitrary-precision value without first reducing it to binary64."""

    exact = mp.mpf(value)
    hi = np.float32(mp.nstr(exact, 80))
    if not np.isfinite(hi):
        raise OverflowError(f"value cannot be represented by double-single FP32 components: {value}")
    remainder = exact - mp.mpf(float(hi))
    lo = np.float32(mp.nstr(remainder, 80))
    return renormalize_cpu(hi, lo)


def two_sum_cpu(a: np.float32, b: np.float32) -> DoubleSingle:
    a = _f32(a)
    b = _f32(b)
    s = _f32(a + b)
    bb = _f32(s - a)
    error = _f32(_f32(a - _f32(s - bb)) + _f32(b - bb))
    return DoubleSingle(s, error)


def quick_two_sum_cpu(a: np.float32, b: np.float32) -> DoubleSingle:
    """Error-free transform requiring abs(a) >= abs(b)."""

    a = _f32(a)
    b = _f32(b)
    if abs(float(a)) < abs(float(b)):
        raise ValueError("quick_two_sum requires abs(a) >= abs(b)")
    s = _f32(a + b)
    error = _f32(b - _f32(s - a))
    return DoubleSingle(s, error)


def two_product_cpu(a: np.float32, b: np.float32) -> DoubleSingle:
    """Error-free FP32 product transform, modeling fma.rn.f32."""

    a = _f32(a)
    b = _f32(b)
    product = _f32(a * b)
    # The exact product of two binary32 values fits in binary64. This models
    # fma(a, b, -product) and is used only by CPU tests/reference code.
    error = _f32(float(a) * float(b) - float(product))
    return DoubleSingle(product, error)


def renormalize_cpu(hi: np.float32, lo: np.float32) -> DoubleSingle:
    return two_sum_cpu(_f32(hi), _f32(lo))


def add_cpu(a: DoubleSingle, b: DoubleSingle) -> DoubleSingle:
    high = two_sum_cpu(a.hi, b.hi)
    low = two_sum_cpu(a.lo, b.lo)
    carry = two_sum_cpu(high.lo, low.hi)
    combined = two_sum_cpu(high.hi, carry.hi)
    tail = _f32(_f32(low.lo + carry.lo) + combined.lo)
    return renormalize_cpu(combined.hi, tail)


def sub_cpu(a: DoubleSingle, b: DoubleSingle) -> DoubleSingle:
    return add_cpu(a, DoubleSingle(_f32(-b.hi), _f32(-b.lo)))


def mul_cpu(a: DoubleSingle, b: DoubleSingle, *, include_lo_lo: bool = True) -> DoubleSingle:
    product = two_product_cpu(a.hi, b.hi)
    correction = _f32(product.lo + _f32(a.hi * b.lo))
    correction = _f32(correction + _f32(a.lo * b.hi))
    if include_lo_lo:
        correction = _f32(correction + _f32(a.lo * b.lo))
    return renormalize_cpu(product.hi, correction)


def mul_float_cpu(a: DoubleSingle, b: np.float32) -> DoubleSingle:
    product = two_product_cpu(a.hi, _f32(b))
    correction = _f32(product.lo + _f32(a.lo * _f32(b)))
    return renormalize_cpu(product.hi, correction)


def square_cpu(a: DoubleSingle, *, include_lo_lo: bool = True) -> DoubleSingle:
    product = two_product_cpu(a.hi, a.hi)
    correction = _f32(product.lo + _f32(_f32(2.0) * _f32(a.hi * a.lo)))
    if include_lo_lo:
        correction = _f32(correction + _f32(a.lo * a.lo))
    return renormalize_cpu(product.hi, correction)


def diff_squares_cpu(
    a: DoubleSingle,
    b: DoubleSingle,
    *,
    include_lo_lo: bool = True,
) -> DoubleSingle:
    """Cancellation-safe difference of fully compensated DS squares."""

    return sub_cpu(
        square_cpu(a, include_lo_lo=include_lo_lo),
        square_cpu(b, include_lo_lo=include_lo_lo),
    )


def coordinate_components(
    center_x: str,
    center_y: str,
    view_width: str,
    width: int,
    height: int,
    *,
    precision_bits: int = 192,
) -> tuple[DoubleSingle, DoubleSingle, DoubleSingle, DoubleSingle]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    with mp.workprec(precision_bits):
        cx = mp.mpf(center_x)
        cy = mp.mpf(center_y)
        vw = mp.mpf(view_width)
        vh = vw * height / width
        dx = vw / width
        dy = vh / height
        x0 = cx - vw / 2 + dx / 2
        y0 = cy - vh / 2 + dy / 2
        return split_mpf(x0), split_mpf(y0), split_mpf(dx), split_mpf(dy)


def coordinate_at(origin: DoubleSingle, step: DoubleSingle, index: int) -> DoubleSingle:
    if index < 0:
        raise ValueError("pixel index must be non-negative")
    if index >= 1 << 24:
        raise ValueError("FP32 pixel index is only exact below 2**24")
    return add_cpu(origin, mul_float_cpu(step, np.float32(index)))


def grid_unique_fraction(origin: DoubleSingle, step: DoubleSingle, count: int) -> float:
    if count <= 1:
        return 1.0
    values = [coordinate_at(origin, step, index) for index in range(count)]
    distinct = sum(
        (left.hi != right.hi) or (left.lo != right.lo)
        for left, right in zip(values, values[1:])
    )
    return (distinct + 1) / count


def mandelbrot_reference(
    cr: str,
    ci: str,
    max_iterations: int,
    *,
    escape_radius: float = 2.0,
    precision_bits: int = 256,
    keep_orbit: bool = True,
) -> ReferencePointResult:
    with mp.workprec(precision_bits):
        real = mp.mpf(cr)
        imag = mp.mpf(ci)
        zr = mp.mpf("0")
        zi = mp.mpf("0")
        escape_squared = mp.mpf(escape_radius) ** 2
        orbit: list[tuple[str, str]] = []
        for iteration in range(max_iterations):
            zr, zi = zr * zr - zi * zi + real, 2 * zr * zi + imag
            if keep_orbit:
                orbit.append((mp.nstr(zr, 80), mp.nstr(zi, 80)))
            magnitude_squared = zr * zr + zi * zi
            if magnitude_squared > escape_squared:
                magnitude = mp.sqrt(magnitude_squared)
                smooth = iteration + 1 - mp.log(mp.log(magnitude)) / mp.log(2)
                return ReferencePointResult(True, iteration + 1, float(smooth), tuple(orbit))
        return ReferencePointResult(False, max_iterations, float(max_iterations), tuple(orbit))


def mandelbrot_double_single_cpu(
    cr: DoubleSingle,
    ci: DoubleSingle,
    max_iterations: int,
    *,
    escape_radius: float = 2.0,
    include_lo_lo: bool = True,
) -> DoubleSinglePointResult:
    zr = DoubleSingle(np.float32(0.0), np.float32(0.0))
    zi = DoubleSingle(np.float32(0.0), np.float32(0.0))
    orbit: list[tuple[DoubleSingle, DoubleSingle]] = []
    escape_squared = np.float32(escape_radius * escape_radius)
    for iteration in range(max_iterations):
        next_real = add_cpu(
            diff_squares_cpu(zr, zi, include_lo_lo=include_lo_lo),
            cr,
        )
        next_imag = add_cpu(
            mul_float_cpu(mul_cpu(zr, zi, include_lo_lo=include_lo_lo), np.float32(2.0)),
            ci,
        )
        zr, zi = next_real, next_imag
        orbit.append((zr, zi))
        magnitude = add_cpu(
            square_cpu(zr, include_lo_lo=include_lo_lo),
            square_cpu(zi, include_lo_lo=include_lo_lo),
        )
        if magnitude.as_float() > float(escape_squared):
            high_magnitude = np.float32(zr.hi * zr.hi + zi.hi * zi.hi)
            radius = math.sqrt(max(float(high_magnitude), 1e-30))
            smooth = iteration + 1 - math.log(max(math.log(radius), 1e-30)) / math.log(2.0)
            return DoubleSinglePointResult(True, iteration + 1, smooth, tuple(orbit))
    return DoubleSinglePointResult(False, max_iterations, float(max_iterations), tuple(orbit))


@cuda.jit(device=True, inline=True)
def ds_two_sum(a, b):
    s = float32(a + b)
    bb = float32(s - a)
    error = float32(float32(a - float32(s - bb)) + float32(b - bb))
    return s, error


@cuda.jit(device=True, inline=True)
def ds_quick_two_sum(a, b):
    # Precondition: abs(a) >= abs(b). Kept as a measured primitive; the hot
    # cancellation-prone paths below deliberately use ds_two_sum instead.
    s = float32(a + b)
    error = float32(b - float32(s - a))
    return s, error


@cuda.jit(device=True, inline=True)
def ds_two_product(a, b):
    product = float32(a * b)
    error = float32(cuda.fma(a, b, -product))
    return product, error


@cuda.jit(device=True, inline=True)
def ds_renorm(hi, lo):
    return ds_two_sum(hi, lo)


@cuda.jit(device=True, inline=True)
def ds_add(ahi, alo, bhi, blo):
    sh, sl = ds_two_sum(ahi, bhi)
    th, tl = ds_two_sum(alo, blo)
    uh, ul = ds_two_sum(sl, th)
    vh, vl = ds_two_sum(sh, uh)
    tail = float32(float32(tl + ul) + vl)
    return ds_renorm(vh, tail)


@cuda.jit(device=True, inline=True)
def ds_sub(ahi, alo, bhi, blo):
    return ds_add(ahi, alo, -bhi, -blo)


@cuda.jit(device=True, inline=True)
def ds_mul(ahi, alo, bhi, blo, include_lo_lo):
    ph, pl = ds_two_product(ahi, bhi)
    correction = float32(pl + float32(ahi * blo))
    correction = float32(correction + float32(alo * bhi))
    if include_lo_lo:
        correction = float32(correction + float32(alo * blo))
    return ds_renorm(ph, correction)


@cuda.jit(device=True, inline=True)
def ds_mul_float(ahi, alo, b):
    ph, pl = ds_two_product(ahi, b)
    correction = float32(pl + float32(alo * b))
    return ds_renorm(ph, correction)


@cuda.jit(device=True, inline=True)
def ds_square(ahi, alo, include_lo_lo):
    ph, pl = ds_two_product(ahi, ahi)
    correction = float32(pl + float32(float32(2.0) * float32(ahi * alo)))
    if include_lo_lo:
        correction = float32(correction + float32(alo * alo))
    return ds_renorm(ph, correction)


@cuda.jit(device=True, inline=True)
def ds_diff_squares(ahi, alo, bhi, blo, include_lo_lo):
    a2h, a2l = ds_square(ahi, alo, include_lo_lo)
    b2h, b2l = ds_square(bhi, blo, include_lo_lo)
    return ds_sub(a2h, a2l, b2h, b2l)


@cuda.jit(device=True, inline=True)
def ds_compare_scalar(ahi, alo, scalar):
    if ahi > scalar:
        return 1
    if ahi < scalar:
        return -1
    if alo > float32(0.0):
        return 1
    if alo < float32(0.0):
        return -1
    return 0


@cuda.jit(device=True, inline=True)
def ds_escape_decision(zrh, zrl, zih, zil, escape_squared, mode, include_lo_lo):
    high_magnitude = float32(float32(zrh * zrh) + float32(zih * zih))
    if mode == ESCAPE_HIGH_ONLY:
        return high_magnitude > escape_squared, high_magnitude

    r2h, r2l = ds_square(zrh, zrl, include_lo_lo)
    i2h, i2l = ds_square(zih, zil, include_lo_lo)
    mh, ml = ds_add(r2h, r2l, i2h, i2l)
    if mode == ESCAPE_FULL:
        return ds_compare_scalar(mh, ml, escape_squared) > 0, mh

    # Conservative band for deciding whether the represented DS value is safely
    # on one side of the threshold. Near the band, fall back to the full DS test.
    correction_bound = float32(
        float32(float32(2.0) * float32(abs(zrh) * abs(zrl)))
        + float32(float32(2.0) * float32(abs(zih) * abs(zil)))
    )
    correction_bound = float32(
        correction_bound + float32(float32(zrl * zrl) + float32(zil * zil))
    )
    rounding_bound = float32(
        float32(16.0 * F32_EPSILON) * float32(max(float32(1.0), abs(high_magnitude)))
    )
    band = float32(correction_bound + rounding_bound)
    if high_magnitude > float32(escape_squared + band):
        return True, high_magnitude
    if high_magnitude < float32(escape_squared - band):
        return False, high_magnitude
    return ds_compare_scalar(mh, ml, escape_squared) > 0, mh


@cuda.jit(device=True, inline=True)
def ds_coordinate(origin_hi, origin_lo, step_hi, step_lo, index):
    offset_hi, offset_lo = ds_mul_float(step_hi, step_lo, float32(index))
    return ds_add(origin_hi, origin_lo, offset_hi, offset_lo)


@cuda.jit(device=True, inline=True)
def ds_mandelbrot_point(
    crh,
    crl,
    cih,
    cil,
    max_iterations,
    escape_squared,
    include_lo_lo,
    escape_mode,
    generic_multiply,
):
    zrh = float32(0.0)
    zrl = float32(0.0)
    zih = float32(0.0)
    zil = float32(0.0)
    for iteration in range(max_iterations):
        if generic_multiply:
            rrh, rrl = ds_mul(zrh, zrl, zrh, zrl, include_lo_lo)
            iih, iil = ds_mul(zih, zil, zih, zil, include_lo_lo)
            nrh, nrl = ds_sub(rrh, rrl, iih, iil)
            rih, ril = ds_mul(zrh, zrl, zih, zil, include_lo_lo)
            nih, nil = ds_mul_float(rih, ril, float32(2.0))
        else:
            nrh, nrl = ds_diff_squares(zrh, zrl, zih, zil, include_lo_lo)
            rih, ril = ds_mul(zrh, zrl, zih, zil, include_lo_lo)
            nih, nil = ds_mul_float(rih, ril, float32(2.0))
        zrh, zrl = ds_add(nrh, nrl, crh, crl)
        zih, zil = ds_add(nih, nil, cih, cil)
        escaped, magnitude_high = ds_escape_decision(
            zrh, zrl, zih, zil, escape_squared, escape_mode, include_lo_lo
        )
        if escaped:
            magnitude = float32(math.sqrt(max(magnitude_high, float32(1e-30))))
            inner_log = float32(math.log(magnitude))
            if inner_log < float32(1e-30):
                inner_log = float32(1e-30)
            smooth = float32(iteration) + float32(1.0) - float32(
                math.log(inner_log) / math.log(float32(2.0))
            )
            return iteration + 1, smooth, zrh, zrl, zih, zil
    return max_iterations, float32(max_iterations), zrh, zrl, zih, zil


@cuda.jit
def mandelbrot_f32_kernel(
    escape_iterations,
    smooth_iterations,
    width,
    height,
    x0,
    y0,
    dx,
    dy,
    max_iterations,
    escape_squared,
):
    px, py = cuda.grid(2)
    if px >= width or py >= height:
        return
    cr = float32(x0 + float32(px) * dx)
    ci = float32(y0 + float32(py) * dy)
    zr = float32(0.0)
    zi = float32(0.0)
    for iteration in range(max_iterations):
        zr, zi = float32(zr * zr - zi * zi + cr), float32(float32(2.0) * zr * zi + ci)
        magnitude_squared = float32(zr * zr + zi * zi)
        if magnitude_squared > escape_squared:
            magnitude = float32(math.sqrt(magnitude_squared))
            inner_log = float32(max(math.log(magnitude), 1e-30))
            smooth_iterations[py, px] = float32(
                float32(iteration) + float32(1.0) - math.log(inner_log) / math.log(float32(2.0))
            )
            escape_iterations[py, px] = iteration + 1
            return
    escape_iterations[py, px] = max_iterations
    smooth_iterations[py, px] = float32(max_iterations)


@cuda.jit
def mandelbrot_f64_kernel(
    escape_iterations,
    smooth_iterations,
    width,
    height,
    x0,
    y0,
    dx,
    dy,
    max_iterations,
    escape_squared,
):
    px, py = cuda.grid(2)
    if px >= width or py >= height:
        return
    cr = float64(x0 + float64(px) * dx)
    ci = float64(y0 + float64(py) * dy)
    zr = float64(0.0)
    zi = float64(0.0)
    for iteration in range(max_iterations):
        zr, zi = zr * zr - zi * zi + cr, float64(2.0) * zr * zi + ci
        magnitude_squared = zr * zr + zi * zi
        if magnitude_squared > escape_squared:
            magnitude = math.sqrt(magnitude_squared)
            inner_log = max(math.log(magnitude), 1e-300)
            smooth_iterations[py, px] = float32(
                float64(iteration) + float64(1.0) - math.log(inner_log) / math.log(float64(2.0))
            )
            escape_iterations[py, px] = iteration + 1
            return
    escape_iterations[py, px] = max_iterations
    smooth_iterations[py, px] = float32(max_iterations)


def make_ds_kernel(*, include_lo_lo: bool, escape_mode: int, generic_multiply: bool):
    """Create a constant-specialized research kernel without global fast-math."""

    include_lo_lo_constant = bool(include_lo_lo)
    escape_mode_constant = int(escape_mode)
    generic_multiply_constant = bool(generic_multiply)

    @cuda.jit
    def kernel(
        escape_iterations,
        smooth_iterations,
        final_orbit,
        width,
        height,
        x0_hi,
        x0_lo,
        y0_hi,
        y0_lo,
        dx_hi,
        dx_lo,
        dy_hi,
        dy_lo,
        max_iterations,
        escape_squared,
    ):
        px, py = cuda.grid(2)
        if px >= width or py >= height:
            return
        crh, crl = ds_coordinate(x0_hi, x0_lo, dx_hi, dx_lo, px)
        cih, cil = ds_coordinate(y0_hi, y0_lo, dy_hi, dy_lo, py)
        iteration, smooth, zrh, zrl, zih, zil = ds_mandelbrot_point(
            crh,
            crl,
            cih,
            cil,
            max_iterations,
            escape_squared,
            include_lo_lo_constant,
            escape_mode_constant,
            generic_multiply_constant,
        )
        escape_iterations[py, px] = iteration
        smooth_iterations[py, px] = smooth
        final_orbit[py, px, 0] = zrh
        final_orbit[py, px, 1] = zrl
        final_orbit[py, px, 2] = zih
        final_orbit[py, px, 3] = zil

    return kernel


DS_GENERIC_FULL_KERNEL = make_ds_kernel(
    include_lo_lo=True, escape_mode=ESCAPE_FULL, generic_multiply=True
)
DS_SPECIALIZED_NO_LO2_HIGH_KERNEL = make_ds_kernel(
    include_lo_lo=False, escape_mode=ESCAPE_HIGH_ONLY, generic_multiply=False
)
DS_SPECIALIZED_LO2_HIGH_KERNEL = make_ds_kernel(
    include_lo_lo=True, escape_mode=ESCAPE_HIGH_ONLY, generic_multiply=False
)
DS_SPECIALIZED_LO2_FULL_KERNEL = make_ds_kernel(
    include_lo_lo=True, escape_mode=ESCAPE_FULL, generic_multiply=False
)
DS_SPECIALIZED_LO2_ADAPTIVE_KERNEL = make_ds_kernel(
    include_lo_lo=True, escape_mode=ESCAPE_ADAPTIVE, generic_multiply=False
)

KERNEL_VARIANTS = {
    "fp32": mandelbrot_f32_kernel,
    "fp64": mandelbrot_f64_kernel,
    "ds-generic-full": DS_GENERIC_FULL_KERNEL,
    "ds-specialized-no-lo2-high": DS_SPECIALIZED_NO_LO2_HIGH_KERNEL,
    "ds-specialized-lo2-high": DS_SPECIALIZED_LO2_HIGH_KERNEL,
    "ds-specialized-lo2-full": DS_SPECIALIZED_LO2_FULL_KERNEL,
    "ds-specialized-lo2-adaptive": DS_SPECIALIZED_LO2_ADAPTIVE_KERNEL,
}


def subnormal_relevance(values: Iterable[DoubleSingle]) -> dict[str, float | int | bool]:
    components = [abs(float(component)) for value in values for component in (value.hi, value.lo)]
    nonzero = [value for value in components if value > 0.0]
    minimum = min(nonzero, default=0.0)
    count = sum(0.0 < value < F32_TINY for value in components)
    return {
        "minimum_nonzero_component": minimum,
        "subnormal_component_count": count,
        "subnormal_components_present": count > 0,
        "fp32_normal_minimum": F32_TINY,
    }
