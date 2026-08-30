from __future__ import annotations

from dataclasses import dataclass
import math

import mpmath as mp
import numpy as np
from numba import cuda, float32

from ..deep_zoom import request_view_text
from ..models import RenderRequest


@dataclass(frozen=True, slots=True)
class DoubleSingle:
    hi: np.float32
    lo: np.float32


def _f32(value) -> np.float32:
    return np.float32(value)


def _two_sum_cpu(a: np.float32, b: np.float32) -> DoubleSingle:
    a = _f32(a)
    b = _f32(b)
    total = _f32(a + b)
    b_virtual = _f32(total - a)
    error = _f32(_f32(a - _f32(total - b_virtual)) + _f32(b - b_virtual))
    return DoubleSingle(total, error)


def split_mpf(value: mp.mpf) -> DoubleSingle:
    """Split an arbitrary-precision value without reducing it to binary64."""

    exact = mp.mpf(value)
    hi = np.float32(mp.nstr(exact, 80))
    if not np.isfinite(hi):
        raise OverflowError("value is outside the FP32 exponent range")
    remainder = exact - mp.mpf(float(hi))
    lo = np.float32(mp.nstr(remainder, 80))
    return _two_sum_cpu(hi, lo)


def launch_geometry(request: RenderRequest) -> tuple[np.float32, ...]:
    """Return exact CPU-to-DS origin and step components for one frame."""

    if request.width >= 1 << 24 or request.height >= 1 << 24:
        raise ValueError("double-single pixel indices must be smaller than 2**24")
    view = request_view_text(request)
    with mp.workprec(request.reference_bits):
        center_x = mp.mpf(view.center_x)
        center_y = mp.mpf(view.center_y)
        view_width = mp.mpf(view.width)
        view_height = view_width * request.height / request.width
        dx = view_width / request.width
        dy = view_height / request.height
        x0 = center_x - view_width / 2 + dx / 2
        y0 = center_y - view_height / 2 + dy / 2
        components = (
            split_mpf(x0),
            split_mpf(y0),
            split_mpf(dx),
            split_mpf(dy),
        )
    return tuple(component for value in components for component in (value.hi, value.lo))


@cuda.jit(device=True, inline=True)
def _ds_two_sum(a, b):
    total = float32(a + b)
    b_virtual = float32(total - a)
    error = float32(
        float32(a - float32(total - b_virtual)) + float32(b - b_virtual)
    )
    return total, error


@cuda.jit(device=True, inline=True)
def _ds_two_product(a, b):
    product = float32(a * b)
    error = float32(cuda.fma(a, b, -product))
    return product, error


@cuda.jit(device=True, inline=True)
def _ds_renormalize(hi, lo):
    return _ds_two_sum(hi, lo)


@cuda.jit(device=True, inline=True)
def _ds_add(a_hi, a_lo, b_hi, b_lo):
    sum_hi, sum_lo = _ds_two_sum(a_hi, b_hi)
    tail_hi, tail_lo = _ds_two_sum(a_lo, b_lo)
    carry_hi, carry_lo = _ds_two_sum(sum_lo, tail_hi)
    merged_hi, merged_lo = _ds_two_sum(sum_hi, carry_hi)
    tail = float32(float32(tail_lo + carry_lo) + merged_lo)
    return _ds_renormalize(merged_hi, tail)


@cuda.jit(device=True, inline=True)
def _ds_sub(a_hi, a_lo, b_hi, b_lo):
    return _ds_add(a_hi, a_lo, -b_hi, -b_lo)


@cuda.jit(device=True, inline=True)
def _ds_mul(a_hi, a_lo, b_hi, b_lo):
    product_hi, product_lo = _ds_two_product(a_hi, b_hi)
    correction = float32(product_lo + float32(a_hi * b_lo))
    correction = float32(correction + float32(a_lo * b_hi))
    correction = float32(correction + float32(a_lo * b_lo))
    return _ds_renormalize(product_hi, correction)


@cuda.jit(device=True, inline=True)
def _ds_mul_float(a_hi, a_lo, b):
    product_hi, product_lo = _ds_two_product(a_hi, b)
    correction = float32(product_lo + float32(a_lo * b))
    return _ds_renormalize(product_hi, correction)


@cuda.jit(device=True, inline=True)
def _ds_square(a_hi, a_lo):
    product_hi, product_lo = _ds_two_product(a_hi, a_hi)
    correction = float32(
        product_lo + float32(float32(2.0) * float32(a_hi * a_lo))
    )
    correction = float32(correction + float32(a_lo * a_lo))
    return _ds_renormalize(product_hi, correction)


@cuda.jit(device=True, inline=True)
def _ds_diff_squares(a_hi, a_lo, b_hi, b_lo):
    a2_hi, a2_lo = _ds_square(a_hi, a_lo)
    b2_hi, b2_lo = _ds_square(b_hi, b_lo)
    return _ds_sub(a2_hi, a2_lo, b2_hi, b2_lo)


@cuda.jit(device=True, inline=True)
def _ds_coordinate(origin_hi, origin_lo, step_hi, step_lo, index):
    offset_hi, offset_lo = _ds_mul_float(step_hi, step_lo, float32(index))
    return _ds_add(origin_hi, origin_lo, offset_hi, offset_lo)


@cuda.jit(device=True, inline=True)
def _mandelbrot_value_ds(
    c_real_hi,
    c_real_lo,
    c_imag_hi,
    c_imag_lo,
    max_iterations,
    escape_squared,
):
    z_real_hi = float32(0.0)
    z_real_lo = float32(0.0)
    z_imag_hi = float32(0.0)
    z_imag_lo = float32(0.0)
    zero = float32(0.0)
    one = float32(1.0)
    two = float32(2.0)
    tiny = float32(1e-30)

    for iteration in range(max_iterations):
        next_real_hi, next_real_lo = _ds_diff_squares(
            z_real_hi, z_real_lo, z_imag_hi, z_imag_lo
        )
        product_hi, product_lo = _ds_mul(
            z_real_hi, z_real_lo, z_imag_hi, z_imag_lo
        )
        next_imag_hi, next_imag_lo = _ds_mul_float(
            product_hi, product_lo, two
        )
        z_real_hi, z_real_lo = _ds_add(
            next_real_hi, next_real_lo, c_real_hi, c_real_lo
        )
        z_imag_hi, z_imag_lo = _ds_add(
            next_imag_hi, next_imag_lo, c_imag_hi, c_imag_lo
        )

        magnitude_squared = float32(
            float32(z_real_hi * z_real_hi) + float32(z_imag_hi * z_imag_hi)
        )
        if magnitude_squared > escape_squared:
            magnitude = float32(math.sqrt(max(magnitude_squared, tiny)))
            inner_log = float32(math.log(magnitude))
            if inner_log < tiny:
                inner_log = tiny
            smooth = float32(iteration) + one - float32(
                math.log(inner_log) / math.log(two)
            )
            value = float32(smooth / float32(max_iterations))
            if value < zero:
                value = zero
            return value, False
    return zero, True


@cuda.jit
def cuda_mandelbrot_double_single_kernel(
    values,
    inside,
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
    c_real_hi, c_real_lo = _ds_coordinate(x0_hi, x0_lo, dx_hi, dx_lo, px)
    c_imag_hi, c_imag_lo = _ds_coordinate(y0_hi, y0_lo, dy_hi, dy_lo, py)
    value, is_inside = _mandelbrot_value_ds(
        c_real_hi,
        c_real_lo,
        c_imag_hi,
        c_imag_lo,
        max_iterations,
        escape_squared,
    )
    values[py, px] = value
    inside[py, px] = is_inside
