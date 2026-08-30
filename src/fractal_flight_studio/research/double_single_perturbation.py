from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import cuda, float32

from .double_single import (
    DoubleSingle,
    ds_add,
    ds_compare_scalar,
    ds_coordinate,
    ds_diff_squares,
    ds_mul,
    ds_mul_float,
    ds_square,
    split_mpf,
)


@dataclass(frozen=True, slots=True)
class SplitOrbit:
    real_hi: np.ndarray
    real_lo: np.ndarray
    imag_hi: np.ndarray
    imag_lo: np.ndarray

    @property
    def transfer_bytes(self) -> int:
        return sum(array.nbytes for array in (self.real_hi, self.real_lo, self.imag_hi, self.imag_lo))


def split_float64_array(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a finite binary64 array into non-overlapping binary32 components."""

    source = np.ascontiguousarray(values, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("double-single transfer values must be finite")
    high = np.ascontiguousarray(source, dtype=np.float32)
    if not np.all(np.isfinite(high)):
        raise OverflowError("double-single high components overflow binary32")
    low = np.ascontiguousarray(source - high.astype(np.float64), dtype=np.float32)
    return high, low


def split_reference_orbit(orbit_real: np.ndarray, orbit_imag: np.ndarray) -> SplitOrbit:
    if orbit_real.shape != orbit_imag.shape:
        raise ValueError("real and imaginary reference orbit arrays must have the same shape")
    real_hi, real_lo = split_float64_array(orbit_real)
    imag_hi, imag_lo = split_float64_array(orbit_imag)
    return SplitOrbit(real_hi, real_lo, imag_hi, imag_lo)


def split_scalar(value: float) -> DoubleSingle:
    return split_mpf(repr(float(value)))


@cuda.jit(device=True, inline=True)
def _ds_compare_pair(ahi, alo, bhi, blo):
    if ahi > bhi:
        return 1
    if ahi < bhi:
        return -1
    if alo > blo:
        return 1
    if alo < blo:
        return -1
    return 0


@cuda.jit(device=True, inline=True)
def _ds_complex_add(arh, arl, aih, ail, brh, brl, bih, bil):
    rrh, rrl = ds_add(arh, arl, brh, brl)
    rih, ril = ds_add(aih, ail, bih, bil)
    return rrh, rrl, rih, ril


@cuda.jit(device=True, inline=True)
def _ds_complex_square(zrh, zrl, zih, zil):
    rrh, rrl = ds_diff_squares(zrh, zrl, zih, zil, True)
    rih, ril = ds_mul(zrh, zrl, zih, zil, True)
    rih, ril = ds_mul_float(rih, ril, float32(2.0))
    return rrh, rrl, rih, ril


@cuda.jit(device=True, inline=True)
def _ds_complex_product(arh, arl, aih, ail, brh, brl, bih, bil):
    rrh, rrl = ds_mul(arh, arl, brh, brl, True)
    tih, til = ds_mul(aih, ail, bih, bil, True)
    rrh, rrl = ds_add(rrh, rrl, -tih, -til)

    rih, ril = ds_mul(arh, arl, bih, bil, True)
    tih, til = ds_mul(aih, ail, brh, brl, True)
    rih, ril = ds_add(rih, ril, tih, til)
    return rrh, rrl, rih, ril


@cuda.jit(device=True, inline=True)
def _ds_magnitude_squared(zrh, zrl, zih, zil):
    r2h, r2l = ds_square(zrh, zrl, True)
    i2h, i2l = ds_square(zih, zil, True)
    return ds_add(r2h, r2l, i2h, i2l)


@cuda.jit
def double_single_perturbation_kernel(
    values,
    inside,
    glitch_repaired,
    rebased,
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
    max_iter,
    escape_squared,
    orbit_real_hi,
    orbit_real_lo,
    orbit_imag_hi,
    orbit_imag_lo,
    reference_rebase_limit,
    glitch_tolerance,
    min_ref_sq,
):
    """Research perturbation kernel using only FP32 double-single arithmetic.

    The CPU reference orbit is transferred as two FP32 components per real and
    imaginary value. Escape, rebase and glitch comparisons use represented
    double-single magnitudes; smooth coloring uses the high magnitude component.
    """

    px, py = cuda.grid(2)
    if px >= width or py >= height:
        return

    dcxh, dcxl = ds_coordinate(x0_hi, x0_lo, dx_hi, dx_lo, px)
    dcyh, dcyl = ds_coordinate(y0_hi, y0_lo, dy_hi, dy_lo, py)
    dzrh = float32(0.0)
    dzrl = float32(0.0)
    dzih = float32(0.0)
    dzil = float32(0.0)
    reference_index = 0
    zero = float32(0.0)
    tiny = float32(1e-30)
    glitch_repaired[py, px] = False
    rebased[py, px] = False

    for iteration in range(max_iter):
        ref_rh = orbit_real_hi[reference_index]
        ref_rl = orbit_real_lo[reference_index]
        ref_ih = orbit_imag_hi[reference_index]
        ref_il = orbit_imag_lo[reference_index]

        if reference_index >= reference_rebase_limit:
            dzrh, dzrl, dzih, dzil = _ds_complex_add(
                ref_rh, ref_rl, ref_ih, ref_il, dzrh, dzrl, dzih, dzil
            )
            reference_index = 0
            ref_rh = zero
            ref_rl = zero
            ref_ih = zero
            ref_il = zero
            rebased[py, px] = True

        actual_rh, actual_rl, actual_ih, actual_il = _ds_complex_add(
            ref_rh, ref_rl, ref_ih, ref_il, dzrh, dzrl, dzih, dzil
        )
        delta_sq_h, delta_sq_l = _ds_magnitude_squared(dzrh, dzrl, dzih, dzil)
        actual_sq_h, actual_sq_l = _ds_magnitude_squared(
            actual_rh, actual_rl, actual_ih, actual_il
        )
        if reference_index > 0 and _ds_compare_pair(
            actual_sq_h, actual_sq_l, delta_sq_h, delta_sq_l
        ) < 0:
            dzrh = actual_rh
            dzrl = actual_rl
            dzih = actual_ih
            dzil = actual_il
            reference_index = 0
            ref_rh = zero
            ref_rl = zero
            ref_ih = zero
            ref_il = zero
            rebased[py, px] = True

        cross_rh, cross_rl, cross_ih, cross_il = _ds_complex_product(
            ref_rh, ref_rl, ref_ih, ref_il, dzrh, dzrl, dzih, dzil
        )
        cross_rh, cross_rl = ds_mul_float(cross_rh, cross_rl, float32(2.0))
        cross_ih, cross_il = ds_mul_float(cross_ih, cross_il, float32(2.0))
        square_rh, square_rl, square_ih, square_il = _ds_complex_square(
            dzrh, dzrl, dzih, dzil
        )
        next_dzrh, next_dzrl, next_dzih, next_dzil = _ds_complex_add(
            cross_rh,
            cross_rl,
            cross_ih,
            cross_il,
            square_rh,
            square_rl,
            square_ih,
            square_il,
        )
        next_dzrh, next_dzrl = ds_add(next_dzrh, next_dzrl, dcxh, dcxl)
        next_dzih, next_dzil = ds_add(next_dzih, next_dzil, dcyh, dcyl)

        next_reference_index = reference_index + 1
        zrh, zrl = ds_add(
            orbit_real_hi[next_reference_index],
            orbit_real_lo[next_reference_index],
            next_dzrh,
            next_dzrl,
        )
        zih, zil = ds_add(
            orbit_imag_hi[next_reference_index],
            orbit_imag_lo[next_reference_index],
            next_dzih,
            next_dzil,
        )
        magnitude_h, magnitude_l = _ds_magnitude_squared(zrh, zrl, zih, zil)

        if ds_compare_scalar(magnitude_h, magnitude_l, escape_squared) > 0:
            high_magnitude = magnitude_h
            if high_magnitude < tiny:
                high_magnitude = tiny
            magnitude = float32(math.sqrt(high_magnitude))
            inner_log = float32(math.log(magnitude))
            if inner_log < tiny:
                inner_log = tiny
            smooth = float32(iteration) + float32(1.0) - float32(
                math.log(inner_log) / math.log(float32(2.0))
            )
            value = float32(smooth / float32(max_iter))
            if value < zero:
                value = zero
            values[py, px] = value
            inside[py, px] = False
            return

        ref_next_h, ref_next_l = _ds_magnitude_squared(
            orbit_real_hi[next_reference_index],
            orbit_real_lo[next_reference_index],
            orbit_imag_hi[next_reference_index],
            orbit_imag_lo[next_reference_index],
        )
        if ds_compare_scalar(ref_next_h, ref_next_l, min_ref_sq) < 0:
            ref_next_h = min_ref_sq
            ref_next_l = zero
        threshold_h, threshold_l = ds_mul_float(
            ref_next_h, ref_next_l, glitch_tolerance
        )

        if _ds_compare_pair(magnitude_h, magnitude_l, threshold_h, threshold_l) < 0:
            dzrh = zrh
            dzrl = zrl
            dzih = zih
            dzil = zil
            reference_index = 0
            glitch_repaired[py, px] = True
            rebased[py, px] = True
        else:
            dzrh = next_dzrh
            dzrl = next_dzrl
            dzih = next_dzih
            dzil = next_dzil
            reference_index = next_reference_index

    values[py, px] = zero
    inside[py, px] = True
