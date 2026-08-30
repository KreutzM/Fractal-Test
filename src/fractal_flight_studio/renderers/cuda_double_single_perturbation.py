from __future__ import annotations

from dataclasses import dataclass
import math

import mpmath as mp
import numpy as np
from numba import cuda, float32

from .cuda_double_single import (
    DoubleSingle,
    _ds_add,
    _ds_coordinate,
    _ds_diff_squares,
    _ds_mul,
    _ds_mul_float,
    _ds_square,
    split_mpf,
)

_FLOAT32_TINY = np.float32(np.finfo(np.float32).tiny)
_FLOAT32_REFERENCE_COMPONENT_FLOOR = math.sqrt(float(_FLOAT32_TINY))


@dataclass(frozen=True, slots=True)
class SplitOrbit:
    real_hi: np.ndarray
    real_lo: np.ndarray
    imag_hi: np.ndarray
    imag_lo: np.ndarray

    @property
    def transfer_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (self.real_hi, self.real_lo, self.imag_hi, self.imag_lo)
        )


@dataclass(frozen=True, slots=True)
class PerturbationLaunchData:
    orbit: SplitOrbit
    x0: DoubleSingle
    y0: DoubleSingle
    dx: DoubleSingle
    dy: DoubleSingle


class DoubleSinglePerturbationUnavailable(ValueError):
    """Raised when unscaled FP32 double-single cannot preserve a perturbation input."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _split_float64_array(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.ascontiguousarray(values, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise DoubleSinglePerturbationUnavailable("reference-nonfinite")
    high = np.ascontiguousarray(source, dtype=np.float32)
    if not np.all(np.isfinite(high)):
        raise DoubleSinglePerturbationUnavailable("reference-overflow")
    low = np.ascontiguousarray(source - high.astype(np.float64), dtype=np.float32)
    if not np.all(np.isfinite(low)):
        raise DoubleSinglePerturbationUnavailable("reference-residual-overflow")
    reconstructed = high.astype(np.float64) + low.astype(np.float64)
    if np.any((source != 0.0) & (reconstructed == 0.0)):
        raise DoubleSinglePerturbationUnavailable("reference-underflow")
    return high, low


def split_reference_orbit(orbit_real: np.ndarray, orbit_imag: np.ndarray) -> SplitOrbit:
    if orbit_real.shape != orbit_imag.shape:
        raise DoubleSinglePerturbationUnavailable("reference-shape-mismatch")
    real = np.ascontiguousarray(orbit_real, dtype=np.float64)
    imag = np.ascontiguousarray(orbit_imag, dtype=np.float64)
    nonzero = (real != 0.0) | (imag != 0.0)
    component_scale = np.maximum(np.abs(real), np.abs(imag))
    if np.any(nonzero & (component_scale < _FLOAT32_REFERENCE_COMPONENT_FLOOR)):
        # The production DS kernel clamps reference magnitudes at float32 tiny,
        # while native FP64 can represent a much lower glitch floor. Retain FP64
        # whenever that distinction can affect a nonzero reference sample.
        raise DoubleSinglePerturbationUnavailable("reference-magnitude-range")
    real_hi, real_lo = _split_float64_array(real)
    imag_hi, imag_lo = _split_float64_array(imag)
    return SplitOrbit(real_hi, real_lo, imag_hi, imag_lo)


def split_scalar(value: float, name: str) -> DoubleSingle:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise DoubleSinglePerturbationUnavailable(f"{name}-nonfinite")
    try:
        split = split_mpf(mp.mpf(repr(scalar)))
    except OverflowError as exc:
        raise DoubleSinglePerturbationUnavailable(f"{name}-overflow") from exc
    reconstructed = float(split.hi) + float(split.lo)
    if scalar != 0.0 and reconstructed == 0.0:
        raise DoubleSinglePerturbationUnavailable(f"{name}-underflow")
    return split


def prepare_perturbation_launch(perturb) -> PerturbationLaunchData:
    return PerturbationLaunchData(
        orbit=split_reference_orbit(perturb.orbit_real, perturb.orbit_imag),
        x0=split_scalar(perturb.x0_rel, "x0"),
        y0=split_scalar(perturb.y0_rel, "y0"),
        dx=split_scalar(perturb.dx, "dx"),
        dy=split_scalar(perturb.dy, "dy"),
    )


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
def _ds_compare_scalar(ahi, alo, scalar):
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
def _ds_complex_add(arh, arl, aih, ail, brh, brl, bih, bil):
    rrh, rrl = _ds_add(arh, arl, brh, brl)
    rih, ril = _ds_add(aih, ail, bih, bil)
    return rrh, rrl, rih, ril


@cuda.jit(device=True, inline=True)
def _ds_complex_square(zrh, zrl, zih, zil):
    rrh, rrl = _ds_diff_squares(zrh, zrl, zih, zil)
    rih, ril = _ds_mul(zrh, zrl, zih, zil)
    rih, ril = _ds_mul_float(rih, ril, float32(2.0))
    return rrh, rrl, rih, ril


@cuda.jit(device=True, inline=True)
def _ds_complex_product(arh, arl, aih, ail, brh, brl, bih, bil):
    rrh, rrl = _ds_mul(arh, arl, brh, brl)
    tih, til = _ds_mul(aih, ail, bih, bil)
    rrh, rrl = _ds_add(rrh, rrl, -tih, -til)

    rih, ril = _ds_mul(arh, arl, bih, bil)
    tih, til = _ds_mul(aih, ail, brh, brl)
    rih, ril = _ds_add(rih, ril, tih, til)
    return rrh, rrl, rih, ril


@cuda.jit(device=True, inline=True)
def _ds_magnitude_squared(zrh, zrl, zih, zil):
    r2h, r2l = _ds_square(zrh, zrl)
    i2h, i2l = _ds_square(zih, zil)
    return _ds_add(r2h, r2l, i2h, i2l)


@cuda.jit
def cuda_mandelbrot_double_single_perturbation_kernel(
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
    px, py = cuda.grid(2)
    if px >= width or py >= height:
        return

    dcxh, dcxl = _ds_coordinate(x0_hi, x0_lo, dx_hi, dx_lo, px)
    dcyh, dcyl = _ds_coordinate(y0_hi, y0_lo, dy_hi, dy_lo, py)
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
        cross_rh, cross_rl = _ds_mul_float(cross_rh, cross_rl, float32(2.0))
        cross_ih, cross_il = _ds_mul_float(cross_ih, cross_il, float32(2.0))
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
        next_dzrh, next_dzrl = _ds_add(next_dzrh, next_dzrl, dcxh, dcxl)
        next_dzih, next_dzil = _ds_add(next_dzih, next_dzil, dcyh, dcyl)

        next_reference_index = reference_index + 1
        zrh, zrl = _ds_add(
            orbit_real_hi[next_reference_index],
            orbit_real_lo[next_reference_index],
            next_dzrh,
            next_dzrl,
        )
        zih, zil = _ds_add(
            orbit_imag_hi[next_reference_index],
            orbit_imag_lo[next_reference_index],
            next_dzih,
            next_dzil,
        )
        magnitude_h, magnitude_l = _ds_magnitude_squared(zrh, zrl, zih, zil)

        if _ds_compare_scalar(magnitude_h, magnitude_l, escape_squared) > 0:
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
        if _ds_compare_scalar(ref_next_h, ref_next_l, min_ref_sq) < 0:
            ref_next_h = min_ref_sq
            ref_next_l = zero
        threshold_h, threshold_l = _ds_mul_float(
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
