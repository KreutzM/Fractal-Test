from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from ..deep_zoom import (
    GLITCH_TOLERANCE,
    MIN_REFERENCE_MAGNITUDE_SQUARED,
    PerturbationReferenceCache,
    should_use_perturbation,
)
from ..models import FractalKind, Precision, RenderRequest
from .base import Renderer, RenderResult

_KIND_CODES = {
    FractalKind.MANDELBROT: 0,
    FractalKind.JULIA: 1,
    FractalKind.BURNING_SHIP: 2,
    FractalKind.MULTIBROT: 3,
    FractalKind.NEWTON: 4,
}


@njit(cache=True, inline="always")
def _complex_pow_f32(zr, zi, power):
    rr = zr
    ii = zi
    for _ in range(1, power):
        old_rr = rr
        rr = old_rr * zr - ii * zi
        ii = old_rr * zi + ii * zr
    return rr, ii


@njit(cache=True, inline="always")
def _complex_pow_f64(zr, zi, power):
    rr = zr
    ii = zi
    for _ in range(1, power):
        old_rr = rr
        rr = old_rr * zr - ii * zi
        ii = old_rr * zi + ii * zr
    return rr, ii


@njit(cache=True, inline="always")
def _escape_value_f32(
    x, y, kind, max_iter, color_iter, escape_squared, julia_r, julia_i, exponent
):
    zero = np.float32(0.0)
    one = np.float32(1.0)
    two = np.float32(2.0)
    tiny = np.float32(1e-30)
    if kind == 1:
        zr, zi = x, y
        cr, ci = julia_r, julia_i
    else:
        zr, zi = zero, zero
        cr, ci = x, y

    for iteration in range(max_iter):
        if kind == 2:
            zr = abs(zr)
            zi = abs(zi)
        if kind == 3:
            next_r, next_i = _complex_pow_f32(zr, zi, exponent)
        else:
            next_r = zr * zr - zi * zi
            next_i = two * zr * zi
        zr = next_r + cr
        zi = next_i + ci
        magnitude_squared = zr * zr + zi * zi
        if magnitude_squared > escape_squared:
            magnitude = np.float32(math.sqrt(magnitude_squared))
            base = np.float32(exponent if kind == 3 else 2)
            inner_log = np.float32(math.log(magnitude))
            if inner_log < tiny:
                inner_log = tiny
            smooth = (
                np.float32(iteration)
                + one
                - np.float32(math.log(inner_log)) / np.float32(math.log(base))
            )
            value = smooth / np.float32(color_iter)
            if value < zero:
                value = zero
            return value, False
    return zero, True


@njit(cache=True, inline="always")
def _escape_value_f64(
    x, y, kind, max_iter, color_iter, escape_squared, julia_r, julia_i, exponent
):
    if kind == 1:
        zr, zi = x, y
        cr, ci = julia_r, julia_i
    else:
        zr, zi = 0.0, 0.0
        cr, ci = x, y

    for iteration in range(max_iter):
        if kind == 2:
            zr = abs(zr)
            zi = abs(zi)
        if kind == 3:
            next_r, next_i = _complex_pow_f64(zr, zi, exponent)
        else:
            next_r = zr * zr - zi * zi
            next_i = 2.0 * zr * zi
        zr = next_r + cr
        zi = next_i + ci
        magnitude_squared = zr * zr + zi * zi
        if magnitude_squared > escape_squared:
            magnitude = math.sqrt(magnitude_squared)
            base = float(exponent if kind == 3 else 2)
            smooth = iteration + 1.0 - math.log(max(math.log(magnitude), 1e-30)) / math.log(base)
            return max(0.0, smooth / color_iter), False
    return 0.0, True


@njit(cache=True, inline="always")
def _newton_value_f32(x, y, max_iter, color_iter):
    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    two = np.float32(2.0)
    three = np.float32(3.0)
    root_imag = np.float32(0.8660254037844386)
    tolerance_squared = np.float32(1e-12)
    denominator_floor = np.float32(1e-30)
    zr, zi = x, y

    for iteration in range(max_iter):
        z2r = zr * zr - zi * zi
        z2i = two * zr * zi
        z3r = z2r * zr - z2i * zi
        z3i = z2r * zi + z2i * zr
        fr = z3r - one
        fi = z3i
        dr = three * z2r
        di = three * z2i
        denominator = dr * dr + di * di
        if denominator < denominator_floor:
            return zero, True
        quotient_r = (fr * dr + fi * di) / denominator
        quotient_i = (fi * dr - fr * di) / denominator
        zr -= quotient_r
        zi -= quotient_i

        d0 = (zr - one) * (zr - one) + zi * zi
        d1 = (zr + half) * (zr + half) + (zi - root_imag) * (zi - root_imag)
        d2 = (zr + half) * (zr + half) + (zi + root_imag) * (zi + root_imag)
        local = one - np.float32(iteration) / np.float32(color_iter)
        shade = np.float32(0.12) + np.float32(0.76) * local
        if d0 < tolerance_squared:
            return shade / three, False
        if d1 < tolerance_squared:
            return (one + shade) / three, False
        if d2 < tolerance_squared:
            return (two + shade) / three, False
    return zero, True


@njit(cache=True, inline="always")
def _newton_value_f64(x, y, max_iter, color_iter):
    zr, zi = x, y
    roots_r = (1.0, -0.5, -0.5)
    roots_i = (0.0, 0.8660254037844386, -0.8660254037844386)
    tolerance_squared = 1e-12

    for iteration in range(max_iter):
        z2r = zr * zr - zi * zi
        z2i = 2.0 * zr * zi
        z3r = z2r * zr - z2i * zi
        z3i = z2r * zi + z2i * zr
        fr = z3r - 1.0
        fi = z3i
        dr = 3.0 * z2r
        di = 3.0 * z2i
        denominator = dr * dr + di * di
        if denominator < 1e-30:
            return 0.0, True
        quotient_r = (fr * dr + fi * di) / denominator
        quotient_i = (fi * dr - fr * di) / denominator
        zr -= quotient_r
        zi -= quotient_i

        for root in range(3):
            rr = zr - roots_r[root]
            ri = zi - roots_i[root]
            if rr * rr + ri * ri < tolerance_squared:
                local = 1.0 - iteration / color_iter
                return (root + 0.12 + 0.76 * local) / 3.0, False
    return 0.0, True


@njit(parallel=True, cache=True)
def _render_kernel_f32(
    width,
    height,
    x0,
    y0,
    dx,
    dy,
    kind,
    max_iter,
    color_iter,
    escape_squared,
    julia_r,
    julia_i,
    exponent,
):
    values = np.empty((height, width), dtype=np.float32)
    inside = np.empty((height, width), dtype=np.bool_)
    for py in prange(height):
        y = y0 + np.float32(py) * dy
        for px in range(width):
            x = x0 + np.float32(px) * dx
            if kind == 4:
                value, is_inside = _newton_value_f32(x, y, max_iter, color_iter)
            else:
                value, is_inside = _escape_value_f32(
                    x,
                    y,
                    kind,
                    max_iter,
                    color_iter,
                    escape_squared,
                    julia_r,
                    julia_i,
                    exponent,
                )
            values[py, px] = value
            inside[py, px] = is_inside
    return values, inside


@njit(parallel=True, cache=True)
def _render_kernel_f64(
    width,
    height,
    x0,
    y0,
    dx,
    dy,
    kind,
    max_iter,
    color_iter,
    escape_squared,
    julia_r,
    julia_i,
    exponent,
):
    values = np.empty((height, width), dtype=np.float32)
    inside = np.empty((height, width), dtype=np.bool_)
    for py in prange(height):
        y = y0 + py * dy
        for px in range(width):
            x = x0 + px * dx
            if kind == 4:
                value, is_inside = _newton_value_f64(x, y, max_iter, color_iter)
            else:
                value, is_inside = _escape_value_f64(
                    x,
                    y,
                    kind,
                    max_iter,
                    color_iter,
                    escape_squared,
                    julia_r,
                    julia_i,
                    exponent,
                )
            values[py, px] = value
            inside[py, px] = is_inside
    return values, inside




@njit(cache=True, parallel=True)
def _render_perturb_kernel_f64(
    width,
    height,
    x0_rel,
    y0_rel,
    dx,
    dy,
    max_iter,
    color_iter,
    escape_squared,
    orbit_real,
    orbit_imag,
    reference_rebase_limit,
    glitch_tolerance,
    min_ref_sq,
):
    values = np.zeros((height, width), dtype=np.float32)
    inside = np.ones((height, width), dtype=np.bool_)
    glitch_repaired = np.zeros((height, width), dtype=np.bool_)
    rebased = np.zeros((height, width), dtype=np.bool_)
    tiny = 1e-30

    for py in prange(height):
        dcy = y0_rel + py * dy
        for px in range(width):
            dcx = x0_rel + px * dx
            dzr = 0.0
            dzi = 0.0
            reference_index = 0

            for iteration in range(max_iter):
                ref_r = orbit_real[reference_index]
                ref_i = orbit_imag[reference_index]

                if reference_index >= reference_rebase_limit:
                    dzr = ref_r + dzr
                    dzi = ref_i + dzi
                    reference_index = 0
                    ref_r = 0.0
                    ref_i = 0.0
                    rebased[py, px] = True

                # True rebasing: keep the same high-precision parameter delta-c
                # and only change the decomposition z = Z_ref + delta-z.
                actual_r = ref_r + dzr
                actual_i = ref_i + dzi
                delta_sq = dzr * dzr + dzi * dzi
                actual_sq = actual_r * actual_r + actual_i * actual_i
                if reference_index > 0 and actual_sq < delta_sq:
                    dzr = actual_r
                    dzi = actual_i
                    reference_index = 0
                    ref_r = 0.0
                    ref_i = 0.0
                    rebased[py, px] = True

                next_dzr = (
                    2.0 * (ref_r * dzr - ref_i * dzi)
                    + (dzr * dzr - dzi * dzi)
                    + dcx
                )
                next_dzi = (
                    2.0 * (ref_r * dzi + ref_i * dzr)
                    + 2.0 * dzr * dzi
                    + dcy
                )
                next_reference_index = reference_index + 1
                zr = orbit_real[next_reference_index] + next_dzr
                zi = orbit_imag[next_reference_index] + next_dzi
                magnitude_squared = zr * zr + zi * zi

                if magnitude_squared > escape_squared:
                    magnitude = math.sqrt(magnitude_squared)
                    inner_log = math.log(magnitude)
                    if inner_log < tiny:
                        inner_log = tiny
                    smooth = iteration + 1.0 - math.log(inner_log) / math.log(2.0)
                    value = smooth / color_iter
                    if value < 0.0:
                        value = 0.0
                    values[py, px] = value
                    inside[py, px] = False
                    break

                ref_next_sq = (
                    orbit_real[next_reference_index] * orbit_real[next_reference_index]
                    + orbit_imag[next_reference_index] * orbit_imag[next_reference_index]
                )
                if ref_next_sq < min_ref_sq:
                    ref_next_sq = min_ref_sq

                # Catastrophic cancellation is repaired by representing the
                # already-computed absolute orbit value relative to Z_0 = 0.
                # No absolute deep coordinate c is ever reconstructed in FP64.
                if magnitude_squared < glitch_tolerance * ref_next_sq:
                    dzr = zr
                    dzi = zi
                    reference_index = 0
                    glitch_repaired[py, px] = True
                    rebased[py, px] = True
                else:
                    dzr = next_dzr
                    dzi = next_dzi
                    reference_index = next_reference_index

    return values, inside, glitch_repaired, rebased


class CpuRenderer(Renderer):
    name = "cpu-numba"

    def __init__(self) -> None:
        self._reference_cache = PerturbationReferenceCache()

    def is_available(self) -> bool:
        return True

    def warm_up(self) -> None:
        self.render(RenderRequest(width=8, height=8, max_iterations=4, precision=Precision.FLOAT32))
        self.render(RenderRequest(width=8, height=8, max_iterations=4, precision=Precision.FLOAT64))

    @staticmethod
    def _geometry(request: RenderRequest, dtype):
        view_height = request.viewport.width * request.height / request.width
        dx = dtype(request.viewport.width / request.width)
        dy = dtype(view_height / request.height)
        x0 = dtype(request.viewport.center_x - request.viewport.width * 0.5 + float(dx) * 0.5)
        y0 = dtype(request.viewport.center_y - view_height * 0.5 + float(dy) * 0.5)
        return x0, y0, dx, dy

    def render(self, request: RenderRequest) -> RenderResult:
        request.validate()
        if should_use_perturbation(request):
            perturb = self._reference_cache.prepare(request)
            start = time.perf_counter()
            values, inside, glitch, rebased = _render_perturb_kernel_f64(
                request.width,
                request.height,
                perturb.x0_rel,
                perturb.y0_rel,
                perturb.dx,
                perturb.dy,
                request.max_iterations,
                request.effective_color_iterations,
                request.escape_radius * request.escape_radius,
                perturb.orbit_real,
                perturb.orbit_imag,
                perturb.reference_rebase_limit,
                GLITCH_TOLERANCE,
                MIN_REFERENCE_MAGNITUDE_SQUARED,
            )
            elapsed = time.perf_counter() - start
            return RenderResult(
                values=values,
                inside=inside,
                backend=self.name,
                elapsed_seconds=elapsed,
                details={
                    "precision": request.precision.value,
                    "true_precision_kernel": True,
                    "render_mode": "perturbation",
                    "reference_bits": perturb.reference_bits,
                    "color_iterations": request.effective_color_iterations,
                    "reference_orbit_length": len(perturb.orbit_real),
                    "reference_rebase_limit": perturb.reference_rebase_limit,
                    "rebasing_enabled": True,
                    "glitch_detection_enabled": True,
                    "reference_reused": perturb.reference_reused,
                    "reference_anchor_x": perturb.reference_anchor_x_text,
                    "reference_anchor_y": perturb.reference_anchor_y_text,
                    "rebase_pixels": int(np.count_nonzero(rebased)),
                    "glitch_pixels": int(np.count_nonzero(glitch)),
                },
            )

        dtype = np.float32 if request.precision is Precision.FLOAT32 else np.float64
        kernel = _render_kernel_f32 if request.precision is Precision.FLOAT32 else _render_kernel_f64
        x0, y0, dx, dy = self._geometry(request, dtype)
        start = time.perf_counter()
        values, inside = kernel(
            request.width,
            request.height,
            x0,
            y0,
            dx,
            dy,
            _KIND_CODES[request.fractal],
            request.max_iterations,
            request.effective_color_iterations,
            dtype(request.escape_radius * request.escape_radius),
            dtype(request.julia_c_real),
            dtype(request.julia_c_imag),
            request.exponent,
        )
        elapsed = time.perf_counter() - start
        return RenderResult(
            values=values,
            inside=inside,
            backend=self.name,
            elapsed_seconds=elapsed,
            details={
                "precision": request.precision.value,
                "true_precision_kernel": True,
                "render_mode": "direct",
                "color_iterations": request.effective_color_iterations,
            },
        )
