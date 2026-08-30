from __future__ import annotations

import math
import time

import numpy as np
from numba import cuda, float32, float64

from ..deep_zoom import (
    GLITCH_TOLERANCE,
    MIN_REFERENCE_MAGNITUDE_SQUARED,
    PerturbationReferenceCache,
    should_use_perturbation,
)
from ..models import FractalKind, Precision, RenderRequest
from ..palettes import palette_lut
from .base import FrameResult, Renderer, RenderResult

_KIND_CODES = {
    FractalKind.MANDELBROT: 0,
    FractalKind.JULIA: 1,
    FractalKind.BURNING_SHIP: 2,
    FractalKind.MULTIBROT: 3,
    FractalKind.NEWTON: 4,
}


@cuda.jit(device=True, inline=True)
def _pow_f32(zr, zi, power):
    rr = zr
    ii = zi
    for _ in range(1, power):
        old_rr = rr
        rr = old_rr * zr - ii * zi
        ii = old_rr * zi + ii * zr
    return rr, ii


@cuda.jit(device=True, inline=True)
def _pow_f64(zr, zi, power):
    rr = zr
    ii = zi
    for _ in range(1, power):
        old_rr = rr
        rr = old_rr * zr - ii * zi
        ii = old_rr * zi + ii * zr
    return rr, ii


@cuda.jit(device=True, inline=True)
def _newton_f32(x, y, max_iter):
    zero = float32(0.0)
    one = float32(1.0)
    half = float32(0.5)
    two = float32(2.0)
    three = float32(3.0)
    root_imag = float32(0.8660254037844386)
    tolerance_squared = float32(1e-12)
    denominator_floor = float32(1e-30)
    zr = x
    zi = y

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
        qr = (fr * dr + fi * di) / denominator
        qi = (fi * dr - fr * di) / denominator
        zr -= qr
        zi -= qi

        d0 = (zr - one) * (zr - one) + zi * zi
        d1 = (zr + half) * (zr + half) + (zi - root_imag) * (zi - root_imag)
        d2 = (zr + half) * (zr + half) + (zi + root_imag) * (zi + root_imag)
        local = one - float32(iteration) / float32(max_iter)
        shade = float32(0.12) + float32(0.76) * local
        if d0 < tolerance_squared:
            return shade / three, False
        if d1 < tolerance_squared:
            return (one + shade) / three, False
        if d2 < tolerance_squared:
            return (two + shade) / three, False
    return zero, True


@cuda.jit(device=True, inline=True)
def _newton_f64(x, y, max_iter):
    zr = x
    zi = y
    for iteration in range(max_iter):
        z2r = zr * zr - zi * zi
        z2i = float64(2.0) * zr * zi
        z3r = z2r * zr - z2i * zi
        z3i = z2r * zi + z2i * zr
        fr = z3r - float64(1.0)
        fi = z3i
        dr = float64(3.0) * z2r
        di = float64(3.0) * z2i
        denominator = dr * dr + di * di
        if denominator < float64(1e-30):
            return float64(0.0), True
        qr = (fr * dr + fi * di) / denominator
        qi = (fi * dr - fr * di) / denominator
        zr -= qr
        zi -= qi

        half = float64(0.5)
        root_imag = float64(0.8660254037844386)
        d0 = (zr - float64(1.0)) * (zr - float64(1.0)) + zi * zi
        d1 = (zr + half) * (zr + half) + (zi - root_imag) * (zi - root_imag)
        d2 = (zr + half) * (zr + half) + (zi + root_imag) * (zi + root_imag)
        local = float64(1.0) - float64(iteration) / float64(max_iter)
        shade = float64(0.12) + float64(0.76) * local
        if d0 < float64(1e-12):
            return shade / float64(3.0), False
        if d1 < float64(1e-12):
            return (float64(1.0) + shade) / float64(3.0), False
        if d2 < float64(1e-12):
            return (float64(2.0) + shade) / float64(3.0), False
    return float64(0.0), True


@cuda.jit(device=True, inline=True)
def _escape_f32(x, y, kind, max_iter, escape_squared, julia_r, julia_i, exponent):
    zero = float32(0.0)
    one = float32(1.0)
    two = float32(2.0)
    tiny = float32(1e-30)
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
            next_r, next_i = _pow_f32(zr, zi, exponent)
        else:
            next_r = zr * zr - zi * zi
            next_i = two * zr * zi
        zr = next_r + cr
        zi = next_i + ci
        magnitude_squared = zr * zr + zi * zi
        if magnitude_squared > escape_squared:
            magnitude = float32(math.sqrt(magnitude_squared))
            base = float32(exponent if kind == 3 else 2)
            inner_log = float32(math.log(magnitude))
            if inner_log < tiny:
                inner_log = tiny
            smooth = float32(iteration) + one - float32(math.log(inner_log)) / float32(math.log(base))
            value = smooth / float32(max_iter)
            if value < zero:
                value = zero
            return value, False
    return zero, True


@cuda.jit(device=True, inline=True)
def _escape_f64(x, y, kind, max_iter, escape_squared, julia_r, julia_i, exponent):
    zero = float64(0.0)
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
            next_r, next_i = _pow_f64(zr, zi, exponent)
        else:
            next_r = zr * zr - zi * zi
            next_i = float64(2.0) * zr * zi
        zr = next_r + cr
        zi = next_i + ci
        magnitude_squared = zr * zr + zi * zi
        if magnitude_squared > escape_squared:
            magnitude = math.sqrt(magnitude_squared)
            base = float64(exponent if kind == 3 else 2)
            inner_log = math.log(magnitude)
            if inner_log < float64(1e-30):
                inner_log = float64(1e-30)
            smooth = float64(iteration) + float64(1.0) - math.log(inner_log) / math.log(base)
            value = smooth / float64(max_iter)
            if value < zero:
                value = zero
            return value, False
    return zero, True


@cuda.jit
def _cuda_kernel_f32(
    values,
    inside,
    width,
    height,
    x0,
    y0,
    dx,
    dy,
    kind,
    max_iter,
    escape_squared,
    julia_r,
    julia_i,
    exponent,
):
    px, py = cuda.grid(2)
    if px >= width or py >= height:
        return
    x = x0 + float32(px) * dx
    y = y0 + float32(py) * dy
    if kind == 4:
        value, is_inside = _newton_f32(x, y, max_iter)
    else:
        value, is_inside = _escape_f32(
            x, y, kind, max_iter, escape_squared, julia_r, julia_i, exponent
        )
    values[py, px] = value
    inside[py, px] = is_inside


@cuda.jit
def _cuda_kernel_f64(
    values,
    inside,
    width,
    height,
    x0,
    y0,
    dx,
    dy,
    kind,
    max_iter,
    escape_squared,
    julia_r,
    julia_i,
    exponent,
):
    px, py = cuda.grid(2)
    if px >= width or py >= height:
        return
    x = x0 + float64(px) * dx
    y = y0 + float64(py) * dy
    if kind == 4:
        value, is_inside = _newton_f64(x, y, max_iter)
    else:
        value, is_inside = _escape_f64(
            x, y, kind, max_iter, escape_squared, julia_r, julia_i, exponent
        )
    values[py, px] = float32(value)
    inside[py, px] = is_inside


@cuda.jit
def _cuda_perturb_kernel_f64(
    values,
    inside,
    glitch_repaired,
    rebased,
    width,
    height,
    x0_rel,
    y0_rel,
    dx,
    dy,
    max_iter,
    escape_squared,
    orbit_real,
    orbit_imag,
    reference_rebase_limit,
    glitch_tolerance,
    min_ref_sq,
):
    px, py = cuda.grid(2)
    if px >= width or py >= height:
        return

    dcx = x0_rel + float64(px) * dx
    dcy = y0_rel + float64(py) * dy
    dzr = float64(0.0)
    dzi = float64(0.0)
    reference_index = 0
    zero = float32(0.0)
    tiny = float64(1e-30)
    glitch_repaired[py, px] = False
    rebased[py, px] = False

    for iteration in range(max_iter):
        ref_r = orbit_real[reference_index]
        ref_i = orbit_imag[reference_index]

        if reference_index >= reference_rebase_limit:
            dzr = ref_r + dzr
            dzi = ref_i + dzi
            reference_index = 0
            ref_r = float64(0.0)
            ref_i = float64(0.0)
            rebased[py, px] = True

        actual_r = ref_r + dzr
        actual_i = ref_i + dzi
        delta_sq = dzr * dzr + dzi * dzi
        actual_sq = actual_r * actual_r + actual_i * actual_i
        if reference_index > 0 and actual_sq < delta_sq:
            dzr = actual_r
            dzi = actual_i
            reference_index = 0
            ref_r = float64(0.0)
            ref_i = float64(0.0)
            rebased[py, px] = True

        next_dzr = (
            float64(2.0) * (ref_r * dzr - ref_i * dzi)
            + (dzr * dzr - dzi * dzi)
            + dcx
        )
        next_dzi = (
            float64(2.0) * (ref_r * dzi + ref_i * dzr)
            + float64(2.0) * dzr * dzi
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
            smooth = (
                float64(iteration)
                + float64(1.0)
                - math.log(inner_log) / math.log(float64(2.0))
            )
            value = smooth / float64(max_iter)
            if value < float64(0.0):
                value = float64(0.0)
            values[py, px] = float32(value)
            inside[py, px] = False
            return

        ref_next_sq = (
            orbit_real[next_reference_index] * orbit_real[next_reference_index]
            + orbit_imag[next_reference_index] * orbit_imag[next_reference_index]
        )
        if ref_next_sq < min_ref_sq:
            ref_next_sq = min_ref_sq

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

    values[py, px] = zero
    inside[py, px] = True


@cuda.jit
def _colorize_kernel(values, inside, rgb, lut, cycles, phase):
    px, py = cuda.grid(2)
    height = values.shape[0]
    width = values.shape[1]
    if px >= width or py >= height:
        return
    if inside[py, px]:
        rgb[py, px, 0] = 0
        rgb[py, px, 1] = 0
        rgb[py, px, 2] = 0
        return
    wrapped = values[py, px] * cycles + phase
    wrapped = wrapped - math.floor(wrapped)
    index = int(wrapped * float32(lut.shape[0] - 1))
    if index < 0:
        index = 0
    elif index >= lut.shape[0]:
        index = lut.shape[0] - 1
    rgb[py, px, 0] = lut[index, 0]
    rgb[py, px, 1] = lut[index, 1]
    rgb[py, px, 2] = lut[index, 2]


class CudaRenderer(Renderer):
    name = "cuda-numba"

    def __init__(self) -> None:
        self._shape: tuple[int, int] | None = None
        self._values_device = None
        self._inside_device = None
        self._rgb_device = None
        self._glitch_device = None
        self._rebase_device = None
        self._values_host = None
        self._inside_host = None
        self._rgb_host = None
        self._glitch_host = None
        self._rebase_host = None
        self._stream = None
        self._palette_name: str | None = None
        self._palette_device = None
        self._orbit_real_device = None
        self._orbit_imag_device = None
        self._orbit_reference_key = None
        self._reference_cache = PerturbationReferenceCache()

    def is_available(self) -> bool:
        try:
            return bool(cuda.is_available())
        except Exception:
            return False

    def _ensure_buffers(self, height: int, width: int) -> float:
        if self._shape == (height, width):
            return 0.0
        started = time.perf_counter()
        self._stream = cuda.stream()
        self._values_device = cuda.device_array((height, width), dtype=np.float32, stream=self._stream)
        self._inside_device = cuda.device_array((height, width), dtype=np.bool_, stream=self._stream)
        self._rgb_device = cuda.device_array((height, width, 3), dtype=np.uint8, stream=self._stream)
        self._glitch_device = cuda.device_array((height, width), dtype=np.bool_, stream=self._stream)
        self._rebase_device = cuda.device_array((height, width), dtype=np.bool_, stream=self._stream)
        self._values_host = cuda.pinned_array((height, width), dtype=np.float32)
        self._inside_host = cuda.pinned_array((height, width), dtype=np.bool_)
        self._rgb_host = cuda.pinned_array((height, width, 3), dtype=np.uint8)
        self._glitch_host = cuda.pinned_array((height, width), dtype=np.bool_)
        self._rebase_host = cuda.pinned_array((height, width), dtype=np.bool_)
        self._shape = (height, width)
        return time.perf_counter() - started

    def _ensure_palette(self, name: str) -> float:
        if self._palette_name == name and self._palette_device is not None:
            return 0.0
        started = time.perf_counter()
        lut = np.ascontiguousarray(palette_lut(name), dtype=np.uint8)
        self._palette_device = cuda.to_device(lut, stream=self._stream)
        self._palette_name = name
        return time.perf_counter() - started

    @staticmethod
    def _launch_geometry(request: RenderRequest):
        scalar = np.float32 if request.precision is Precision.FLOAT32 else np.float64
        view_height = request.viewport.width * request.height / request.width
        dx = scalar(request.viewport.width / request.width)
        dy = scalar(view_height / request.height)
        x0 = scalar(request.viewport.center_x - request.viewport.width * 0.5 + float(dx) * 0.5)
        y0 = scalar(request.viewport.center_y - view_height * 0.5 + float(dy) * 0.5)
        return scalar, x0, y0, dx, dy

    @staticmethod
    def _device_name() -> str:
        try:
            raw_name = cuda.current_context().device.name
            if isinstance(raw_name, bytes):
                raw_name = raw_name.decode(errors="replace")
            return str(raw_name)
        except Exception:
            return "NVIDIA CUDA device"

    def _upload_reference_orbit(self, perturb) -> float:
        if self._orbit_reference_key == perturb.reference_key:
            return 0.0
        started = time.perf_counter()
        self._orbit_real_device = cuda.to_device(perturb.orbit_real, stream=self._stream)
        self._orbit_imag_device = cuda.to_device(perturb.orbit_imag, stream=self._stream)
        self._orbit_reference_key = perturb.reference_key
        return time.perf_counter() - started

    def _launch_fractal(self, request: RenderRequest, blocks, threads, perturb=None) -> float:
        if perturb is not None:
            upload_seconds = self._upload_reference_orbit(perturb)
            _cuda_perturb_kernel_f64[blocks, threads, self._stream](
                self._values_device,
                self._inside_device,
                self._glitch_device,
                self._rebase_device,
                request.width,
                request.height,
                np.float64(perturb.x0_rel),
                np.float64(perturb.y0_rel),
                np.float64(perturb.dx),
                np.float64(perturb.dy),
                request.max_iterations,
                np.float64(request.escape_radius * request.escape_radius),
                self._orbit_real_device,
                self._orbit_imag_device,
                perturb.reference_rebase_limit,
                np.float64(GLITCH_TOLERANCE),
                np.float64(MIN_REFERENCE_MAGNITUDE_SQUARED),
            )
            return upload_seconds

        scalar, x0, y0, dx, dy = self._launch_geometry(request)
        kernel = _cuda_kernel_f32 if request.precision is Precision.FLOAT32 else _cuda_kernel_f64
        kernel[blocks, threads, self._stream](
            self._values_device,
            self._inside_device,
            request.width,
            request.height,
            x0,
            y0,
            dx,
            dy,
            _KIND_CODES[request.fractal],
            request.max_iterations,
            scalar(request.escape_radius * request.escape_radius),
            scalar(request.julia_c_real),
            scalar(request.julia_c_imag),
            request.exponent,
        )
        return 0.0

    def render(self, request: RenderRequest) -> RenderResult:
        request.validate()
        if not self.is_available():
            raise RuntimeError("CUDA is not available")

        allocation_seconds = self._ensure_buffers(request.height, request.width)
        threads = (16, 16)
        blocks = (
            (request.width + threads[0] - 1) // threads[0],
            (request.height + threads[1] - 1) // threads[1],
        )
        perturb = self._reference_cache.prepare(request) if should_use_perturbation(request) else None
        started = time.perf_counter()
        orbit_upload_seconds = self._launch_fractal(request, blocks, threads, perturb)
        self._values_device.copy_to_host(self._values_host, stream=self._stream)
        self._inside_device.copy_to_host(self._inside_host, stream=self._stream)
        if perturb is not None:
            self._glitch_device.copy_to_host(self._glitch_host, stream=self._stream)
            self._rebase_device.copy_to_host(self._rebase_host, stream=self._stream)
        self._stream.synchronize()
        values = np.array(self._values_host, copy=True)
        inside = np.array(self._inside_host, copy=True)
        elapsed = time.perf_counter() - started
        details = {
            "precision": request.precision.value,
            "device": self._device_name(),
            "allocation_seconds": allocation_seconds,
            "persistent_buffers": True,
            "render_mode": "perturbation" if perturb is not None else "direct",
            "reference_bits": perturb.reference_bits if perturb is not None else 0,
            "reference_rebase_limit": perturb.reference_rebase_limit if perturb is not None else 0,
            "reference_upload_seconds": orbit_upload_seconds,
            "reference_reused": perturb.reference_reused if perturb is not None else False,
            "reference_anchor_x": perturb.reference_anchor_x_text if perturb is not None else "",
            "reference_anchor_y": perturb.reference_anchor_y_text if perturb is not None else "",
        }
        if perturb is not None:
            details["rebasing_enabled"] = True
            details["glitch_detection_enabled"] = True
            details["glitch_pixels"] = int(np.count_nonzero(self._glitch_host))
            details["rebase_pixels"] = int(np.count_nonzero(self._rebase_host))
        return RenderResult(
            values=values,
            inside=inside,
            backend=self.name,
            elapsed_seconds=elapsed,
            details=details,
        )

    def render_frame(
        self,
        request: RenderRequest,
        palette: str = "inferno",
        cycles: float = 1.0,
        phase: float = 0.0,
    ) -> FrameResult:
        request.validate()
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if not self.is_available():
            raise RuntimeError("CUDA is not available")

        allocation_seconds = self._ensure_buffers(request.height, request.width)
        palette_seconds = self._ensure_palette(palette)
        threads = (16, 16)
        blocks = (
            (request.width + threads[0] - 1) // threads[0],
            (request.height + threads[1] - 1) // threads[1],
        )
        perturb = self._reference_cache.prepare(request) if should_use_perturbation(request) else None

        started = time.perf_counter()
        orbit_upload_seconds = self._launch_fractal(request, blocks, threads, perturb)
        _colorize_kernel[blocks, threads, self._stream](
            self._values_device,
            self._inside_device,
            self._rgb_device,
            self._palette_device,
            np.float32(cycles),
            np.float32(phase),
        )
        self._rgb_device.copy_to_host(self._rgb_host, stream=self._stream)
        self._stream.synchronize()
        rgb = np.array(self._rgb_host, copy=True)
        elapsed = time.perf_counter() - started

        return FrameResult(
            rgb=rgb,
            backend=self.name,
            elapsed_seconds=elapsed,
            details={
                "precision": request.precision.value,
                "device": self._device_name(),
                "allocation_seconds": allocation_seconds,
                "palette_upload_seconds": palette_seconds,
                "persistent_buffers": True,
                "optimized_frame_path": True,
                "transfer": "single RGB readback",
                "render_mode": "perturbation" if perturb is not None else "direct",
                "reference_bits": perturb.reference_bits if perturb is not None else 0,
                "reference_rebase_limit": perturb.reference_rebase_limit if perturb is not None else 0,
                "reference_upload_seconds": orbit_upload_seconds,
                "reference_reused": perturb.reference_reused if perturb is not None else False,
                "reference_anchor_x": perturb.reference_anchor_x_text if perturb is not None else "",
                "reference_anchor_y": perturb.reference_anchor_y_text if perturb is not None else "",
                "rebasing_enabled": perturb is not None,
                "glitch_detection_enabled": perturb is not None,
            },
        )
