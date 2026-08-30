from __future__ import annotations

import time

import numpy as np
from numba import cuda

from ..deep_zoom import GLITCH_TOLERANCE, direct_pixel_size, direct_precision_pixel_limit
from ..models import FractalKind, Precision, RenderMode, RenderRequest
from .cuda_double_single import (
    cuda_mandelbrot_double_single_kernel,
    launch_geometry as double_single_launch_geometry,
)
from .cuda_double_single_perturbation import (
    DoubleSinglePerturbationUnavailable,
    cuda_mandelbrot_double_single_perturbation_kernel,
    prepare_perturbation_launch,
)
from .cuda_tonemap import CudaRenderer as _BaseCudaRenderer


class CudaRenderer(_BaseCudaRenderer):
    """CUDA renderer with internal double-single Mandelbrot precision tiers."""

    def __init__(self) -> None:
        super().__init__()
        self._last_arithmetic = Precision.FLOAT32.value
        self._last_double_single_mode = ""
        self._last_double_single_fallback_reason = ""
        self._ds_orbit_real_hi_device = None
        self._ds_orbit_real_lo_device = None
        self._ds_orbit_imag_hi_device = None
        self._ds_orbit_imag_lo_device = None
        self._ds_orbit_reference_key = None

    @staticmethod
    def _can_use_double_single(request: RenderRequest) -> bool:
        """Return whether AUTO direct rendering may use the DS Mandelbrot tier."""

        if request.render_mode is not RenderMode.AUTO:
            return False
        if request.precision is not Precision.FLOAT64:
            return False
        if request.fractal is not FractalKind.MANDELBROT:
            return False
        if request.width >= 1 << 24 or request.height >= 1 << 24:
            return False
        return direct_pixel_size(request) > direct_precision_pixel_limit(
            request, Precision.FLOAT64
        )

    @staticmethod
    def _can_use_double_single_perturbation(
        request: RenderRequest, perturb
    ) -> bool:
        """Return whether AUTO perturbation may attempt the guarded DS tier."""

        return (
            perturb is not None
            and request.render_mode is RenderMode.AUTO
            and request.precision is Precision.FLOAT64
            and request.fractal is FractalKind.MANDELBROT
            and request.width < 1 << 24
            and request.height < 1 << 24
        )

    def _upload_double_single_reference_orbit(self, perturb, orbit) -> float:
        if self._ds_orbit_reference_key == perturb.reference_key:
            return 0.0
        started = time.perf_counter()
        self._ds_orbit_real_hi_device = cuda.to_device(
            orbit.real_hi, stream=self._stream
        )
        self._ds_orbit_real_lo_device = cuda.to_device(
            orbit.real_lo, stream=self._stream
        )
        self._ds_orbit_imag_hi_device = cuda.to_device(
            orbit.imag_hi, stream=self._stream
        )
        self._ds_orbit_imag_lo_device = cuda.to_device(
            orbit.imag_lo, stream=self._stream
        )
        self._ds_orbit_reference_key = perturb.reference_key
        return time.perf_counter() - started

    def _launch_double_single_perturbation(
        self, request: RenderRequest, blocks, threads, perturb
    ) -> float:
        launch = prepare_perturbation_launch(perturb)
        upload_seconds = self._upload_double_single_reference_orbit(
            perturb, launch.orbit
        )
        cuda_mandelbrot_double_single_perturbation_kernel[
            blocks, threads, self._stream
        ](
            self._values_device,
            self._inside_device,
            self._glitch_device,
            self._rebase_device,
            request.width,
            request.height,
            launch.x0.hi,
            launch.x0.lo,
            launch.y0.hi,
            launch.y0.lo,
            launch.dx.hi,
            launch.dx.lo,
            launch.dy.hi,
            launch.dy.lo,
            request.max_iterations,
            np.float32(request.escape_radius * request.escape_radius),
            self._ds_orbit_real_hi_device,
            self._ds_orbit_real_lo_device,
            self._ds_orbit_imag_hi_device,
            self._ds_orbit_imag_lo_device,
            perturb.reference_rebase_limit,
            np.float32(GLITCH_TOLERANCE),
            np.float32(np.finfo(np.float32).tiny),
        )
        return upload_seconds

    def _launch_fractal(self, request: RenderRequest, blocks, threads, perturb=None) -> float:
        self._last_double_single_mode = ""
        self._last_double_single_fallback_reason = ""

        if perturb is not None:
            if self._can_use_double_single_perturbation(request, perturb):
                try:
                    upload_seconds = self._launch_double_single_perturbation(
                        request, blocks, threads, perturb
                    )
                except DoubleSinglePerturbationUnavailable as exc:
                    self._last_arithmetic = Precision.FLOAT64.value
                    self._last_double_single_fallback_reason = exc.reason
                    return super()._launch_fractal(request, blocks, threads, perturb)
                self._last_arithmetic = "double-single"
                self._last_double_single_mode = "perturbation"
                return upload_seconds

            self._last_arithmetic = Precision.FLOAT64.value
            self._last_double_single_fallback_reason = "routing-policy"
            return super()._launch_fractal(request, blocks, threads, perturb)

        if not self._can_use_double_single(request):
            self._last_arithmetic = (
                Precision.FLOAT64.value
                if request.precision is Precision.FLOAT64
                else Precision.FLOAT32.value
            )
            return super()._launch_fractal(request, blocks, threads, perturb)

        try:
            geometry = double_single_launch_geometry(request)
        except (OverflowError, ValueError):
            self._last_arithmetic = Precision.FLOAT64.value
            self._last_double_single_fallback_reason = "direct-coordinate-range"
            return super()._launch_fractal(request, blocks, threads, perturb)

        cuda_mandelbrot_double_single_kernel[blocks, threads, self._stream](
            self._values_device,
            self._inside_device,
            request.width,
            request.height,
            *geometry,
            request.max_iterations,
            np.float32(request.escape_radius * request.escape_radius),
        )
        self._last_arithmetic = "double-single"
        self._last_double_single_mode = "direct"
        return 0.0

    def _annotate_arithmetic(self, result):
        enabled = self._last_arithmetic == "double-single"
        result.details["arithmetic"] = self._last_arithmetic
        result.details["double_single_enabled"] = enabled
        result.details["double_single_mode"] = self._last_double_single_mode
        result.details[
            "double_single_perturbation_enabled"
        ] = enabled and self._last_double_single_mode == "perturbation"
        result.details[
            "double_single_fallback_reason"
        ] = self._last_double_single_fallback_reason
        return result

    def render(self, request: RenderRequest):
        return self._annotate_arithmetic(super().render(request))

    def render_frame(self, request: RenderRequest, *args, **kwargs):
        return self._annotate_arithmetic(super().render_frame(request, *args, **kwargs))
