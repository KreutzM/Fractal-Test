from __future__ import annotations

from dataclasses import replace

from ..deep_zoom import direct_pixel_grid_quality, effective_direct_precision
from ..models import Precision, RenderRequest
from .base import FrameResult, Renderer, RenderResult
from .cpu import CpuRenderer
from .cuda_double_single_renderer import CudaRenderer


class AdaptivePrecisionRenderer(Renderer):
    """Promote unsafe float32 auto frames before delegating to a backend."""

    def __init__(self, delegate: Renderer) -> None:
        self._delegate = delegate
        self.name = delegate.name

    def is_available(self) -> bool:
        return self._delegate.is_available()

    @staticmethod
    def _effective_request(request: RenderRequest) -> tuple[RenderRequest, Precision]:
        precision = effective_direct_precision(request)
        if precision is request.precision:
            return request, precision
        return replace(request, precision=precision), precision

    @staticmethod
    def _annotate(
        result: RenderResult | FrameResult, request: RenderRequest, requested: Precision
    ) -> None:
        render_mode = result.details.get("render_mode", "direct")
        effective = (
            Precision.FLOAT64
            if render_mode == "perturbation"
            else Precision(result.details.get("precision", requested.value))
        )
        result.details["requested_precision"] = requested.value
        result.details["precision"] = effective.value
        result.details["precision_promoted"] = effective is not requested
        if render_mode == "direct":
            result.details.update(direct_pixel_grid_quality(request, effective).as_details())

    def render(self, request: RenderRequest) -> RenderResult:
        effective_request, _ = self._effective_request(request)
        result = self._delegate.render(effective_request)
        self._annotate(result, effective_request, request.precision)
        return result

    def render_frame(self, request: RenderRequest, *args, **kwargs) -> FrameResult:
        effective_request, _ = self._effective_request(request)
        result = self._delegate.render_frame(effective_request, *args, **kwargs)
        self._annotate(result, effective_request, request.precision)
        return result


# Reuse backend instances. In particular, this keeps CUDA contexts, streams and
# device buffers alive across animation frames instead of reallocating them for
# every render request. It also preserves temporally smoothed automatic tone
# parameters across related GUI frames.
_CPU_RENDERER = AdaptivePrecisionRenderer(CpuRenderer())
_CUDA_RENDERER = AdaptivePrecisionRenderer(CudaRenderer())


def available_renderers() -> dict[str, Renderer]:
    candidates: tuple[Renderer, ...] = (_CUDA_RENDERER, _CPU_RENDERER)
    return {renderer.name: renderer for renderer in candidates if renderer.is_available()}


def select_renderer(preference: str = "auto") -> Renderer:
    renderers = available_renderers()
    if preference == "auto":
        return renderers.get("cuda-numba", renderers["cpu-numba"])
    if preference in ("cpu", "cpu-numba"):
        return _CPU_RENDERER
    if preference in ("cuda", "cuda-numba"):
        if not _CUDA_RENDERER.is_available():
            from ..gpu_info import inspect_cuda

            raise RuntimeError(inspect_cuda().summary)
        return _CUDA_RENDERER
    raise ValueError(f"unknown backend preference: {preference}")
