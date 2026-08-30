from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
import json
import platform
from pathlib import Path
import statistics
import time
from typing import Callable, Protocol

from fractal_flight_studio.deep_zoom_targets import DeepZoomTarget, deep_zoom_target
from fractal_flight_studio.models import (
    FractalKind,
    Precision,
    RenderMode,
    RenderRequest,
    Viewport,
)
from fractal_flight_studio.renderers import CpuRenderer, CudaRenderer
from fractal_flight_studio.renderers.base import FrameResult


class _FrameRenderer(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def render_frame(self, request: RenderRequest, **kwargs) -> FrameResult: ...


@dataclass(frozen=True, slots=True)
class PerturbationBenchmarkConfig:
    target_id: str = "seahorse-satellite"
    width: int = 1280
    height: int = 720
    repeats: int = 3
    max_iterations: int | None = None
    reference_bits: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("benchmark dimensions must be positive")
        if self.repeats < 1:
            raise ValueError("benchmark repeats must be at least 1")
        if self.max_iterations is not None and not 1 <= self.max_iterations <= 100_000:
            raise ValueError("benchmark iterations must be between 1 and 100000")
        if self.reference_bits is not None and not 64 <= self.reference_bits <= 16_384:
            raise ValueError("benchmark reference bits must be between 64 and 16384")


def _request_for_target(
    target: DeepZoomTarget,
    config: PerturbationBenchmarkConfig,
) -> RenderRequest:
    if target.fractal is not FractalKind.MANDELBROT:
        raise ValueError("perturbation benchmark targets must be Mandelbrot views")
    iterations = config.max_iterations or target.recommended_iterations
    reference_bits = config.reference_bits or target.reference_bits
    return RenderRequest(
        width=config.width,
        height=config.height,
        viewport=Viewport(
            float(target.center_x_text),
            float(target.center_y_text),
            float(target.view_width_text),
        ),
        fractal=target.fractal,
        max_iterations=iterations,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=reference_bits,
        center_x_text=target.center_x_text,
        center_y_text=target.center_y_text,
        view_width_text=target.view_width_text,
    )


def _warmup_request(request: RenderRequest) -> RenderRequest:
    """Compile and allocate without populating the measured target reference."""

    return RenderRequest(
        width=request.width,
        height=request.height,
        viewport=Viewport(0.25, 0.0, 0.5),
        max_iterations=min(64, request.max_iterations),
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=64,
        center_x_text="0.25",
        center_y_text="0",
        view_width_text="0.5",
    )


def _timed_frame(
    renderer: _FrameRenderer,
    request: RenderRequest,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[FrameResult, float]:
    started = clock()
    result = renderer.render_frame(
        request,
        tone_mapping="linear",
        tone_state=None,
        tone_scene_key=("perturbation-benchmark", request.center_x_text, request.view_width_text),
        tone_smoothing=1.0,
    )
    return result, clock() - started


def _rgb_sha256(result: FrameResult) -> str:
    return hashlib.sha256(memoryview(result.rgb).cast("B")).hexdigest()


def _selected_details(result: FrameResult) -> dict[str, object]:
    names = (
        "device",
        "render_mode",
        "precision",
        "reference_bits",
        "reference_rebase_limit",
        "reference_upload_seconds",
        "reference_reused",
        "compute_seconds",
        "color_seconds",
        "transfer_seconds",
        "tone_analysis_seconds",
        "optimized_frame_path",
    )
    return {name: result.details[name] for name in names if name in result.details}


def _benchmark_backend(
    renderer: _FrameRenderer,
    request: RenderRequest,
    repeats: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, object]:
    _timed_frame(renderer, _warmup_request(request), clock=clock)

    cold_result, cold_wall = _timed_frame(renderer, request, clock=clock)
    if cold_result.details.get("render_mode") != "perturbation":
        raise RuntimeError(f"{renderer.name} did not execute the perturbation path")
    if cold_result.details.get("reference_reused") is not False:
        raise RuntimeError(f"{renderer.name} target reference was unexpectedly warm")

    warm_wall_times: list[float] = []
    warm_frame_times: list[float] = []
    warm_results: list[FrameResult] = []
    for _ in range(repeats):
        result, wall = _timed_frame(renderer, request, clock=clock)
        if result.details.get("reference_reused") is not True:
            raise RuntimeError(f"{renderer.name} did not reuse the target reference")
        warm_results.append(result)
        warm_wall_times.append(wall)
        warm_frame_times.append(result.elapsed_seconds)

    warm_wall = statistics.median(warm_wall_times)
    warm_frame = statistics.median(warm_frame_times)
    pixels = request.width * request.height
    megapixels = pixels / 1_000_000
    last = warm_results[-1]
    return {
        "backend": last.backend,
        "size": [request.width, request.height],
        "pixels": pixels,
        "iterations": request.max_iterations,
        "reference_bits": request.reference_bits,
        "cold_wall_seconds": cold_wall,
        "cold_frame_seconds": cold_result.elapsed_seconds,
        "cold_reference_reused": False,
        "warm_wall_seconds_median": warm_wall,
        "warm_frame_seconds_median": warm_frame,
        "warm_fps": 1.0 / warm_wall,
        "warm_mpix_per_second": megapixels / warm_wall,
        "warm_reference_reused": True,
        "rgb_sha256": _rgb_sha256(last),
        "cold_details": _selected_details(cold_result),
        "warm_details": _selected_details(last),
    }


def _comparison(records: list[dict[str, object]]) -> dict[str, object] | None:
    by_backend = {str(record["backend"]): record for record in records}
    cpu = by_backend.get("cpu-numba")
    cuda = by_backend.get("cuda-numba")
    if cpu is None or cuda is None:
        return None

    cpu_warm = float(cpu["warm_wall_seconds_median"])
    cuda_warm = float(cuda["warm_wall_seconds_median"])
    cpu_cold = float(cpu["cold_wall_seconds"])
    cuda_cold = float(cuda["cold_wall_seconds"])
    warm_ratio = cpu_warm / cuda_warm
    cold_ratio = cpu_cold / cuda_cold
    return {
        "rgb_outputs_match": cpu.get("rgb_sha256") == cuda.get("rgb_sha256"),
        "warm_winner": "cuda-numba" if warm_ratio >= 1.0 else "cpu-numba",
        "warm_cuda_speedup_over_cpu": warm_ratio,
        "cold_winner": "cuda-numba" if cold_ratio >= 1.0 else "cpu-numba",
        "cold_cuda_speedup_over_cpu": cold_ratio,
    }


def _backend_names(preference: str) -> list[str]:
    if preference != "all":
        return [preference]
    names = ["cpu"]
    if CudaRenderer().is_available():
        names.append("cuda")
    return names


def _create_renderer(name: str) -> _FrameRenderer:
    if name == "cpu":
        return CpuRenderer()
    if name == "cuda":
        renderer = CudaRenderer()
        if not renderer.is_available():
            raise RuntimeError("CUDA backend is not available")
        return renderer
    raise ValueError(f"unknown perturbation benchmark backend: {name}")


def _numba_threads() -> int | None:
    try:
        import numba

        return int(numba.get_num_threads())
    except Exception:
        return None


def _print_record(record: dict[str, object]) -> None:
    print(
        f"{str(record['backend']):11s} "
        f"cold {float(record['cold_wall_seconds']) * 1000:9.2f} ms  "
        f"warm {float(record['warm_wall_seconds_median']) * 1000:9.2f} ms  "
        f"{float(record['warm_fps']):7.2f} FPS  "
        f"{float(record['warm_mpix_per_second']):8.2f} MPix/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPU and CUDA FP64 perturbation rendering with a cold target "
            "reference and steady-state reference reuse"
        )
    )
    parser.add_argument("--backend", default="all", choices=["all", "cpu", "cuda"])
    parser.add_argument("--target", default="seahorse-satellite")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--reference-bits", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("perturbation-benchmark-results.json"),
    )
    args = parser.parse_args()

    try:
        config = PerturbationBenchmarkConfig(
            target_id=args.target,
            width=args.width,
            height=args.height,
            repeats=args.repeats,
            max_iterations=args.iterations,
            reference_bits=args.reference_bits,
        )
        target = deep_zoom_target(config.target_id)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    request = _request_for_target(target, config)
    records: list[dict[str, object]] = []
    for backend_name in _backend_names(args.backend):
        record = _benchmark_backend(
            _create_renderer(backend_name),
            request,
            config.repeats,
        )
        records.append(record)
        _print_record(record)

    comparison = _comparison(records)
    if comparison is not None:
        print(
            "Warm winner: "
            f"{comparison['warm_winner']} "
            f"(CUDA/CPU speed factor {float(comparison['warm_cuda_speedup_over_cpu']):.3f})"
        )

    report = {
        "system": platform.platform(),
        "processor": platform.processor(),
        "numba_threads": _numba_threads(),
        "python": platform.python_version(),
        "measurement": (
            "forced FP64 perturbation render_frame with linear tone mapping; wall time "
            "includes reference preparation, kernel, colorization and required host readback"
        ),
        "warmup": (
            "same-size forced perturbation at a distant reference to compile kernels and "
            "allocate buffers without warming the measured target reference"
        ),
        "target": {
            "id": target.id,
            "name": target.name,
            "center_x": target.center_x_text,
            "center_y": target.center_y_text,
            "view_width": target.view_width_text,
        },
        "config": {
            "size": [config.width, config.height],
            "repeats": config.repeats,
            "iterations": request.max_iterations,
            "reference_bits": request.reference_bits,
        },
        "results": records,
        "comparison": comparison,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
