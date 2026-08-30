"""Small CUDA-simulator smoke test used by the test suite."""

from __future__ import annotations

import numpy as np

from .models import FractalKind, Precision, RenderRequest
from .renderers.cpu import CpuRenderer
from .renderers.cuda import CudaRenderer


def main() -> int:
    request = RenderRequest(
        width=24,
        height=16,
        fractal=FractalKind.JULIA,
        max_iterations=40,
        precision=Precision.FLOAT32,
    )
    cpu = CpuRenderer().render(request)
    cuda = CudaRenderer().render(request)
    if not np.array_equal(cpu.inside, cuda.inside):
        raise SystemExit("CUDA simulator mask differs from CPU")
    if not np.allclose(cpu.values, cuda.values, atol=2e-4, rtol=2e-4):
        maximum = float(np.max(np.abs(cpu.values - cuda.values)))
        raise SystemExit(f"CUDA simulator values differ from CPU: {maximum}")
    print("CUDA simulator smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
