from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_cuda_double_single_perturbation_matrix_entry as entry  # noqa: E402


class FakeRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, request):
        self.calls += 1
        return SimpleNamespace(
            values=np.zeros((1, 1), dtype=np.float32),
            inside=np.ones((1, 1), dtype=np.bool_),
            elapsed_seconds=0.001,
            details={"reference_reused": self.calls > 1},
        )


def test_warm_preserves_first_reference_reuse_state() -> None:
    renderer = FakeRenderer()
    result, launches, initial_reused = entry._warm(renderer, object(), 0.0)
    assert launches == 2
    assert initial_reused is False
    assert result.details["reference_reused"] is True


def test_measure_pair_exposes_initial_reuse_to_matrix_gate() -> None:
    auto = FakeRenderer()
    native = FakeRenderer()
    result, *_ = entry._measure_pair(auto, object(), native, object(), 1, 0.0)
    assert result.details["reference_reused"] is False
    assert auto.calls >= 3
