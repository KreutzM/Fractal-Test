from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_stable_benchmark_module():
    script = Path(__file__).parents[1] / "scripts" / "benchmark_double_single_stable.py"
    spec = importlib.util.spec_from_file_location("benchmark_double_single_stable", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_choose_batch_launches_targets_sustained_duration() -> None:
    module = _load_stable_benchmark_module()
    assert module.choose_batch_launches(0.010, 0.250, 4, 256) == 25
    assert module.choose_batch_launches(0.001, 0.250, 4, 256) == 250


def test_choose_batch_launches_respects_limits() -> None:
    module = _load_stable_benchmark_module()
    assert module.choose_batch_launches(1.0, 0.250, 4, 256) == 4
    assert module.choose_batch_launches(1e-6, 0.250, 4, 256) == 256
    assert module.choose_batch_launches(0.0, 0.250, 4, 256) == 256


def test_stable_benchmark_config_rejects_invalid_timing_parameters() -> None:
    module = _load_stable_benchmark_module()
    with pytest.raises(ValueError, match="repeats"):
        module.StableBenchmarkConfig(repeats=2)
    with pytest.raises(ValueError, match="batch target"):
        module.StableBenchmarkConfig(batch_target_seconds=0.0)
    with pytest.raises(ValueError, match="maximum batch"):
        module.StableBenchmarkConfig(min_batch_launches=8, max_batch_launches=4)
