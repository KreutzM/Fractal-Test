from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

_DEFAULT_SMOOTHING = 0.16
_MIN_WINDOW = 1e-9


@dataclass(slots=True)
class ToneMapState:
    """Temporally smoothed parameters for an automatic tone curve."""

    mode: str
    scene_key: tuple[Any, ...] | None
    low: float
    high: float
    strength: float
    gamma: float


def tone_mapping_names() -> tuple[str, ...]:
    return ("auto", "linear", "asinh")


def finite_outside_samples(values: np.ndarray, inside: np.ndarray) -> np.ndarray:
    if values.shape != inside.shape:
        raise ValueError("values and inside masks must have the same shape")
    mask = (~inside) & np.isfinite(values)
    return np.asarray(values[mask], dtype=np.float64)



def sample_grid(width: int, height: int, maximum: int = 4096) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    total = width * height
    if total <= maximum:
        return width, height
    grid_x = max(1, int(math.sqrt(maximum * width / max(height, 1))))
    grid_y = max(1, maximum // grid_x)
    return grid_x, grid_y


def stratified_samples(
    values: np.ndarray,
    inside: np.ndarray,
    maximum: int = 4096,
) -> np.ndarray:
    """Return deterministic image-wide samples without biasing one region.

    The CPU path uses the same bounded sample budget as the CUDA path. This
    keeps automatic parameters consistent while avoiding a full percentile sort
    for large frames.
    """

    if values.shape != inside.shape:
        raise ValueError("values and inside masks must have the same shape")
    if maximum <= 0:
        raise ValueError("maximum must be positive")

    height, width = values.shape
    total = height * width
    if total <= maximum:
        return finite_outside_samples(values, inside)

    grid_x, grid_y = sample_grid(width, height, maximum)
    xs = np.minimum(((np.arange(grid_x) + 0.5) * width / grid_x).astype(np.int64), width - 1)
    ys = np.minimum(((np.arange(grid_y) + 0.5) * height / grid_y).astype(np.int64), height - 1)
    sampled_values = values[np.ix_(ys, xs)].reshape(-1)
    sampled_inside = inside[np.ix_(ys, xs)].reshape(-1)
    mask = (~sampled_inside) & np.isfinite(sampled_values)
    return np.asarray(sampled_values[mask], dtype=np.float64)


def _robust_window(samples: np.ndarray) -> tuple[float, float]:
    if samples.size == 0:
        return 0.0, 1.0
    if samples.size < 24:
        low = float(np.min(samples))
        high = float(np.max(samples))
    else:
        low, high = (float(x) for x in np.percentile(samples, (0.35, 99.65)))
    if not math.isfinite(low) or not math.isfinite(high):
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return 0.0, 1.0
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high <= low:
        center = 0.5 * (high + low)
        half = max(abs(center) * 1e-6, 1e-6)
        low = center - half
        high = center + half
    return low, high


def _auto_targets(samples: np.ndarray, mode: str) -> tuple[float, float, float, float]:
    low, high = _robust_window(samples)
    window = max(high - low, _MIN_WINDOW)
    normalized = np.clip((samples - low) / window, 0.0, 1.0)
    if normalized.size == 0:
        return low, high, 1.0, 1.0

    p10, p50, p90 = (float(x) for x in np.percentile(normalized, (10.0, 50.0, 90.0)))
    spread = max(p90 - p10, 0.025)

    # Narrow clusters benefit from stronger highlight compression. Wide
    # distributions remain close to linear to preserve natural palette spacing.
    strength = float(np.clip(1.65 / spread, 2.0, 14.0))
    if mode == "asinh":
        return low, high, strength, 1.0

    # Move the median gently toward the perceptually useful middle without
    # forcing every frame to identical brightness.
    desired_median = 0.46
    compressed_median = math.asinh(strength * max(p50, 1e-8)) / math.asinh(strength)
    if compressed_median <= 1e-8 or compressed_median >= 1.0:
        gamma = 1.0
    else:
        gamma = math.log(desired_median) / math.log(compressed_median)
    gamma = float(np.clip(gamma, 0.62, 1.38))
    return low, high, strength, gamma


def _blend(previous: float, target: float, alpha: float) -> float:
    return previous + alpha * (target - previous)


def _scene_changed(
    state: ToneMapState | None,
    mode: str,
    scene_key: tuple[Any, ...] | None,
) -> bool:
    return state is None or state.mode != mode or state.scene_key != scene_key


def resolve_tone_state(
    samples: np.ndarray,
    mode: str = "auto",
    state: ToneMapState | None = None,
    scene_key: tuple[Any, ...] | None = None,
    smoothing: float = _DEFAULT_SMOOTHING,
) -> tuple[ToneMapState | None, dict[str, Any]]:
    """Resolve robust and temporally stable tone-curve parameters."""

    if mode not in tone_mapping_names():
        raise ValueError(f"unknown tone mapping mode: {mode}")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must be in the interval (0, 1]")

    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if mode == "linear" or samples.size == 0:
        return None, {
            "tone_mapping": "linear",
            "tone_low": 0.0,
            "tone_high": 1.0,
            "tone_strength": 1.0,
            "tone_gamma": 1.0,
            "tone_scene_reset": state is not None,
            "tone_sample_count": int(samples.size),
        }

    target_low, target_high, target_strength, target_gamma = _auto_targets(samples, mode)
    reset = _scene_changed(state, mode, scene_key)
    if reset:
        next_state = ToneMapState(
            mode=mode,
            scene_key=scene_key,
            low=target_low,
            high=target_high,
            strength=target_strength,
            gamma=target_gamma,
        )
    else:
        assert state is not None
        previous_window = max(state.high - state.low, _MIN_WINDOW)
        target_window = max(target_high - target_low, _MIN_WINDOW)
        center_shift = abs((target_low + target_high) - (state.low + state.high)) * 0.5 / previous_window
        ratio = max(previous_window, target_window) / max(min(previous_window, target_window), _MIN_WINDOW)

        # Large jumps should converge faster; ordinary pan/zoom frames stay
        # heavily damped to avoid visible exposure pumping.
        alpha = smoothing
        if center_shift > 1.5 or ratio > 8.0:
            alpha = max(alpha, 0.55)
        elif center_shift > 0.55 or ratio > 2.5:
            alpha = max(alpha, 0.32)

        next_state = ToneMapState(
            mode=mode,
            scene_key=scene_key,
            low=_blend(state.low, target_low, alpha),
            high=_blend(state.high, target_high, alpha),
            strength=_blend(state.strength, target_strength, alpha),
            gamma=_blend(state.gamma, target_gamma, alpha),
        )

    return next_state, {
        "tone_mapping": mode,
        "tone_low": next_state.low,
        "tone_high": next_state.high,
        "tone_strength": next_state.strength,
        "tone_gamma": next_state.gamma,
        "tone_scene_reset": reset,
        "tone_sample_count": int(samples.size),
    }



def resolve_locked_tone_state(
    mode: str,
    state: ToneMapState | None,
    scene_key: tuple[Any, ...] | None = None,
) -> tuple[ToneMapState | None, dict[str, Any]]:
    """Return an already planned tone state without re-analyzing the frame."""

    if mode not in tone_mapping_names():
        raise ValueError(f"unknown tone mapping mode: {mode}")
    if mode == "linear":
        if state is not None:
            raise ValueError("linear tone mapping does not accept a locked tone state")
        return None, {
            "tone_mapping": "linear",
            "tone_low": 0.0,
            "tone_high": 1.0,
            "tone_strength": 1.0,
            "tone_gamma": 1.0,
            "tone_scene_reset": False,
            "tone_sample_count": 0,
            "tone_state_locked": True,
        }
    if state is None:
        raise ValueError("locked automatic tone mapping requires a tone state")
    if state.mode != mode:
        raise ValueError(
            f"locked tone state mode {state.mode!r} does not match requested mode {mode!r}"
        )
    if scene_key is not None and state.scene_key != scene_key:
        raise ValueError("locked tone state does not match the requested scene")
    return state, {
        "tone_mapping": mode,
        "tone_low": state.low,
        "tone_high": state.high,
        "tone_strength": state.strength,
        "tone_gamma": state.gamma,
        "tone_scene_reset": False,
        "tone_sample_count": 0,
        "tone_state_locked": True,
    }

def apply_curve(values: np.ndarray, state: ToneMapState | None) -> np.ndarray:
    if state is None:
        return np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)

    window = max(state.high - state.low, _MIN_WINDOW)
    mapped = np.clip((values.astype(np.float32) - state.low) / window, 0.0, 1.0)
    denominator = math.asinh(state.strength)
    if denominator > 0.0:
        mapped = np.arcsinh(state.strength * mapped) / denominator
    if state.mode == "auto" and state.gamma != 1.0:
        mapped = np.power(mapped, state.gamma, dtype=np.float32)
    return np.clip(mapped, 0.0, 1.0).astype(np.float32, copy=False)


def apply_tone_mapping(
    values: np.ndarray,
    inside: np.ndarray,
    mode: str = "auto",
    state: ToneMapState | None = None,
    scene_key: tuple[Any, ...] | None = None,
    smoothing: float = _DEFAULT_SMOOTHING,
    locked: bool = False,
) -> tuple[np.ndarray, ToneMapState | None, dict[str, Any]]:
    if locked:
        next_state, details = resolve_locked_tone_state(mode, state, scene_key)
    else:
        samples = stratified_samples(values, inside)
        next_state, details = resolve_tone_state(samples, mode, state, scene_key, smoothing)
        details["tone_state_locked"] = False
    return apply_curve(values, next_state), next_state, details
