from __future__ import annotations

import numpy as np
import pytest

from fractal_flight_studio.surface_lighting import (
    SurfaceLightingSettings,
    apply_surface_lighting,
)
from fractal_flight_studio.tonemapping import ToneMapState


def _rgb(shape: tuple[int, int], value: int = 160) -> np.ndarray:
    return np.full(shape + (3,), value, dtype=np.uint8)


def _enabled(**changes: float | bool) -> SurfaceLightingSettings:
    defaults: dict[str, float | bool] = {
        "enabled": True,
        "strength": 2.0,
        "azimuth_degrees": 315.0,
        "elevation_degrees": 45.0,
        "ambient": 0.25,
        "diffuse": 0.75,
    }
    defaults.update(changes)
    return SurfaceLightingSettings(**defaults)


def test_disabled_surface_lighting_is_byte_identical_and_returns_copy() -> None:
    values = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)
    inside = np.array([[False, True], [False, False]], dtype=np.bool_)
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    result = apply_surface_lighting(values, inside, rgb)

    assert np.array_equal(result, rgb)
    assert result is not rgb
    result[0, 0] = 255
    assert not np.array_equal(result, rgb)


def test_surface_lighting_does_not_mutate_inputs() -> None:
    values = np.array([[0.0, 0.4, 1.0]], dtype=np.float32)
    inside = np.array([[False, True, False]], dtype=np.bool_)
    rgb = _rgb((1, 3))
    original_values = values.copy()
    original_inside = inside.copy()
    original_rgb = rgb.copy()

    apply_surface_lighting(values, inside, rgb, _enabled())

    assert np.array_equal(values, original_values)
    assert np.array_equal(inside, original_inside)
    assert np.array_equal(rgb, original_rgb)


def test_flat_height_field_produces_uniform_lighting() -> None:
    values = np.full((4, 5), 0.5, dtype=np.float32)
    inside = np.zeros(values.shape, dtype=np.bool_)
    rgb = _rgb(values.shape, 200)

    result = apply_surface_lighting(
        values,
        inside,
        rgb,
        _enabled(elevation_degrees=30.0, ambient=0.2, diffuse=0.8),
    )

    assert np.unique(result.reshape(-1, 3), axis=0).shape == (1, 3)
    assert np.array_equal(result[0, 0], np.array([200, 200, 200], dtype=np.uint8))


def test_horizontal_ramp_is_brighter_when_lit_from_the_downhill_side() -> None:
    row = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    values = np.tile(row, (3, 1))
    inside = np.zeros(values.shape, dtype=np.bool_)
    rgb = _rgb(values.shape)

    lit_from_left = apply_surface_lighting(
        values,
        inside,
        rgb,
        _enabled(azimuth_degrees=180.0),
    )
    lit_from_right = apply_surface_lighting(
        values,
        inside,
        rgb,
        _enabled(azimuth_degrees=0.0),
    )

    assert float(lit_from_left.mean()) > float(lit_from_right.mean())


def test_vertical_ramp_uses_screen_space_azimuth() -> None:
    column = np.linspace(0.0, 1.0, 7, dtype=np.float32)[:, np.newaxis]
    values = np.tile(column, (1, 3))
    inside = np.zeros(values.shape, dtype=np.bool_)
    rgb = _rgb(values.shape)

    lit_from_top = apply_surface_lighting(
        values,
        inside,
        rgb,
        _enabled(azimuth_degrees=270.0),
    )
    lit_from_bottom = apply_surface_lighting(
        values,
        inside,
        rgb,
        _enabled(azimuth_degrees=90.0),
    )

    assert float(lit_from_top.mean()) > float(lit_from_bottom.mean())


def test_inside_pixels_remain_byte_identical() -> None:
    values = np.array(
        [[0.0, 0.4, 0.8], [0.1, 0.5, 0.9], [0.2, 0.6, 1.0]],
        dtype=np.float32,
    )
    inside = np.array(
        [[False, False, False], [False, True, False], [False, False, False]],
        dtype=np.bool_,
    )
    rgb = _rgb(values.shape)
    rgb[1, 1] = (7, 11, 13)

    result = apply_surface_lighting(values, inside, rgb, _enabled())

    assert np.array_equal(result[inside], rgb[inside])
    assert np.any(result[~inside] != rgb[~inside])


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_degenerate_image_dimensions_are_supported(shape: tuple[int, int]) -> None:
    values = np.linspace(0.0, 1.0, shape[0] * shape[1], dtype=np.float32).reshape(shape)
    inside = np.zeros(shape, dtype=np.bool_)
    rgb = _rgb(shape)

    result = apply_surface_lighting(values, inside, rgb, _enabled())

    assert result.shape == shape + (3,)
    assert result.dtype == np.uint8


def test_values_are_clipped_before_gradient_evaluation() -> None:
    clipped = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    outside_range = np.array([[-10.0, -1.0, 2.0, 12.0]], dtype=np.float32)
    inside = np.zeros(clipped.shape, dtype=np.bool_)
    rgb = _rgb(clipped.shape)

    expected = apply_surface_lighting(clipped, inside, rgb, _enabled())
    actual = apply_surface_lighting(outside_range, inside, rgb, _enabled())

    assert np.array_equal(actual, expected)


def test_surface_lighting_is_deterministic() -> None:
    values = np.linspace(0.0, 1.0, 30, dtype=np.float32).reshape(5, 6)
    inside = values < 0.2
    rgb = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
    settings = _enabled(azimuth_degrees=23.5, elevation_degrees=61.0)

    first = apply_surface_lighting(values, inside, rgb, settings)
    second = apply_surface_lighting(values, inside, rgb, settings)

    assert np.array_equal(first, second)


def test_tone_mapped_height_reveals_compressed_iteration_gradients() -> None:
    values = np.tile(
        np.linspace(0.705, 0.735, 9, dtype=np.float32),
        (5, 1),
    )
    inside = np.zeros(values.shape, dtype=np.bool_)
    rgb = _rgb(values.shape, 160)
    tone_state = ToneMapState(
        mode="auto",
        scene_key=("compressed-height",),
        low=0.70,
        high=0.74,
        strength=3.0,
        gamma=1.0,
    )

    lit_from_left = apply_surface_lighting(
        values,
        inside,
        rgb,
        _enabled(azimuth_degrees=180.0, elevation_degrees=35.0),
        tone_state=tone_state,
    )
    lit_from_right = apply_surface_lighting(
        values,
        inside,
        rgb,
        _enabled(azimuth_degrees=0.0, elevation_degrees=35.0),
        tone_state=tone_state,
    )

    assert abs(float(lit_from_left.mean()) - float(lit_from_right.mean())) > 20.0
    assert np.count_nonzero(lit_from_left != lit_from_right) > values.size


def test_surface_lighting_rejects_invalid_tone_state() -> None:
    values = np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(3, 3)
    inside = np.zeros(values.shape, dtype=np.bool_)

    with pytest.raises(ValueError, match="ToneMapState"):
        apply_surface_lighting(
            values,
            inside,
            _rgb(values.shape),
            _enabled(),
            tone_state=True,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": 1},
        {"strength": -0.1},
        {"strength": float("nan")},
        {"azimuth_degrees": float("inf")},
        {"elevation_degrees": 0.0},
        {"elevation_degrees": 90.1},
        {"ambient": -0.1},
        {"ambient": 1.1},
        {"diffuse": -0.1},
        {"diffuse": 1.1},
    ],
)
def test_invalid_settings_are_rejected(
    changes: dict[str, float | bool],
) -> None:
    with pytest.raises(ValueError):
        _enabled(**changes)


@pytest.mark.parametrize(
    ("values", "inside", "rgb"),
    [
        (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.bool_),
            np.empty((0, 2, 3), dtype=np.uint8),
        ),
        (
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((2, 2), dtype=np.bool_),
            _rgb((2, 2)),
        ),
        (
            np.array([[0.0, np.nan]], dtype=np.float32),
            np.zeros((1, 2), dtype=np.bool_),
            _rgb((1, 2)),
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.uint8),
            _rgb((2, 2)),
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.bool_),
            np.zeros((2, 2, 4), dtype=np.uint8),
        ),
        (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.bool_),
            np.zeros((2, 2, 3), dtype=np.float32),
        ),
    ],
)
def test_invalid_input_arrays_are_rejected(
    values: np.ndarray,
    inside: np.ndarray,
    rgb: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        apply_surface_lighting(values, inside, rgb, _enabled())
