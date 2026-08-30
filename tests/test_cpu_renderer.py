from dataclasses import replace

import numpy as np
import pytest

from fractal_flight_studio.models import FractalKind, Precision, RenderRequest, Viewport
from fractal_flight_studio.renderers.cpu import CpuRenderer


@pytest.fixture(scope="module")
def renderer():
    instance = CpuRenderer()
    instance.warm_up()
    return instance


@pytest.mark.parametrize("kind", list(FractalKind))
def test_all_fractals_render_finite_arrays(renderer, kind):
    request = RenderRequest(
        width=64,
        height=48,
        fractal=kind,
        max_iterations=60,
        viewport=Viewport(0.0, 0.0, 4.0) if kind is FractalKind.NEWTON else Viewport(),
    )
    result = renderer.render(request)
    assert result.values.shape == (48, 64)
    assert result.inside.shape == (48, 64)
    assert np.isfinite(result.values).all()
    assert (result.values >= 0).all()
    assert (result.values <= 1).all()


def test_mandelbrot_known_inside_and_outside(renderer):
    request = RenderRequest(width=3, height=1, viewport=Viewport(0.0, 0.0, 4.0), max_iterations=200)
    result = renderer.render(request)
    # Center pixel samples c=0; the right pixel samples c=4/3.
    assert result.inside[0, 1]
    assert not result.inside[0, 2]


def test_float32_and_float64_are_close_for_normal_view(renderer):
    request = RenderRequest(width=80, height=60, max_iterations=80, precision=Precision.FLOAT32)
    a = renderer.render(request)
    b = renderer.render(replace(request, precision=Precision.FLOAT64))
    agreement = np.mean(a.inside == b.inside)
    assert agreement > 0.995


def test_escape_values_remain_stable_when_only_iteration_limit_increases(renderer):
    low_request = RenderRequest(
        width=96,
        height=64,
        max_iterations=80,
        color_iterations=160,
        precision=Precision.FLOAT64,
    )
    high_request = replace(low_request, max_iterations=160)

    low = renderer.render(low_request)
    high = renderer.render(high_request)
    escaped_with_low_limit = ~low.inside

    assert np.any(escaped_with_low_limit)
    assert np.array_equal(high.inside[escaped_with_low_limit], low.inside[escaped_with_low_limit])
    assert np.allclose(
        high.values[escaped_with_low_limit],
        low.values[escaped_with_low_limit],
        rtol=0.0,
        atol=1e-7,
    )
