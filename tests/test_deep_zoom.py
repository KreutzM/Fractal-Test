from __future__ import annotations

import numpy as np

from fractal_flight_studio.deep_zoom import (
    PerturbationReferenceCache,
    pixel_grid_quality,
    prepare_perturbation,
    should_use_perturbation,
)
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers.cpu import CpuRenderer


DEEP_CENTER_X = "-0.743643887037158704752191506114774"
DEEP_CENTER_Y = "0.131825904205311970493132056385139"


def test_pixel_grid_quality_detects_collapsed_neighbours():
    quality = pixel_grid_quality(1.0, 0.0, 1e-18, 1e-18, 64, 48)
    assert quality.safe is False
    assert quality.x_unique_fraction < 1.0
    assert quality.maximum_equal_run > 1


def test_reference_cache_reanchors_before_relative_grid_collapses():
    cache = PerturbationReferenceCache()
    cache.prepare(
        RenderRequest(
            width=64,
            height=48,
            max_iterations=50,
            precision=Precision.FLOAT64,
            render_mode=RenderMode.PERTURBATION,
            reference_bits=256,
            center_x_text="-0.5",
            center_y_text="0.0",
            view_width_text="3.5",
            viewport=Viewport(-0.5, 0.0, 3.5),
        )
    )
    deep = cache.prepare(
        RenderRequest(
            width=64,
            height=48,
            max_iterations=50,
            precision=Precision.FLOAT64,
            render_mode=RenderMode.PERTURBATION,
            reference_bits=256,
            center_x_text=DEEP_CENTER_X,
            center_y_text=DEEP_CENTER_Y,
            view_width_text="1e-16",
            viewport=Viewport(float(DEEP_CENTER_X), float(DEEP_CENTER_Y), 1e-16),
        )
    )
    assert deep.reference_reanchored_for_grid is True
    assert deep.reference_reused is False
    assert deep.grid_quality.safe is True


def test_auto_switches_to_perturbation_for_tiny_pixels():
    request = RenderRequest(
        width=320,
        height=200,
        viewport=Viewport(-0.7436438870371587, 0.13182590420531198, 1e-18),
        precision=Precision.FLOAT64,
        center_x_text=DEEP_CENTER_X,
        center_y_text=DEEP_CENTER_Y,
        view_width_text="1e-18",
    )
    assert should_use_perturbation(request) is True


def test_prepare_perturbation_builds_reference_orbit():
    request = RenderRequest(
        width=64,
        height=48,
        max_iterations=64,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        center_x_text=DEEP_CENTER_X,
        center_y_text=DEEP_CENTER_Y,
        view_width_text="1e-20",
        viewport=Viewport(-0.7436438870371587, 0.13182590420531198, 1e-20),
    )
    perturb = prepare_perturbation(request)
    assert perturb.orbit_real.shape == (65,)
    assert perturb.orbit_imag.shape == (65,)
    assert perturb.dx != 0.0
    assert perturb.dy != 0.0


def test_cpu_perturbation_matches_direct_at_normal_zoom():
    renderer = CpuRenderer()
    request = RenderRequest(width=40, height=30, max_iterations=60, precision=Precision.FLOAT64)
    direct = renderer.render(request)
    perturb = renderer.render(
        RenderRequest(
            width=request.width,
            height=request.height,
            viewport=request.viewport,
            fractal=request.fractal,
            max_iterations=request.max_iterations,
            precision=request.precision,
            render_mode=RenderMode.PERTURBATION,
            center_x_text=repr(request.viewport.center_x),
            center_y_text=repr(request.viewport.center_y),
            view_width_text=repr(request.viewport.width),
        )
    )
    assert perturb.details["render_mode"] == "perturbation"
    assert perturb.details["rebasing_enabled"] is True
    assert perturb.details["glitch_detection_enabled"] is True
    assert np.allclose(perturb.values, direct.values, atol=5e-5)
    assert np.array_equal(perturb.inside, direct.inside)


def test_cpu_perturbation_reports_rebases_on_overview():
    renderer = CpuRenderer()
    request = RenderRequest(
        width=120,
        height=90,
        max_iterations=200,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="3.5",
        viewport=Viewport(-0.5, 0.0, 3.5),
    )
    result = renderer.render(request)
    assert result.details["rebase_pixels"] > 0
    assert result.details["glitch_pixels"] >= 0


def test_cpu_perturbation_renders_deep_zoom_structure():
    renderer = CpuRenderer()
    request = RenderRequest(
        width=48,
        height=36,
        max_iterations=400,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=256,
        center_x_text="-0.1011",
        center_y_text="0.9563",
        view_width_text="1e-8",
        viewport=Viewport(-0.1011, 0.9563, 1e-8),
    )
    result = renderer.render(request)
    assert result.values.shape == (36, 48)
    assert result.details["reference_bits"] == 256
    assert np.unique(result.values).size > 4
    assert (~result.inside).any()


def test_reference_cache_stays_anchored_for_overlapping_pan():
    cache = PerturbationReferenceCache()
    first = RenderRequest(
        width=160,
        height=120,
        max_iterations=300,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=256,
        center_x_text="-0.1011",
        center_y_text="0.9563",
        view_width_text="1e-8",
        viewport=Viewport(-0.1011, 0.9563, 1e-8),
    )
    first_data = cache.prepare(first)
    shift_pixels = 7
    shifted_center = "-0.1010999995625"
    second = RenderRequest(
        width=160,
        height=120,
        max_iterations=300,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=256,
        center_x_text=shifted_center,
        center_y_text="0.9563",
        view_width_text="1e-8",
        viewport=Viewport(float(shifted_center), 0.9563, 1e-8),
    )
    second_data = cache.prepare(second)
    assert first_data.reference_reused is False
    assert second_data.reference_reused is True
    assert second_data.reference_key == first_data.reference_key
    assert np.isclose(
        second_data.x0_rel - first_data.x0_rel,
        shift_pixels * first_data.dx,
        rtol=0.0,
        atol=1e-24,
    )


def test_cached_reference_makes_integer_pixel_pan_bit_stable():
    renderer = CpuRenderer()
    width = 120
    height = 90
    shift_pixels = 5
    view_width = 3.5
    pixel_width = view_width / width
    first = RenderRequest(
        width=width,
        height=height,
        max_iterations=200,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=256,
        center_x_text="-0.5",
        center_y_text="0.0",
        view_width_text="3.5",
        viewport=Viewport(-0.5, 0.0, 3.5),
    )
    shifted_x = -0.5 + shift_pixels * pixel_width
    second = RenderRequest(
        width=width,
        height=height,
        max_iterations=200,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=256,
        center_x_text=repr(shifted_x),
        center_y_text="0.0",
        view_width_text="3.5",
        viewport=Viewport(shifted_x, 0.0, 3.5),
    )
    result_a = renderer.render(first)
    result_b = renderer.render(second)
    assert result_b.details["reference_reused"] is True
    assert np.array_equal(
        result_a.values[:, shift_pixels:],
        result_b.values[:, : width - shift_pixels],
    )
    assert np.array_equal(
        result_a.inside[:, shift_pixels:],
        result_b.inside[:, : width - shift_pixels],
    )


def test_reference_selector_prefers_long_lived_candidate():
    request = RenderRequest(
        width=100,
        height=80,
        max_iterations=100,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=256,
        center_x_text="0.5",
        center_y_text="0.0",
        view_width_text="3.5",
        viewport=Viewport(0.5, 0.0, 3.5),
    )
    perturb = prepare_perturbation(request)
    assert perturb.reference_anchor_x_text != "0.5"
    assert perturb.reference_rebase_limit == request.max_iterations
