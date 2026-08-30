import pytest

from fractal_flight_studio.models import RenderMode, RenderRequest, Viewport


def test_zoom_keeps_selected_complex_point_stable():
    viewport = Viewport(-0.5, 0.0, 3.5)
    before = viewport.pixel_to_complex(700, 220, 1000, 600)
    zoomed = viewport.zoom_at(700, 220, 1000, 600, 2.0)
    after = zoomed.pixel_to_complex(700, 220, 1000, 600)
    assert after == before
    assert zoomed.width == 1.75


def test_request_validation():
    RenderRequest(width=10, height=10).validate()
    RenderRequest(width=10, height=10, render_mode=RenderMode.PERTURBATION, reference_bits=512).validate()


def test_color_iteration_scale_defaults_to_escape_limit_and_validates_bounds():
    default = RenderRequest(max_iterations=120)
    default.validate()
    assert default.effective_color_iterations == 120

    stable = RenderRequest(max_iterations=120, color_iterations=400)
    stable.validate()
    assert stable.effective_color_iterations == 400

    with pytest.raises(ValueError, match="color_iterations"):
        RenderRequest(max_iterations=120, color_iterations=119).validate()
