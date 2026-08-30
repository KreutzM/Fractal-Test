from __future__ import annotations

from types import SimpleNamespace

import mpmath as mp
import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_app import FractalStudioApp
from fractal_flight_studio.flight_controller import FlightController
from fractal_flight_studio.deep_zoom_targets import (
    deep_zoom_target,
    favorite_deep_zoom_targets,
    load_deep_zoom_targets,
    parse_deep_zoom_catalog,
)
from fractal_flight_studio.models import FractalKind


class _Var:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_packaged_catalog_loads_ten_curated_targets():
    targets = load_deep_zoom_targets()

    assert len(targets) == 10
    assert favorite_deep_zoom_targets() == targets
    assert len({target.id for target in targets}) == len(targets)
    assert len({target.name.casefold() for target in targets}) == len(targets)
    assert all(target.fractal is FractalKind.MANDELBROT for target in targets)
    assert all(mp.mpf(target.view_width_text) > 0 for target in targets)
    assert all(target.source_url.startswith("https://") for target in targets)


def test_catalog_preserves_exact_coordinate_text():
    target = deep_zoom_target("seahorse-satellite")

    assert target.center_x_text == "-0.74364386269"
    assert target.center_y_text == "0.13182590271"
    assert target.view_width_text == "0.00000013526"
    assert "1200 Iterationen" in target.recommendation_text


def test_catalog_rejects_duplicate_ids():
    record = {
        "id": "duplicate",
        "name": "First",
        "description": "Description",
        "fractal": "mandelbrot",
        "center_x": "-0.5",
        "center_y": "0",
        "view_width": "0.1",
        "recommended_iterations": 400,
        "reference_bits": 256,
        "palette": "inferno",
        "tags": ["Test"],
        "favorite": True,
        "source_url": "https://example.com/first",
    }
    duplicate = dict(record, name="Second", source_url="https://example.com/second")

    with pytest.raises(ValueError, match="duplicate deep-zoom target id"):
        parse_deep_zoom_catalog({"schema_version": 1, "targets": [record, duplicate]})


def test_apply_catalog_target_keeps_text_precision_and_recommendations():
    target = deep_zoom_target("diamond-cross-junction")
    controller = FlightController()
    controller.running = True
    app = SimpleNamespace(
        flight_controller=controller,
        camera=CameraState(),
        fractal_var=_Var("julia"),
        iterations_var=_Var(20),
        reference_bits_var=_Var(128),
        palette_var=_Var("inferno"),
        position_var=_Var(""),
        render_count=0,
        stop_count=0,
    )
    app._stop_flight = lambda: setattr(app, "stop_count", app.stop_count + 1)
    app.request_render = lambda: setattr(app, "render_count", app.render_count + 1)

    FractalStudioApp._apply_deep_zoom_target(app, target, load_view=False)

    assert app.stop_count == 0
    assert app.fractal_var.get() == "mandelbrot"
    assert app.iterations_var.get() == 1500
    assert app.reference_bits_var.get() == 384
    assert app.palette_var.get() == "electric"
    assert controller.target_text is None
    assert app.camera == CameraState()
    assert app.render_count == 0

    controller.running = False
    FractalStudioApp._apply_deep_zoom_target(app, target, load_view=True)

    assert app.camera == CameraState(
        target.center_x_text,
        target.center_y_text,
        target.view_width_text,
    )
    assert app.render_count == 1
    assert "Diamond Cross Junction als Ansicht geladen" in app.position_var.get()
