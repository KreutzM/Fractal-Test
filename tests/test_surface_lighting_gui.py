from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fractal_flight_studio.app import FractalStudioApp
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.service import render_rgb
from fractal_flight_studio.surface_lighting import (
    CUSTOM_SURFACE_LIGHTING_PRESET,
    SURFACE_LIGHTING_PRESETS,
    SurfaceLightingSettings,
    surface_lighting_preset_for,
    surface_lighting_preset_names,
    surface_lighting_settings_for_preset,
)


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _app_settings(**changes):
    values = {
        "surface_lighting_preset_var": _Var(CUSTOM_SURFACE_LIGHTING_PRESET),
        "surface_lighting_enabled_var": _Var(False),
        "surface_lighting_strength_var": _Var(1.5),
        "surface_lighting_azimuth_var": _Var(315.0),
        "surface_lighting_elevation_var": _Var(45.0),
    }
    for name, value in changes.items():
        values[name] = _Var(value)
    return SimpleNamespace(
        **values,
        _surface_lighting_ambient=0.35,
        _surface_lighting_diffuse=0.65,
        _applying_surface_lighting_preset=False,
    )


def test_app_surface_lighting_defaults_are_disabled() -> None:
    settings = FractalStudioApp._surface_lighting_settings(_app_settings())

    assert settings == SurfaceLightingSettings()


def test_app_surface_lighting_reads_gui_values() -> None:
    settings = FractalStudioApp._surface_lighting_settings(
        _app_settings(
            surface_lighting_enabled_var=True,
            surface_lighting_strength_var=2.25,
            surface_lighting_azimuth_var=210.0,
            surface_lighting_elevation_var=52.0,
        )
    )

    assert settings.enabled is True
    assert settings.strength == 2.25
    assert settings.azimuth_degrees == 210.0
    assert settings.elevation_degrees == 52.0
    assert settings.ambient == 0.35
    assert settings.diffuse == 0.65


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("surface_lighting_strength_var", -0.1),
        ("surface_lighting_azimuth_var", float("nan")),
        ("surface_lighting_elevation_var", 0.0),
    ],
)
def test_app_surface_lighting_rejects_invalid_gui_values(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        FractalStudioApp._surface_lighting_settings(
            _app_settings(**{field: value})
        )



def test_surface_lighting_presets_are_complete_distinct_settings() -> None:
    names = surface_lighting_preset_names()

    assert names[0] == CUSTOM_SURFACE_LIGHTING_PRESET
    assert names[1:] == tuple(preset.name for preset in SURFACE_LIGHTING_PRESETS)
    assert len(set(names)) == len(names)
    assert len({preset.settings for preset in SURFACE_LIGHTING_PRESETS}) == len(
        SURFACE_LIGHTING_PRESETS
    )
    assert all(preset.settings.enabled for preset in SURFACE_LIGHTING_PRESETS)


def test_app_applies_complete_surface_lighting_preset() -> None:
    app = _app_settings(surface_lighting_preset_var="Dramatisch")
    app._surface_lighting_settings = lambda: FractalStudioApp._surface_lighting_settings(app)
    app._refresh_surface_lighting_preset = lambda *_args: (
        FractalStudioApp._refresh_surface_lighting_preset(app)
    )

    FractalStudioApp.apply_surface_lighting_preset(app)

    expected = surface_lighting_settings_for_preset("Dramatisch")
    assert FractalStudioApp._surface_lighting_settings(app) == expected
    assert app.surface_lighting_preset_var.get() == "Dramatisch"


def test_manual_surface_lighting_values_are_custom() -> None:
    settings = surface_lighting_settings_for_preset("Sanft")

    assert surface_lighting_preset_for(settings) == "Sanft"
    assert (
        surface_lighting_preset_for(
            SurfaceLightingSettings(
                enabled=True,
                strength=settings.strength + 0.1,
                azimuth_degrees=settings.azimuth_degrees,
                elevation_degrees=settings.elevation_degrees,
                ambient=settings.ambient,
                diffuse=settings.diffuse,
            )
        )
        == CUSTOM_SURFACE_LIGHTING_PRESET
    )


def test_unknown_surface_lighting_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown surface-lighting preset"):
        surface_lighting_settings_for_preset("Unbekannt")

def test_render_rgb_forwards_surface_lighting(monkeypatch) -> None:
    captured = {}

    class _Renderer:
        def render_frame(self, request, *args, **kwargs):
            captured["request"] = request
            captured["surface_lighting"] = kwargs.get("surface_lighting")
            return SimpleNamespace(
                rgb=np.zeros((request.height, request.width, 3), dtype=np.uint8),
                backend="fake",
                details={},
            )

    monkeypatch.setattr(
        "fractal_flight_studio.service.select_renderer",
        lambda _backend: _Renderer(),
    )
    request = RenderRequest(width=12, height=8, max_iterations=20)
    settings = SurfaceLightingSettings(enabled=True, strength=2.0)

    rgb, result = render_rgb(
        request,
        backend="fake",
        surface_lighting=settings,
    )

    assert rgb.shape == (8, 12, 3)
    assert result.backend == "fake"
    assert captured["request"] is request
    assert captured["surface_lighting"] is settings
