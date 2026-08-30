from __future__ import annotations

import json
from pathlib import Path

import pytest

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.flight_path import (
    CameraPath,
    CenterInterpolation,
    Easing,
    FlightKeyframe,
)
from fractal_flight_studio.flight_plan import (
    FlightPlanDefaults,
    FlightPlanDocument,
    FlightScene,
    PaletteTransition,
    RenderCue,
    RenderProfile,
    RenderTrack,
)
from fractal_flight_studio.flight_plan_io import (
    FLIGHT_PLAN_EXTENSION,
    FLIGHT_PLAN_FORMAT,
    FLIGHT_PLAN_SCHEMA_VERSION,
    LEGACY_FLIGHT_PLAN_FORMAT,
    FlightPlanFormatError,
    deserialize_flight_plan,
    load_flight_plan,
    save_flight_plan,
    serialize_flight_plan,
    suggested_flight_plan_name,
)
from fractal_flight_studio.models import FractalKind
from fractal_flight_studio.surface_lighting import SurfaceLightingSettings


def _document() -> FlightPlanDocument:
    path = CameraPath(
        (
            FlightKeyframe(
                "0",
                CameraState("-0.5", "0", "4"),
                Easing.SMOOTHERSTEP,
                CenterInterpolation.FOCUS,
            ),
            FlightKeyframe(
                "2.5000000000000000001",
                CameraState(
                    "-0.743643887037158704752191506114774",
                    "0.131825904205311970493132056385139",
                    "1e-420",
                ),
                Easing.LINEAR,
                CenterInterpolation.LINEAR,
            ),
        ),
        digits=180,
    )
    return FlightPlanDocument(
        name="Seahorse Δ",
        path=path,
        scene=FlightScene(FractalKind.MANDELBROT, 2, "-0.8", "0.156"),
        render_track=RenderTrack(
            (
                RenderCue(
                    "0",
                    RenderProfile(800, 256, "inferno", "1.25"),
                    PaletteTransition.HOLD,
                ),
                RenderCue(
                    "1.25",
                    RenderProfile(3000, 768, "ocean", "2.5"),
                    PaletteTransition.BLEND,
                ),
            ),
            digits=180,
        ),
        surface_lighting=SurfaceLightingSettings(
            enabled=True,
            strength=2.25,
            azimuth_degrees=210.0,
            elevation_degrees=52.0,
            ambient=0.3,
            diffuse=0.7,
        ),
    )


def _legacy_payload() -> dict[str, object]:
    current = json.loads(serialize_flight_plan(_document()))
    return {
        "format": LEGACY_FLIGHT_PLAN_FORMAT,
        "schema_version": 1,
        "name": current["name"],
        "digits": current["digits"],
        "keyframes": current["camera_keyframes"],
    }


def test_round_trip_preserves_exact_camera_scene_and_render_timeline(tmp_path: Path) -> None:
    target = tmp_path / f"seahorse{FLIGHT_PLAN_EXTENSION}"

    save_flight_plan(target, _document())
    loaded = load_flight_plan(target)

    assert loaded == _document()
    assert loaded.source_schema_version == FLIGHT_PLAN_SCHEMA_VERSION
    assert loaded.path.keyframes[1].camera.center_x_text.endswith("4774")
    assert loaded.path.keyframes[1].camera.view_width_text == "1e-420"
    assert loaded.path.keyframes[0].center_interpolation is CenterInterpolation.FOCUS
    assert loaded.scene.fractal is FractalKind.MANDELBROT
    assert loaded.render_track.cues[1].profile.max_iterations == 3000
    assert loaded.render_track.cues[1].palette_transition is PaletteTransition.BLEND
    assert loaded.path.digits == loaded.render_track.digits == 180


def test_serialization_is_deterministic_utf8_and_uses_string_numbers() -> None:
    first = serialize_flight_plan(_document())
    second = serialize_flight_plan(_document())
    value = json.loads(first)

    assert first == second
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert "Seahorse Δ" in first
    assert value["format"] == FLIGHT_PLAN_FORMAT
    assert value["schema_version"] == FLIGHT_PLAN_SCHEMA_VERSION
    assert value["camera_keyframes"][1]["center_x"].endswith("4774")
    assert isinstance(value["camera_keyframes"][1]["view_width"], str)
    assert isinstance(value["scene"]["julia_c_real"], str)
    assert value["surface_lighting"] == {
        "enabled": True,
        "strength": 2.25,
        "azimuth_degrees": 210.0,
        "elevation_degrees": 52.0,
        "ambient": 0.3,
        "diffuse": 0.7,
    }
    assert isinstance(value["render_cues"][1]["cycles"], str)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version=99), "unsupported schema_version"),
        (lambda value: value.update(format="other"), "format must be"),
        (lambda value: value.update(digits=True), "digits must be an integer"),
        (lambda value: value.update(extra=True), "unexpected 'extra'"),
        (lambda value: value.pop("name"), "missing 'name'"),
        (
            lambda value: value.update(camera_keyframes=[]),
            "at least two camera keyframes",
        ),
        (
            lambda value: value["camera_keyframes"][1].update(view_width=1e-20),
            "view_width must be a string",
        ),
        (
            lambda value: value["camera_keyframes"][0].update(easing="bounce"),
            "'bounce' is not a valid Easing",
        ),
        (
            lambda value: value["scene"].update(exponent=True),
            "scene.exponent must be an integer",
        ),
        (lambda value: value.update(render_cues=[]), "at least one render cue"),
        (
            lambda value: value["render_cues"][0].update(max_iterations="800"),
            "max_iterations must be an integer",
        ),
        (
            lambda value: value["render_cues"][0].update(palette="unknown"),
            "unknown render palette",
        ),
        (
            lambda value: value["render_cues"][0].update(cycles=1.0),
            "cycles must be a string",
        ),
        (
            lambda value: value["render_cues"][1].update(palette_transition="wipe"),
            "'wipe' is not a valid PaletteTransition",
        ),
        (
            lambda value: value["surface_lighting"].update(enabled=1),
            "enabled must be a boolean",
        ),
        (
            lambda value: value["surface_lighting"].update(elevation_degrees=0),
            "elevation_degrees must be in the interval",
        ),
        (
            lambda value: value["surface_lighting"].pop("ambient"),
            "missing 'ambient'",
        ),
    ],
)
def test_schema_validation_rejects_invalid_documents(mutation, message: str) -> None:
    value = json.loads(serialize_flight_plan(_document()))
    mutation(value)

    with pytest.raises(FlightPlanFormatError, match=message):
        deserialize_flight_plan(json.dumps(value))


def test_schema_one_migrates_with_caller_defaults_and_resaves_as_schema_three() -> None:
    lighting = SurfaceLightingSettings(enabled=True, strength=1.8)
    defaults = FlightPlanDefaults(
        FlightScene(FractalKind.MULTIBROT, 4, "-0.7", "0.2"),
        RenderProfile(2200, 640, "ember", "1.75"),
        lighting,
    )

    migrated = deserialize_flight_plan(
        json.dumps(_legacy_payload()),
        source="legacy.json",
        migration_defaults=defaults,
    )

    assert migrated.source_schema_version == 1
    assert migrated.scene == defaults.scene
    assert migrated.render_track.first_profile == defaults.render_profile
    assert migrated.surface_lighting == lighting
    assert migrated.render_track.cues[0].time_seconds_text == "0"
    current = json.loads(serialize_flight_plan(migrated))
    assert current["schema_version"] == FLIGHT_PLAN_SCHEMA_VERSION
    assert current["format"] == FLIGHT_PLAN_FORMAT
    assert "camera_keyframes" in current and "render_cues" in current
    assert "keyframes" not in current


def test_schema_two_migrates_with_disabled_lighting_and_resaves_as_schema_three() -> None:
    value = json.loads(serialize_flight_plan(_document()))
    value["schema_version"] = 2
    value.pop("surface_lighting")

    migrated = deserialize_flight_plan(json.dumps(value), source="schema2.json")

    assert migrated.source_schema_version == 2
    assert migrated.surface_lighting == SurfaceLightingSettings()
    current = json.loads(serialize_flight_plan(migrated))
    assert current["schema_version"] == FLIGHT_PLAN_SCHEMA_VERSION
    assert current["surface_lighting"]["enabled"] is False


def test_legacy_format_is_required_for_schema_one() -> None:
    value = _legacy_payload()
    value["format"] = FLIGHT_PLAN_FORMAT

    with pytest.raises(FlightPlanFormatError, match="for schema 1"):
        deserialize_flight_plan(json.dumps(value))


def test_duplicate_json_members_and_nonstandard_constants_are_rejected() -> None:
    with pytest.raises(FlightPlanFormatError, match="duplicate JSON member 'name'"):
        deserialize_flight_plan(
            '{"schema_version":2,"format":"x","name":"x","name":"y",'
            '"digits":80,"scene":{},"camera_keyframes":[],"render_cues":[]}'
        )

    with pytest.raises(FlightPlanFormatError, match="invalid JSON constant"):
        deserialize_flight_plan(
            '{"schema_version":NaN,"format":"x","name":"x",'
            '"digits":80,"scene":{},"camera_keyframes":[],"render_cues":[]}'
        )


def test_invalid_camera_path_is_reported_with_source_context() -> None:
    value = json.loads(serialize_flight_plan(_document()))
    value["camera_keyframes"][0]["time_seconds"] = "1"

    with pytest.raises(FlightPlanFormatError, match="plan.json: invalid camera path"):
        deserialize_flight_plan(json.dumps(value), source="plan.json")


def test_render_cue_cannot_extend_beyond_camera_path() -> None:
    value = json.loads(serialize_flight_plan(_document()))
    value["render_cues"][1]["time_seconds"] = "3"

    with pytest.raises(FlightPlanFormatError, match="must not extend beyond"):
        deserialize_flight_plan(json.dumps(value), source="plan.json")


def test_load_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(FlightPlanFormatError, match="not valid UTF-8"):
        load_flight_plan(path)


def test_atomic_save_keeps_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fractal_flight_studio.flight_plan_io as module

    target = tmp_path / "plan.json"
    target.write_text("old content", encoding="utf-8")

    def fail_replace(_source, _target) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        save_flight_plan(target, _document())

    assert target.read_text(encoding="utf-8") == "old content"
    assert list(tmp_path.glob(".plan.json.*.tmp")) == []


def test_suggested_name_strips_known_extensions() -> None:
    assert suggested_flight_plan_name("My Zoom.fractal-flight.json") == "My Zoom"
    assert suggested_flight_plan_name("plain.json") == "plain"
    assert suggested_flight_plan_name(".json") == "Flugplan"
