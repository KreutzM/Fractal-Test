from __future__ import annotations

from pathlib import Path

import mpmath as mp
import pytest

from fractal_flight_studio.flight_plan_io import (
    FLIGHT_PLAN_SCHEMA_VERSION,
    load_flight_plan,
    deserialize_flight_plan,
    serialize_flight_plan,
)
from fractal_flight_studio.models import FractalKind


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "flight_plans"
EXPECTED = {
    "seahorse-odyssey.fractal-flight.json": (
        "-0.743643887037151",
        "0.13182590420533",
        "5e-13",
    ),
    "diamond-crossfire.fractal-flight.json": (
        "0.370624233423",
        "-0.670428331878",
        "1e-10",
    ),
    "ember-island.fractal-flight.json": (
        "-1.767894257415",
        "0.00435598854373",
        "1e-9",
    ),
    "elephant-cathedral.fractal-flight.json": (
        "0.289360239955357142873046875",
        "0.0126053470284598214184326171875",
        "7.095791044689357280731201171875e-11",
    ),
    "twin-spiral-nocturne.fractal-flight.json": (
        "-0.748176",
        "0.0993450972222222222175",
        "8.47710503472222900390625e-9",
    ),
    "northern-storm-voyage.fractal-flight.json": (
        "-0.152517553293360397217700532637536525726318359375",
        "1.0389131006397446611108897830359637737274169921875",
        "1e-12",
    ),
}
LONG_EXAMPLES = {
    "elephant-cathedral.fractal-flight.json": ("210", 18, 12),
    "twin-spiral-nocturne.fractal-flight.json": ("260", 20, 15),
    "northern-storm-voyage.fractal-flight.json": ("298", 24, 18),
}


@pytest.mark.parametrize("filename", EXPECTED)
def test_example_flight_plan_is_long_deep_and_deterministic(filename: str) -> None:
    path = EXAMPLES / filename
    document = load_flight_plan(path)
    final = document.path.keyframes[-1].camera
    expected_x, expected_y, expected_width = EXPECTED[filename]

    assert document.scene.fractal is FractalKind.MANDELBROT
    assert mp.mpf(document.duration_text) >= 60
    assert len(document.path.keyframes) >= 10
    assert document.render_track is not None
    assert len(document.render_track.cues) >= 6
    assert final.center_x_text == expected_x
    assert final.center_y_text == expected_y
    assert mp.mpf(final.view_width_text) == mp.mpf(expected_width)
    assert mp.mpf(final.view_width_text) <= mp.mpf("1e-8")
    assert document.source_schema_version == 2
    migrated = deserialize_flight_plan(serialize_flight_plan(document))
    assert migrated.source_schema_version == FLIGHT_PLAN_SCHEMA_VERSION
    assert migrated == document


@pytest.mark.parametrize("filename", EXPECTED)
def test_example_flight_plan_evaluates_complete_timeline(filename: str) -> None:
    document = load_flight_plan(EXAMPLES / filename)
    duration = mp.mpf(document.duration_text)

    samples = tuple(document.evaluate(duration * index / 12) for index in range(13))

    assert samples[0].camera == document.path.keyframes[0].camera
    end = document.path.keyframes[-1].camera
    assert samples[-1].camera.center_x_text == end.center_x_text
    assert samples[-1].camera.center_y_text == end.center_y_text
    assert mp.mpf(samples[-1].camera.view_width_text) == mp.mpf(end.view_width_text)
    assert all(sample.render.max_iterations >= 1 for sample in samples)
    assert max(sample.render.reference_bits for sample in samples) >= 512
    assert len({sample.render.palette.description for sample in samples}) >= 2


@pytest.mark.parametrize("filename,expectation", LONG_EXAMPLES.items())
def test_extended_example_is_between_three_and_five_minutes(
    filename: str,
    expectation: tuple[str, int, int],
) -> None:
    expected_duration, minimum_keyframes, minimum_cues = expectation
    document = load_flight_plan(EXAMPLES / filename)

    assert document.duration_text == expected_duration
    assert mp.mpf("180") <= mp.mpf(document.duration_text) <= mp.mpf("300")
    assert len(document.path.keyframes) >= minimum_keyframes
    assert document.render_track is not None
    assert len(document.render_track.cues) >= minimum_cues


def test_example_flight_plans_end_at_distinct_targets() -> None:
    endpoints = {
        (
            load_flight_plan(EXAMPLES / filename).path.keyframes[-1].camera.center_x_text,
            load_flight_plan(EXAMPLES / filename).path.keyframes[-1].camera.center_y_text,
        )
        for filename in EXPECTED
    }

    assert len(endpoints) == len(EXPECTED)
