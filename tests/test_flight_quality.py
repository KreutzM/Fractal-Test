from __future__ import annotations

import numpy as np

from fractal_flight_studio.flight_quality import analyze_frame_visual_quality


def test_uniform_colour_field_is_rejected():
    rgb = np.full((96, 128, 3), (32, 64, 96), dtype=np.uint8)
    quality = analyze_frame_visual_quality(rgb)
    assert quality.safe is False
    assert quality.reason == "nahezu einfarbige Fläche"


def test_repeated_four_pixel_blocks_are_rejected():
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    rgb = np.repeat(np.repeat(source, 4, axis=0), 4, axis=1)
    quality = analyze_frame_visual_quality(rgb)
    assert quality.safe is False
    assert quality.repeated_row_fraction > 0.70
    assert quality.repeated_column_fraction > 0.70


def test_detailed_frame_is_accepted():
    y, x = np.indices((96, 128))
    rgb = np.stack(
        ((x * 11 + y * 3) % 256, (x * 5 + y * 17) % 256, (x * 19 + y * 7) % 256),
        axis=-1,
    ).astype(np.uint8)
    quality = analyze_frame_visual_quality(rgb)
    assert quality.safe is True
    assert quality.dominant_color_fraction < 0.985
