import numpy as np

from fractal_flight_studio.palettes import colorize, palette_names


def test_all_palettes_generate_rgb_and_black_inside():
    values = np.array([[0.0, 0.5], [0.99, 0.25]], dtype=np.float32)
    inside = np.array([[True, False], [False, False]])
    for name in palette_names():
        rgb = colorize(values, inside, name)
        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8
        assert np.array_equal(rgb[0, 0], [0, 0, 0])
