import math

from fractal_flight_studio.animation import FlightPath
from fractal_flight_studio.models import Viewport


def test_flight_path_uses_exact_endpoints_and_log_zoom():
    start = Viewport(-0.5, 0.0, 4.0)
    end = Viewport(-0.75, 0.1, 0.04)
    path = FlightPath(start, end)
    assert path.viewport_at(0.0) == start
    assert path.viewport_at(1.0) == end
    midpoint = path.viewport_at(0.5)
    assert math.isclose(midpoint.width, math.sqrt(start.width * end.width), rel_tol=1e-12)
