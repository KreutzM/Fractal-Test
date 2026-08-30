import numpy as np

from fractal_flight_studio.models import FractalKind, RenderRequest
from fractal_flight_studio.palettes import tone_mapped_colorize
from fractal_flight_studio.renderers.cpu import CpuRenderer
from fractal_flight_studio.tonemapping import apply_tone_mapping, resolve_tone_state


def test_auto_tone_mapping_expands_narrow_value_band():
    values = np.linspace(0.4200, 0.4205, 64, dtype=np.float32).reshape(8, 8)
    inside = np.zeros_like(values, dtype=np.bool_)
    mapped, state, details = apply_tone_mapping(values, inside, mode="auto")
    assert state is not None
    assert details["tone_mapping"] == "auto"
    assert mapped.min() <= 0.05
    assert mapped.max() >= 0.95


def test_auto_tone_mapping_ignores_sparse_outliers():
    central = np.linspace(0.20, 0.30, 1000, dtype=np.float32)
    values = np.concatenate((np.array([0.0, 1.0], dtype=np.float32), central)).reshape(1, -1)
    inside = np.zeros_like(values, dtype=np.bool_)
    _mapped, state, _details = apply_tone_mapping(values, inside, mode="auto")
    assert state is not None
    assert state.low > 0.15
    assert state.high < 0.35


def test_auto_tone_mapping_smooths_parameter_changes():
    values_a = np.linspace(0.10, 0.40, 64, dtype=np.float32).reshape(8, 8)
    values_b = np.linspace(0.25, 0.55, 64, dtype=np.float32).reshape(8, 8)
    inside = np.zeros_like(values_a, dtype=np.bool_)
    _, state_a, _ = apply_tone_mapping(values_a, inside, mode="auto", scene_key=("demo",), smoothing=0.1)
    assert state_a is not None
    _, state_b, details = apply_tone_mapping(
        values_b, inside, mode="auto", state=state_a, scene_key=("demo",), smoothing=0.1
    )
    assert state_b is not None
    target_state, _ = resolve_tone_state(values_b.reshape(-1), "auto", scene_key=("fresh",))
    assert target_state is not None
    assert state_a.low < state_b.low < target_state.low
    assert details["tone_scene_reset"] is False


def test_scene_key_change_resets_automatic_exposure():
    values = np.linspace(0.2, 0.4, 64, dtype=np.float32).reshape(8, 8)
    inside = np.zeros_like(values, dtype=np.bool_)
    _, state_a, _ = apply_tone_mapping(values, inside, mode="auto", scene_key=("a",))
    _, state_b, details = apply_tone_mapping(
        values + 0.2, inside, mode="auto", state=state_a, scene_key=("b",)
    )
    assert state_b is not None
    assert details["tone_scene_reset"] is True


def test_tone_mapped_colorize_keeps_inside_black():
    values = np.array([[0.2, 0.21], [0.22, 0.23]], dtype=np.float32)
    inside = np.array([[True, False], [False, False]])
    rgb, _state, details = tone_mapped_colorize(values, inside, palette="inferno", tone_mapping="auto")
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[0, 0], [0, 0, 0])
    assert details["tone_mapping"] == "auto"


def test_newton_auto_preserves_root_encoding_with_linear_mapping():
    request = RenderRequest(width=48, height=32, max_iterations=40, fractal=FractalKind.NEWTON)
    renderer = CpuRenderer()
    automatic = renderer.render_frame(request, tone_mapping="auto")
    linear = renderer.render_frame(request, tone_mapping="linear")
    assert np.array_equal(automatic.rgb, linear.rgb)
    assert automatic.details["tone_mapping_requested"] == "auto"
    assert automatic.details["tone_mapping"] == "linear"


def test_large_frame_uses_bounded_stratified_sample():
    values = np.linspace(0.1, 0.9, 640 * 400, dtype=np.float32).reshape(400, 640)
    inside = np.zeros_like(values, dtype=np.bool_)
    _mapped, _state, details = apply_tone_mapping(values, inside, mode="auto")
    assert 1000 <= details["tone_sample_count"] <= 4096
