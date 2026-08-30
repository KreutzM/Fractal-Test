from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class FrameVisualQuality:
    dominant_color_fraction: float
    quantized_color_count: int
    edge_fraction: float
    repeated_row_fraction: float
    repeated_column_fraction: float
    safe: bool
    reason: str | None = None


def _sample_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.intp)
    return np.linspace(0, length - 1, maximum, dtype=np.intp)


def _quantize_rgb(rgb: np.ndarray) -> np.ndarray:
    # Five bits per channel suppress tiny tone-mapping differences without
    # hiding visible blocks or large colour fields.
    return np.right_shift(rgb.astype(np.uint8, copy=False), 3)


def analyze_frame_visual_quality(rgb: np.ndarray) -> FrameVisualQuality:
    """Classify block repetition and colour-field collapse before GUI display."""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("expected a non-empty RGB image")

    height, width, _ = rgb.shape

    sample_y = _sample_indices(height, 128)
    sample_x = _sample_indices(width, 128)
    sample = _quantize_rgb(rgb[np.ix_(sample_y, sample_x)])
    packed = (
        sample[..., 0].astype(np.uint16) << 10
        | sample[..., 1].astype(np.uint16) << 5
        | sample[..., 2].astype(np.uint16)
    )
    _, counts = np.unique(packed, return_counts=True)
    dominant = float(counts.max() / packed.size)
    color_count = int(counts.size)

    luminance = (
        sample[..., 0].astype(np.int16) * 54
        + sample[..., 1].astype(np.int16) * 183
        + sample[..., 2].astype(np.int16) * 19
    )
    gradients = []
    if sample.shape[1] > 1:
        gradients.append(np.abs(np.diff(luminance, axis=1)))
    if sample.shape[0] > 1:
        gradients.append(np.abs(np.diff(luminance, axis=0)))
    edge_pixels = sum(int(np.count_nonzero(gradient >= 64)) for gradient in gradients)
    edge_total = sum(int(gradient.size) for gradient in gradients)
    edge_fraction = edge_pixels / edge_total if edge_total else 0.0

    # Compare a bounded sample of consecutive source-row and source-column
    # pairs. Numeric pixel collapse creates many exactly repeated lines, while
    # the cost remains independent of the full frame size.
    row_columns = _sample_indices(width, 256)
    column_rows = _sample_indices(height, 256)
    row_starts = _sample_indices(max(0, height - 1), 256)
    column_starts = _sample_indices(max(0, width - 1), 256)
    repeated_rows = (
        np.all(
            _quantize_rgb(rgb[np.ix_(row_starts, row_columns)])
            == _quantize_rgb(rgb[np.ix_(row_starts + 1, row_columns)]),
            axis=(1, 2),
        )
        if height > 1
        else np.zeros(0, dtype=bool)
    )
    repeated_columns = (
        np.all(
            _quantize_rgb(rgb[np.ix_(column_rows, column_starts)])
            == _quantize_rgb(rgb[np.ix_(column_rows, column_starts + 1)]),
            axis=(0, 2),
        )
        if width > 1
        else np.zeros(0, dtype=bool)
    )
    row_fraction = float(np.mean(repeated_rows)) if repeated_rows.size else 0.0
    column_fraction = float(np.mean(repeated_columns)) if repeated_columns.size else 0.0

    reason = None
    if dominant >= 0.985:
        reason = "nahezu einfarbige Fläche"
    elif color_count <= 8 and edge_fraction < 0.001:
        reason = "keine verwertbare Farb- oder Kantenstruktur"
    elif row_fraction >= 0.25 and column_fraction >= 0.25:
        reason = "wiederholtes zweidimensionales Pixelraster"
    elif max(row_fraction, column_fraction) >= 0.60:
        reason = "wiederholte Pixelstreifen"

    return FrameVisualQuality(
        dominant_color_fraction=dominant,
        quantized_color_count=color_count,
        edge_fraction=edge_fraction,
        repeated_row_fraction=row_fraction,
        repeated_column_fraction=column_fraction,
        safe=reason is None,
        reason=reason,
    )
