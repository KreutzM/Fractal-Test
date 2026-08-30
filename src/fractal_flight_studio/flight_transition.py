from __future__ import annotations

import mpmath as mp

from ._flight_transition_core import (
    FreeTargetValues,
    TransitionMode,
    TransitionPlan,
    TransitionSettings,
    TransitionTarget,
    end_render_profile,
    merge_render_cues,
    plan_transition,
)
from .camera import CameraState


def suggested_target_width(
    source_camera: CameraState,
    *,
    zoom_factor_text: str = "1",
    digits: int = 80,
) -> str:
    """Return an exact free-target width without hidden zoom by default."""

    with mp.workdps(digits):
        _x, _y, width = source_camera.values(digits=digits)
        try:
            factor = mp.mpf(str(zoom_factor_text).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError("zoom factor must be a decimal number") from exc
        if not mp.isfinite(factor) or factor < 1:
            raise ValueError("zoom factor must be at least one")
        return mp.nstr(width / factor, n=digits, min_fixed=-6, max_fixed=12)


__all__ = [
    "FreeTargetValues",
    "TransitionMode",
    "TransitionPlan",
    "TransitionSettings",
    "TransitionTarget",
    "end_render_profile",
    "merge_render_cues",
    "plan_transition",
    "suggested_target_width",
]
