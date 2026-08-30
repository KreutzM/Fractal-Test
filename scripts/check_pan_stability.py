from __future__ import annotations

import argparse
import mpmath as mp
import numpy as np

from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.renderers import select_renderer


def _request(args: argparse.Namespace, center_x_text: str) -> RenderRequest:
    return RenderRequest(
        width=args.width,
        height=args.height,
        viewport=Viewport(float(mp.mpf(center_x_text)), float(mp.mpf(args.center_y)), float(mp.mpf(args.view_width))),
        max_iterations=args.iterations,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.PERTURBATION,
        reference_bits=args.reference_bits,
        center_x_text=center_x_text,
        center_y_text=args.center_y,
        view_width_text=args.view_width,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that overlapping perturbation frames remain stable while panning"
    )
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--reference-bits", type=int, default=256)
    parser.add_argument("--center-x", default="-0.1011")
    parser.add_argument("--center-y", default="0.9563")
    parser.add_argument("--view-width", default="1e-8")
    parser.add_argument("--shift-pixels", type=int, default=7)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    if not 1 <= args.shift_pixels < args.width:
        parser.error("--shift-pixels must be between 1 and width-1")

    digits = max(80, int(args.reference_bits * 0.31) + 10)
    with mp.workdps(digits):
        shifted_center = mp.mpf(args.center_x) + (
            mp.mpf(args.shift_pixels) * mp.mpf(args.view_width) / args.width
        )
        shifted_center_text = mp.nstr(shifted_center, digits)

    renderer = select_renderer(args.backend)
    first = renderer.render(_request(args, args.center_x))
    second = renderer.render(_request(args, shifted_center_text))

    first_values = first.values[:, args.shift_pixels :]
    second_values = second.values[:, : args.width - args.shift_pixels]
    first_inside = first.inside[:, args.shift_pixels :]
    second_inside = second.inside[:, : args.width - args.shift_pixels]

    max_difference = float(np.max(np.abs(first_values - second_values)))
    classification_mismatches = int(np.count_nonzero(first_inside != second_inside))
    reference_reused = bool(second.details.get("reference_reused"))

    print(f"backend: {second.backend}")
    print(f"reference reused: {reference_reused}")
    print(f"maximum overlapping value difference: {max_difference:.9g}")
    print(f"inside/outside mismatches: {classification_mismatches}")
    print(f"rebase pixels: {second.details.get('rebase_pixels', 'not read back')}")
    print(f"glitch repairs: {second.details.get('glitch_pixels', 'not read back')}")

    stable = (
        reference_reused
        and max_difference <= args.tolerance
        and classification_mismatches == 0
    )
    print("result: STABLE" if stable else "result: UNSTABLE")
    return 0 if stable else 1


if __name__ == "__main__":
    raise SystemExit(main())
