from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Iterable

import numba
import numpy as np

from fractal_flight_studio.deep_zoom import should_use_perturbation
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest, Viewport
from fractal_flight_studio.palettes import tone_mapped_colorize
from fractal_flight_studio.renderers.cuda_double_single_renderer import CudaRenderer
from fractal_flight_studio.surface_lighting import (
    SurfaceLightingSettings,
    apply_surface_lighting,
)


@dataclass(frozen=True, slots=True)
class ValidationCase:
    id: str
    request: RenderRequest
    expected_arithmetic: str
    expected_render_mode: str


def _cases(width: int, height: int) -> tuple[ValidationCase, ...]:
    return (
        ValidationCase(
            "direct-fp32",
            RenderRequest(
                width=width,
                height=height,
                max_iterations=320,
                precision=Precision.FLOAT32,
                render_mode=RenderMode.DIRECT,
            ),
            "float32",
            "direct",
        ),
        ValidationCase(
            "direct-double-single",
            RenderRequest(
                width=width,
                height=height,
                max_iterations=800,
                precision=Precision.FLOAT64,
                render_mode=RenderMode.AUTO,
                center_x_text="-0.743643887037",
                center_y_text="0.131825904205",
                view_width_text="0.0001",
                viewport=Viewport(-0.743643887037, 0.131825904205, 0.0001),
            ),
            "double-single",
            "direct",
        ),
        ValidationCase(
            "deep-double-single-perturbation",
            RenderRequest(
                width=width,
                height=height,
                max_iterations=1200,
                precision=Precision.FLOAT64,
                render_mode=RenderMode.AUTO,
                reference_bits=384,
                center_x_text="-0.743643887037151",
                center_y_text="0.13182590420533",
                view_width_text="1e-13",
                viewport=Viewport(
                    -0.743643887037151,
                    0.13182590420533,
                    1e-13,
                ),
            ),
            "double-single",
            "perturbation",
        ),
    )


def _auto_relief_request(width: int, height: int) -> RenderRequest:
    """Representative view where auto tone mapping must produce visible relief."""

    return RenderRequest(
        width=width,
        height=height,
        max_iterations=1200,
        precision=Precision.FLOAT64,
        render_mode=RenderMode.AUTO,
        reference_bits=256,
        center_x_text="-0.74364386269",
        center_y_text="0.13182590271",
        view_width_text="0.00000013526",
        viewport=Viewport(-0.74364386269, 0.13182590271, 0.00000013526),
    )


def _median_frame_seconds(
    renderer: CudaRenderer,
    request: RenderRequest,
    settings: SurfaceLightingSettings | None,
    repeats: int,
) -> float:
    for _ in range(2):
        renderer.render_frame(
            request,
            tone_mapping="linear",
            surface_lighting=settings,
        )
    samples = []
    for _ in range(repeats):
        result = renderer.render_frame(
            request,
            tone_mapping="linear",
            surface_lighting=settings,
        )
        samples.append(float(result.elapsed_seconds))
    return statistics.median(samples)


def validate_case(
    renderer: CudaRenderer,
    case: ValidationCase,
    settings: SurfaceLightingSettings,
    repeats: int,
) -> dict[str, object]:
    planned_render_mode = (
        "perturbation" if should_use_perturbation(case.request) else "direct"
    )
    if planned_render_mode != case.expected_render_mode:
        raise RuntimeError(
            f"validation case {case.id!r} plans {planned_render_mode}, "
            f"expected {case.expected_render_mode}; adjust its view or dimensions"
        )

    rendered = renderer.render(case.request)
    colored, _tone_state, _tone_details = tone_mapped_colorize(
        rendered.values,
        rendered.inside,
        palette="inferno",
        tone_mapping="linear",
    )
    expected = apply_surface_lighting(
        rendered.values,
        rendered.inside,
        colored,
        settings,
    )
    actual = renderer.render_frame(
        case.request,
        palette="inferno",
        tone_mapping="linear",
        surface_lighting=settings,
    )
    baseline = renderer.render_frame(case.request, tone_mapping="linear")
    disabled = renderer.render_frame(
        case.request,
        tone_mapping="linear",
        surface_lighting=SurfaceLightingSettings(enabled=False),
    )

    delta = np.abs(actual.rgb.astype(np.int16) - expected.astype(np.int16))
    maximum_delta = int(delta.max(initial=0))
    mismatch_fraction = float(np.count_nonzero(delta) / delta.size)
    disabled_matches = bool(np.array_equal(disabled.rgb, baseline.rgb))
    optimized = bool(actual.details.get("optimized_frame_path"))
    transfer = str(actual.details.get("transfer", ""))
    arithmetic = str(actual.details.get("arithmetic", ""))
    render_mode = str(actual.details.get("render_mode", ""))
    route_matches = (
        arithmetic == case.expected_arithmetic
        and render_mode == case.expected_render_mode
    )

    unlit_seconds = _median_frame_seconds(
        renderer, case.request, None, repeats
    )
    lit_seconds = _median_frame_seconds(
        renderer, case.request, settings, repeats
    )

    passed = (
        maximum_delta <= 1
        and disabled_matches
        and optimized
        and transfer == "single RGB readback"
        and route_matches
    )
    return {
        "id": case.id,
        "passed": passed,
        "maximum_channel_delta": maximum_delta,
        "channel_mismatch_fraction": mismatch_fraction,
        "disabled_matches_baseline": disabled_matches,
        "optimized_frame_path": optimized,
        "transfer": transfer,
        "render_mode": render_mode,
        "planned_render_mode": planned_render_mode,
        "arithmetic": arithmetic,
        "expected_render_mode": case.expected_render_mode,
        "expected_arithmetic": case.expected_arithmetic,
        "route_matches": route_matches,
        "median_unlit_seconds": unlit_seconds,
        "median_lit_seconds": lit_seconds,
        "lighting_overhead_seconds": lit_seconds - unlit_seconds,
        "lighting_time_ratio": lit_seconds / unlit_seconds if unlit_seconds else None,
    }


def validate_auto_relief(
    renderer: CudaRenderer,
    width: int,
    height: int,
    settings: SurfaceLightingSettings,
) -> dict[str, object]:
    request = _auto_relief_request(width, height)
    planned_render_mode = "perturbation" if should_use_perturbation(request) else "direct"
    if planned_render_mode != "direct":
        raise RuntimeError(
            "auto-relief validation must remain on the direct Double-Single route"
        )

    rendered = renderer.render(request)
    scene_key = ("surface-lighting-auto-relief", width, height)
    colored, tone_state, _tone_details = tone_mapped_colorize(
        rendered.values,
        rendered.inside,
        palette="inferno",
        tone_mapping="auto",
        scene_key=scene_key,
        tone_smoothing=1.0,
    )
    if tone_state is None:
        raise RuntimeError("auto-relief validation did not produce a tone state")

    expected = apply_surface_lighting(
        rendered.values,
        rendered.inside,
        colored,
        settings,
        tone_state=tone_state,
    )
    frame_kwargs = {
        "palette": "inferno",
        "tone_mapping": "auto",
        "tone_state": tone_state,
        "tone_scene_key": scene_key,
        "tone_state_locked": True,
    }
    baseline = renderer.render_frame(request, **frame_kwargs)
    actual = renderer.render_frame(
        request,
        surface_lighting=settings,
        **frame_kwargs,
    )
    opposite = renderer.render_frame(
        request,
        surface_lighting=replace(
            settings,
            azimuth_degrees=(settings.azimuth_degrees + 180.0) % 360.0,
        ),
        **frame_kwargs,
    )

    parity_delta = np.abs(actual.rgb.astype(np.int16) - expected.astype(np.int16))
    maximum_delta = int(parity_delta.max(initial=0))
    visible = np.any(baseline.rgb != 0, axis=2)
    visible_count = int(np.count_nonzero(visible))
    if visible_count == 0:
        raise RuntimeError("auto-relief validation frame contains no visible pixels")

    relief_delta = np.abs(
        actual.rgb.astype(np.int16) - baseline.rgb.astype(np.int16)
    )[visible]
    direction_delta = np.abs(
        actual.rgb.astype(np.int16) - opposite.rgb.astype(np.int16)
    )[visible]
    changed_fraction = float(
        np.mean(np.any(actual.rgb != baseline.rgb, axis=2)[visible])
    )
    mean_relief_delta = float(relief_delta.mean())
    mean_direction_delta = float(direction_delta.mean())
    baseline_mean = float(baseline.rgb[visible].mean())
    exposure_ratio = (
        float(actual.rgb[visible].mean()) / baseline_mean if baseline_mean else 1.0
    )
    optimized = bool(actual.details.get("optimized_frame_path"))
    transfer = str(actual.details.get("transfer", ""))
    arithmetic = str(actual.details.get("arithmetic", ""))
    render_mode = str(actual.details.get("render_mode", ""))

    passed = (
        maximum_delta <= 1
        and mean_relief_delta >= 3.0
        and mean_direction_delta >= 4.0
        and changed_fraction >= 0.25
        and 0.60 <= exposure_ratio <= 1.40
        and optimized
        and transfer == "single RGB readback"
        and arithmetic == "double-single"
        and render_mode == "direct"
    )
    return {
        "id": "auto-tone-visible-relief",
        "passed": passed,
        "maximum_channel_delta": maximum_delta,
        "visible_pixel_count": visible_count,
        "mean_relief_channel_delta": mean_relief_delta,
        "mean_direction_channel_delta": mean_direction_delta,
        "changed_visible_pixel_fraction": changed_fraction,
        "lit_to_unlit_mean_ratio": exposure_ratio,
        "optimized_frame_path": optimized,
        "transfer": transfer,
        "render_mode": render_mode,
        "planned_render_mode": planned_render_mode,
        "arithmetic": arithmetic,
        "height_source": actual.details.get("surface_lighting_height_source"),
        "slope_scale": actual.details.get("surface_lighting_slope_scale"),
        "flat_neutral": actual.details.get("surface_lighting_flat_neutral"),
    }


def run(
    cases: Iterable[ValidationCase],
    settings: SurfaceLightingSettings,
    repeats: int,
) -> dict[str, object]:
    renderer = CudaRenderer()
    if not renderer.is_available():
        raise RuntimeError("CUDA is not available")
    planned_cases = tuple(cases)
    if not planned_cases:
        raise ValueError("at least one validation case is required")
    records = [
        validate_case(renderer, case, settings, repeats) for case in planned_cases
    ]
    first_request = planned_cases[0].request
    auto_relief = validate_auto_relief(
        renderer,
        first_request.width,
        first_request.height,
        settings,
    )
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numba": numba.__version__,
            "device": renderer._device_name(),
        },
        "settings": asdict(settings),
        "cases": records,
        "auto_relief": auto_relief,
        "passed": (
            all(bool(record["passed"]) for record in records)
            and bool(auto_relief["passed"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate GPU-native surface lighting on a physical CUDA device."
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("dimensions must be positive")
    if args.repeats < 1:
        parser.error("repeats must be at least one")

    settings = SurfaceLightingSettings(
        enabled=True,
        strength=2.0,
        azimuth_degrees=315.0,
        elevation_degrees=45.0,
        ambient=0.35,
        diffuse=0.65,
    )
    started = time.perf_counter()
    report = run(_cases(args.width, args.height), settings, args.repeats)
    report["elapsed_seconds"] = time.perf_counter() - started
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
