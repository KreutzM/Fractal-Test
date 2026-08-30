from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import mpmath as mp

from .animation import FlightPath
from .deep_zoom import digits_for_bits
from .models import FractalKind, Precision, RenderMode, RenderRequest, Viewport
from .palettes import palette_names
from .tonemapping import tone_mapping_names
from .service import save_png


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fractal", choices=[x.value for x in FractalKind], default="mandelbrot")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--center-x", type=str, default="-0.5")
    parser.add_argument("--center-y", type=str, default="0.0")
    parser.add_argument("--view-width", type=str, default="3.5")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--precision", choices=[x.value for x in Precision], default="float32")
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--palette", choices=palette_names(), default="inferno")
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--tone-mapping", choices=tone_mapping_names(), default="auto")
    parser.add_argument("--julia-real", type=float, default=-0.8)
    parser.add_argument("--julia-imag", type=float, default=0.156)
    parser.add_argument("--exponent", type=int, default=3)
    parser.add_argument("--render-mode", choices=[x.value for x in RenderMode], default="auto")
    parser.add_argument("--reference-bits", type=int, default=256)


def _proxy_width(view_width_text: str) -> float:
    value = mp.mpf(view_width_text)
    f = float(value)
    if f == 0.0 and value > 0:
        return 1e-300
    return f


def _request_from_args(args: argparse.Namespace) -> RenderRequest:
    return RenderRequest(
        width=args.width,
        height=args.height,
        viewport=Viewport(float(mp.mpf(args.center_x)), float(mp.mpf(args.center_y)), _proxy_width(args.view_width)),
        fractal=FractalKind(args.fractal),
        max_iterations=args.iterations,
        julia_c_real=args.julia_real,
        julia_c_imag=args.julia_imag,
        exponent=args.exponent,
        precision=Precision(args.precision),
        render_mode=RenderMode(args.render_mode),
        reference_bits=args.reference_bits,
        center_x_text=args.center_x,
        center_y_text=args.center_y,
        view_width_text=args.view_width,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal-render")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render one PNG")
    _common_parser(render)
    render.add_argument("--output", type=Path, required=True)

    flight = subparsers.add_parser("flight", help="render a logarithmic zoom frame sequence")
    _common_parser(flight)
    flight.add_argument("--target-x", type=str, required=True)
    flight.add_argument("--target-y", type=str, required=True)
    flight.add_argument("--target-width", type=str, required=True)
    flight.add_argument("--frames", type=int, default=120)
    flight.add_argument("--output-dir", type=Path, required=True)

    benchmark = subparsers.add_parser("benchmark", help="run a repeatable render benchmark")
    _common_parser(benchmark)
    benchmark.set_defaults(width=1280, height=720, iterations=500)
    return parser


def _build_float_viewport(x_text: str, y_text: str, width_text: str) -> Viewport:
    return Viewport(float(mp.mpf(x_text)), float(mp.mpf(y_text)), _proxy_width(width_text))


def _interpolate_view_text(start: RenderRequest, target_x: str, target_y: str, target_width: str, t: float) -> tuple[str, str, str]:
    if t <= 0.0:
        return start.center_x_text or repr(start.viewport.center_x), start.center_y_text or repr(start.viewport.center_y), start.view_width_text or repr(start.viewport.width)
    if t >= 1.0:
        return target_x, target_y, target_width
    bits = max(start.reference_bits, 256)
    dps = digits_for_bits(bits)
    with mp.workdps(dps):
        eased = mp.mpf(t) * mp.mpf(t) * (mp.mpf("3.0") - mp.mpf("2.0") * mp.mpf(t))
        sx = mp.mpf(start.center_x_text or repr(start.viewport.center_x))
        sy = mp.mpf(start.center_y_text or repr(start.viewport.center_y))
        sw = mp.mpf(start.view_width_text or repr(start.viewport.width))
        ex = mp.mpf(target_x)
        ey = mp.mpf(target_y)
        ew = mp.mpf(target_width)
        log_width = mp.log(sw) * (1 - eased) + mp.log(ew) * eased
        width = mp.e ** log_width
        cx = sx * (1 - eased) + ex * eased
        cy = sy * (1 - eased) + ey * eased
        digits = digits_for_bits(bits)
        return mp.nstr(cx, digits), mp.nstr(cy, digits), mp.nstr(width, digits)



def _tone_scene_key(request: RenderRequest, tone_mapping: str) -> tuple[str, ...]:
    return (
        request.fractal.value,
        request.precision.value,
        request.render_mode.value,
        str(request.reference_bits),
        tone_mapping,
        str(request.max_iterations),
        str(request.exponent),
        f"{request.julia_c_real:.12g}",
        f"{request.julia_c_imag:.12g}",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = _request_from_args(args)

    if args.command == "render":
        result = save_png(request, args.output, args.backend, args.palette, args.cycles, args.phase, args.tone_mapping)
        print(f"saved {args.output} via {result.backend} in {result.elapsed_seconds:.3f}s")
        return 0

    if args.command == "flight":
        if args.frames < 2:
            raise SystemExit("--frames must be at least 2")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        total = 0.0
        tone_state = None
        tone_scene_key = _tone_scene_key(request, args.tone_mapping)
        for index in range(args.frames):
            cx_text, cy_text, width_text = _interpolate_view_text(
                request,
                args.target_x,
                args.target_y,
                args.target_width,
                index / (args.frames - 1),
            )
            frame_request = replace(
                request,
                viewport=_build_float_viewport(cx_text, cy_text, width_text),
                center_x_text=cx_text,
                center_y_text=cy_text,
                view_width_text=width_text,
            )
            output = args.output_dir / f"frame_{index:05d}.png"
            result = save_png(
                frame_request,
                output,
                args.backend,
                args.palette,
                args.cycles,
                args.phase + index / args.frames,
                args.tone_mapping,
                tone_state,
                tone_scene_key,
                0.08,
            )
            tone_state = result.details.get("tone_state")
            total += result.elapsed_seconds
            print(f"{index + 1}/{args.frames}: {result.elapsed_seconds:.3f}s", end="\r")
        print(f"\nrendered {args.frames} frames in {total:.2f}s kernel time")
        return 0

    if args.command == "benchmark":
        warmup = replace(request, width=32, height=32, max_iterations=16)
        save_png(warmup, Path("benchmark_warmup.png"), args.backend, args.palette, tone_mapping=args.tone_mapping)
        Path("benchmark_warmup.png").unlink(missing_ok=True)
        result = save_png(request, Path("benchmark.png"), args.backend, args.palette, tone_mapping=args.tone_mapping)
        megapixels = request.width * request.height / 1_000_000
        print(
            f"backend={result.backend} precision={request.precision.value} "
            f"size={request.width}x{request.height} iterations={request.max_iterations} "
            f"mode={request.render_mode.value} refbits={request.reference_bits} tone={args.tone_mapping} "
            f"time={result.elapsed_seconds:.3f}s throughput={megapixels/result.elapsed_seconds:.2f} MPix/s"
        )
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
