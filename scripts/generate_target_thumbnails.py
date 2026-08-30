"""Generate the compressed preview atlas for the curated target browser."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import textwrap
import zlib

from PIL import Image

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.deep_zoom import digits_for_bits
from fractal_flight_studio.deep_zoom_targets import load_deep_zoom_targets
from fractal_flight_studio.models import Precision, RenderMode, RenderRequest
from fractal_flight_studio.renderers import select_renderer

THUMBNAIL_WIDTH = 48
THUMBNAIL_HEIGHT = 30
_XPM_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%&()*+,-./:;<=>?@[]^_`{|}~"
_XPM_COLORS = 32


def thumbnail_request(target, *, width: int, height: int) -> RenderRequest:
    digits = digits_for_bits(target.reference_bits)
    camera = CameraState(target.center_x_text, target.center_y_text, target.view_width_text)
    return RenderRequest(
        width=width,
        height=height,
        viewport=camera.proxy_viewport(digits=digits),
        fractal=target.fractal,
        max_iterations=target.recommended_iterations,
        precision=Precision.FLOAT32,
        render_mode=RenderMode.AUTO,
        reference_bits=target.reference_bits,
        center_x_text=target.center_x_text,
        center_y_text=target.center_y_text,
        view_width_text=target.view_width_text,
    )


def _xpm_text(image: Image.Image, *, variable_name: str) -> str:
    quantized = image.quantize(colors=_XPM_COLORS, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    used_indices = sorted(index for _count, index in quantized.getcolors(maxcolors=256) or ())
    if len(used_indices) > len(_XPM_SYMBOLS):
        raise ValueError("XPM preview uses more colours than the one-character palette supports")
    symbols = {index: _XPM_SYMBOLS[position] for position, index in enumerate(used_indices)}
    pixels = quantized.load()
    lines = [
        "/* XPM */",
        f"static char * {variable_name}[] = {{",
        f'"{quantized.width} {quantized.height} {len(used_indices)} 1",',
    ]
    for index in used_indices:
        red, green, blue = palette[index * 3 : index * 3 + 3]
        lines.append(f'"{symbols[index]} c #{red:02X}{green:02X}{blue:02X}",')
    for y in range(quantized.height):
        row = "".join(symbols[pixels[x, y]] for x in range(quantized.width))
        suffix = "," if y < quantized.height - 1 else ""
        lines.append(f'"{row}"{suffix}')
    lines.append("};")
    return "\n".join(lines) + "\n"


def _atlas_payload(atlas: dict[str, str]) -> str:
    raw = json.dumps(atlas, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")


def _chunk_module_text(chunk: str, index: int) -> str:
    lines = "\n".join(f"    {part!r}" for part in textwrap.wrap(chunk, 100))
    return (
        f'"""Generated preview-atlas chunk {index}."""\n\n'
        "CHUNK = (\n"
        f"{lines}\n"
        ")\n"
    )


def _module_text(chunk_count: int) -> str:
    imports = "\n".join(
        f"from .target_thumbnail_chunk_{index} import CHUNK as _CHUNK_{index}"
        for index in range(chunk_count)
    )
    joined = " + ".join(f"_CHUNK_{index}" for index in range(chunk_count))
    return f'''"""Generated compressed preview atlas for curated deep-zoom targets."""\n\nfrom __future__ import annotations\n\nimport base64\nfrom functools import lru_cache\nimport json\nimport zlib\n\n{imports}\n\nTHUMBNAIL_WIDTH = {THUMBNAIL_WIDTH}\nTHUMBNAIL_HEIGHT = {THUMBNAIL_HEIGHT}\n\n_ATLAS_B85 = {joined}\n\n\n@lru_cache(maxsize=1)\ndef _atlas() -> dict[str, str]:\n    raw = zlib.decompress(base64.b85decode(_ATLAS_B85.encode("ascii")))\n    return json.loads(raw.decode("ascii"))\n\n\ndef thumbnail_bytes(target_id: str) -> bytes:\n    try:\n        return _atlas()[target_id].encode("ascii")\n    except KeyError as error:\n        raise KeyError(f"Unknown target thumbnail: {{target_id}}") from error\n\n\ndef thumbnail_ids() -> tuple[str, ...]:\n    return tuple(_atlas())\n'''


def generate_thumbnail_atlas(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = select_renderer("cpu")
    atlas: dict[str, str] = {}
    for target in load_deep_zoom_targets():
        request = thumbnail_request(target, width=THUMBNAIL_WIDTH, height=THUMBNAIL_HEIGHT)
        frame = renderer.render_frame(
            request,
            target.palette,
            1.0,
            0.0,
            tone_mapping="auto",
            tone_scene_key=("target-thumbnail", target.id),
            tone_smoothing=1.0,
        )
        atlas[target.id] = _xpm_text(
            Image.fromarray(frame.rgb, mode="RGB"),
            variable_name=target.id.replace("-", "_"),
        )
        print(target.id)
    encoded = _atlas_payload(atlas)
    chunk_size = 3500
    chunks = [encoded[offset : offset + chunk_size] for offset in range(0, len(encoded), chunk_size)]
    for stale in output_path.parent.glob("target_thumbnail_chunk_*.py"):
        stale.unlink()
    for index, chunk in enumerate(chunks):
        chunk_path = output_path.with_name(f"target_thumbnail_chunk_{index}.py")
        chunk_path.write_text(_chunk_module_text(chunk, index), encoding="ascii")
    output_path.write_text(_module_text(len(chunks)), encoding="ascii")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/fractal_flight_studio/target_thumbnail_data.py"),
    )
    args = parser.parse_args()
    generate_thumbnail_atlas(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
