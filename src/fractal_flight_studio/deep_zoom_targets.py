from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import json
import re
from typing import Any, Mapping
from urllib.parse import urlparse

import mpmath as mp

from .models import FractalKind
from .palettes import palette_names

_CATALOG_RESOURCE = "data/deep_zoom_targets.json"
_TARGET_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True, slots=True)
class DeepZoomTarget:
    id: str
    name: str
    description: str
    fractal: FractalKind
    center_x_text: str
    center_y_text: str
    view_width_text: str
    recommended_iterations: int
    reference_bits: int
    palette: str
    tags: tuple[str, ...]
    favorite: bool
    source_url: str

    @property
    def recommendation_text(self) -> str:
        width = mp.nstr(mp.mpf(self.view_width_text), 6)
        return (
            f"{self.description}\n"
            f"Breite {width}; {self.recommended_iterations} Iterationen; "
            f"{self.reference_bits} Bit; Palette {self.palette}."
        )


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"deep-zoom target field {key!r} must be a non-empty string")
    return value.strip()


def _parse_target(record: Mapping[str, Any]) -> DeepZoomTarget:
    target_id = _required_text(record, "id")
    if not _TARGET_ID_PATTERN.fullmatch(target_id):
        raise ValueError(f"invalid deep-zoom target id: {target_id!r}")

    center_x_text = _required_text(record, "center_x")
    center_y_text = _required_text(record, "center_y")
    view_width_text = _required_text(record, "view_width")
    try:
        center_x = mp.mpf(center_x_text)
        center_y = mp.mpf(center_y_text)
        view_width = mp.mpf(view_width_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"target {target_id!r} contains invalid coordinates") from exc
    if not (mp.isfinite(center_x) and mp.isfinite(center_y) and mp.isfinite(view_width)):
        raise ValueError(f"target {target_id!r} coordinates must be finite")
    if view_width <= 0:
        raise ValueError(f"target {target_id!r} view width must be positive")

    iterations = record.get("recommended_iterations")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 20 <= iterations <= 100_000:
        raise ValueError(f"target {target_id!r} has invalid recommended_iterations")

    reference_bits = record.get("reference_bits")
    if not isinstance(reference_bits, int) or isinstance(reference_bits, bool) or not 64 <= reference_bits <= 16_384:
        raise ValueError(f"target {target_id!r} has invalid reference_bits")

    palette = _required_text(record, "palette")
    if palette not in palette_names():
        raise ValueError(f"target {target_id!r} references unknown palette {palette!r}")

    raw_tags = record.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags or not all(isinstance(tag, str) and tag.strip() for tag in raw_tags):
        raise ValueError(f"target {target_id!r} tags must be a non-empty string list")

    favorite = record.get("favorite")
    if not isinstance(favorite, bool):
        raise ValueError(f"target {target_id!r} favorite must be boolean")

    source_url = _required_text(record, "source_url")
    parsed_url = urlparse(source_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError(f"target {target_id!r} source_url must be an HTTPS URL")

    try:
        fractal = FractalKind(_required_text(record, "fractal"))
    except ValueError as exc:
        raise ValueError(f"target {target_id!r} references an unknown fractal") from exc

    return DeepZoomTarget(
        id=target_id,
        name=_required_text(record, "name"),
        description=_required_text(record, "description"),
        fractal=fractal,
        center_x_text=center_x_text,
        center_y_text=center_y_text,
        view_width_text=view_width_text,
        recommended_iterations=iterations,
        reference_bits=reference_bits,
        palette=palette,
        tags=tuple(tag.strip() for tag in raw_tags),
        favorite=favorite,
        source_url=source_url,
    )


def parse_deep_zoom_catalog(payload: Mapping[str, Any]) -> tuple[DeepZoomTarget, ...]:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported deep-zoom target catalog schema")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("deep-zoom target catalog must contain targets")

    targets: list[DeepZoomTarget] = []
    ids: set[str] = set()
    names: set[str] = set()
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping):
            raise ValueError("deep-zoom target entries must be objects")
        target = _parse_target(raw_target)
        normalized_name = target.name.casefold()
        if target.id in ids:
            raise ValueError(f"duplicate deep-zoom target id: {target.id}")
        if normalized_name in names:
            raise ValueError(f"duplicate deep-zoom target name: {target.name}")
        ids.add(target.id)
        names.add(normalized_name)
        targets.append(target)
    return tuple(targets)


@lru_cache(maxsize=1)
def load_deep_zoom_targets() -> tuple[DeepZoomTarget, ...]:
    resource = resources.files("fractal_flight_studio").joinpath(_CATALOG_RESOURCE)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("deep-zoom target catalog root must be an object")
    return parse_deep_zoom_catalog(payload)


def favorite_deep_zoom_targets() -> tuple[DeepZoomTarget, ...]:
    return tuple(target for target in load_deep_zoom_targets() if target.favorite)


def deep_zoom_target(target_id: str) -> DeepZoomTarget:
    for target in load_deep_zoom_targets():
        if target.id == target_id:
            return target
    raise KeyError(f"unknown deep-zoom target: {target_id}")
