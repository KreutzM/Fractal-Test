from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .camera import CameraState
from .flight_path import CameraPath, CenterInterpolation, Easing, FlightKeyframe
from .flight_plan import (
    FlightPlanDefaults,
    FlightPlanDocument,
    FlightScene,
    PaletteTransition,
    RenderCue,
    RenderProfile,
    RenderTrack,
)
from .models import FractalKind
from .surface_lighting import SurfaceLightingSettings

FLIGHT_PLAN_FORMAT = "fractal-flight-studio.flight-plan"
LEGACY_FLIGHT_PLAN_FORMAT = "fractal-flight-studio.camera-path"
FLIGHT_PLAN_SCHEMA_VERSION = 3
FLIGHT_PLAN_EXTENSION = ".fractal-flight.json"
MAX_FLIGHT_PLAN_BYTES = 10 * 1024 * 1024
MAX_FLIGHT_PLAN_KEYFRAMES = 100_000
MAX_FLIGHT_PLAN_RENDER_CUES = 100_000
_MAX_DIGITS = 16_384


class FlightPlanError(ValueError):
    """Base class for invalid or unsupported flight-plan documents."""


class FlightPlanFormatError(FlightPlanError):
    """Raised when a flight-plan document violates its JSON schema."""


def flight_plan_to_dict(document: FlightPlanDocument) -> dict[str, Any]:
    """Convert a document to the current versioned, lossless JSON representation."""

    return {
        "format": FLIGHT_PLAN_FORMAT,
        "schema_version": FLIGHT_PLAN_SCHEMA_VERSION,
        "name": document.name,
        "digits": document.path.digits,
        "scene": {
            "fractal": document.scene.fractal.value,
            "exponent": document.scene.exponent,
            "julia_c_real": document.scene.julia_c_real_text,
            "julia_c_imag": document.scene.julia_c_imag_text,
        },
        "surface_lighting": {
            "enabled": document.surface_lighting.enabled,
            "strength": document.surface_lighting.strength,
            "azimuth_degrees": document.surface_lighting.azimuth_degrees,
            "elevation_degrees": document.surface_lighting.elevation_degrees,
            "ambient": document.surface_lighting.ambient,
            "diffuse": document.surface_lighting.diffuse,
        },
        "camera_keyframes": [
            {
                "time_seconds": frame.time_seconds_text,
                "center_x": frame.camera.center_x_text,
                "center_y": frame.camera.center_y_text,
                "view_width": frame.camera.view_width_text,
                "easing": frame.easing.value,
                "center_interpolation": frame.center_interpolation.value,
            }
            for frame in document.path.keyframes
        ],
        "render_cues": [
            {
                "time_seconds": cue.time_seconds_text,
                "max_iterations": cue.profile.max_iterations,
                "reference_bits": cue.profile.reference_bits,
                "palette": cue.profile.palette,
                "cycles": cue.profile.cycles_text,
                "palette_transition": cue.palette_transition.value,
            }
            for cue in document.render_track.cues
        ],
    }


def serialize_flight_plan(document: FlightPlanDocument) -> str:
    """Return deterministic UTF-8 JSON text ending with one newline."""

    return json.dumps(
        flight_plan_to_dict(document),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"


def deserialize_flight_plan(
    text: str,
    *,
    source: str = "<memory>",
    migration_defaults: FlightPlanDefaults = FlightPlanDefaults(),
) -> FlightPlanDocument:
    """Parse, migrate and strictly validate one versioned flight-plan document."""

    if not isinstance(text, str):
        raise TypeError("flight-plan JSON input must be text")
    if len(text.encode("utf-8")) > MAX_FLIGHT_PLAN_BYTES:
        raise FlightPlanFormatError(
            f"{source}: flight-plan document exceeds {MAX_FLIGHT_PLAN_BYTES} bytes"
        )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except FlightPlanFormatError as exc:
        raise FlightPlanFormatError(f"{source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FlightPlanFormatError(
            f"{source}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc

    root = _require_object(value, source)
    version = root.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise FlightPlanFormatError(f"{source}: schema_version must be an integer")
    if version == 1:
        return _deserialize_v1(root, source=source, defaults=migration_defaults)
    if version == 2:
        return _deserialize_v2(root, source=source)
    if version == FLIGHT_PLAN_SCHEMA_VERSION:
        return _deserialize_v3(root, source=source)
    raise FlightPlanFormatError(
        f"{source}: unsupported schema_version {version}; supported versions are 1, 2 and "
        f"{FLIGHT_PLAN_SCHEMA_VERSION}"
    )


def load_flight_plan(
    path: str | os.PathLike[str],
    *,
    migration_defaults: FlightPlanDefaults = FlightPlanDefaults(),
) -> FlightPlanDocument:
    """Load a bounded UTF-8 flight plan and migrate legacy schema 1 in memory."""

    source_path = Path(path)
    with source_path.open("rb") as handle:
        payload = handle.read(MAX_FLIGHT_PLAN_BYTES + 1)
    if len(payload) > MAX_FLIGHT_PLAN_BYTES:
        raise FlightPlanFormatError(
            f"{source_path}: flight-plan document exceeds {MAX_FLIGHT_PLAN_BYTES} bytes"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FlightPlanFormatError(f"{source_path}: document is not valid UTF-8") from exc
    return deserialize_flight_plan(
        text,
        source=str(source_path),
        migration_defaults=migration_defaults,
    )


def save_flight_plan(
    path: str | os.PathLike[str],
    document: FlightPlanDocument,
) -> Path:
    """Atomically replace a flight-plan file using a sibling temporary file."""

    target = Path(path)
    parent = target.parent
    if not parent.exists():
        raise FileNotFoundError(f"flight-plan directory does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"flight-plan parent is not a directory: {parent}")

    payload = serialize_flight_plan(document)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def suggested_flight_plan_name(path: str | os.PathLike[str]) -> str:
    """Derive a readable document name from a selected file path."""

    filename = Path(path).name
    lower = filename.casefold()
    if lower.endswith(FLIGHT_PLAN_EXTENSION):
        filename = filename[: -len(FLIGHT_PLAN_EXTENSION)]
    elif lower.endswith(".json"):
        filename = filename[:-5]
    return filename.strip() or "Flugplan"


def _deserialize_v1(
    root: Mapping[str, Any],
    *,
    source: str,
    defaults: FlightPlanDefaults,
) -> FlightPlanDocument:
    _require_exact_keys(
        root,
        {"format", "schema_version", "name", "digits", "keyframes"},
        source,
    )
    if root["format"] != LEGACY_FLIGHT_PLAN_FORMAT:
        raise FlightPlanFormatError(
            f"{source}: format must be {LEGACY_FLIGHT_PLAN_FORMAT!r} for schema 1"
        )
    name = _require_string(root["name"], f"{source}: name")
    digits = _validated_digits(root["digits"], source=source)
    path = _deserialize_camera_track(root["keyframes"], digits=digits, source=source)
    try:
        return FlightPlanDocument(
            name,
            path,
            defaults.scene,
            RenderTrack.default(defaults.render_profile, digits=digits),
            defaults.surface_lighting,
            source_schema_version=1,
        )
    except ValueError as exc:
        raise FlightPlanFormatError(f"{source}: invalid migrated flight plan: {exc}") from exc


def _deserialize_v2(
    root: Mapping[str, Any],
    *,
    source: str,
) -> FlightPlanDocument:
    _require_exact_keys(
        root,
        {
            "format",
            "schema_version",
            "name",
            "digits",
            "scene",
            "camera_keyframes",
            "render_cues",
        },
        source,
    )
    if root["format"] != FLIGHT_PLAN_FORMAT:
        raise FlightPlanFormatError(f"{source}: format must be {FLIGHT_PLAN_FORMAT!r}")
    name = _require_string(root["name"], f"{source}: name")
    digits = _validated_digits(root["digits"], source=source)
    scene = _deserialize_scene(root["scene"], source=source)
    path = _deserialize_camera_track(
        root["camera_keyframes"],
        digits=digits,
        source=source,
    )
    render_track = _deserialize_render_track(
        root["render_cues"],
        digits=digits,
        source=source,
    )
    try:
        return FlightPlanDocument(
            name,
            path,
            scene,
            render_track,
            SurfaceLightingSettings(),
            source_schema_version=2,
        )
    except ValueError as exc:
        raise FlightPlanFormatError(f"{source}: invalid flight plan: {exc}") from exc


def _deserialize_v3(
    root: Mapping[str, Any],
    *,
    source: str,
) -> FlightPlanDocument:
    _require_exact_keys(
        root,
        {
            "format",
            "schema_version",
            "name",
            "digits",
            "scene",
            "surface_lighting",
            "camera_keyframes",
            "render_cues",
        },
        source,
    )
    if root["format"] != FLIGHT_PLAN_FORMAT:
        raise FlightPlanFormatError(f"{source}: format must be {FLIGHT_PLAN_FORMAT!r}")
    name = _require_string(root["name"], f"{source}: name")
    digits = _validated_digits(root["digits"], source=source)
    scene = _deserialize_scene(root["scene"], source=source)
    surface_lighting = _deserialize_surface_lighting(
        root["surface_lighting"], source=source
    )
    path = _deserialize_camera_track(
        root["camera_keyframes"],
        digits=digits,
        source=source,
    )
    render_track = _deserialize_render_track(
        root["render_cues"],
        digits=digits,
        source=source,
    )
    try:
        return FlightPlanDocument(
            name,
            path,
            scene,
            render_track,
            surface_lighting,
            source_schema_version=3,
        )
    except ValueError as exc:
        raise FlightPlanFormatError(f"{source}: invalid flight plan: {exc}") from exc


def _deserialize_surface_lighting(
    value: Any,
    *,
    source: str,
) -> SurfaceLightingSettings:
    label = f"{source}: surface_lighting"
    lighting = _require_object(value, label)
    _require_exact_keys(
        lighting,
        {
            "enabled",
            "strength",
            "azimuth_degrees",
            "elevation_degrees",
            "ambient",
            "diffuse",
        },
        label,
    )
    enabled = lighting["enabled"]
    if type(enabled) is not bool:
        raise FlightPlanFormatError(f"{label}.enabled must be a boolean")
    try:
        return SurfaceLightingSettings(
            enabled=enabled,
            strength=_require_number(lighting["strength"], f"{label}.strength"),
            azimuth_degrees=_require_number(
                lighting["azimuth_degrees"], f"{label}.azimuth_degrees"
            ),
            elevation_degrees=_require_number(
                lighting["elevation_degrees"], f"{label}.elevation_degrees"
            ),
            ambient=_require_number(lighting["ambient"], f"{label}.ambient"),
            diffuse=_require_number(lighting["diffuse"], f"{label}.diffuse"),
        )
    except ValueError as exc:
        raise FlightPlanFormatError(f"{label}: {exc}") from exc


def _deserialize_scene(value: Any, *, source: str) -> FlightScene:
    label = f"{source}: scene"
    scene = _require_object(value, label)
    _require_exact_keys(
        scene,
        {"fractal", "exponent", "julia_c_real", "julia_c_imag"},
        label,
    )
    exponent = scene["exponent"]
    if isinstance(exponent, bool) or not isinstance(exponent, int):
        raise FlightPlanFormatError(f"{label}.exponent must be an integer")
    try:
        return FlightScene(
            FractalKind(_require_string(scene["fractal"], f"{label}.fractal")),
            exponent,
            _require_string(scene["julia_c_real"], f"{label}.julia_c_real"),
            _require_string(scene["julia_c_imag"], f"{label}.julia_c_imag"),
        )
    except ValueError as exc:
        raise FlightPlanFormatError(f"{label}: {exc}") from exc


def _deserialize_camera_track(
    value: Any,
    *,
    digits: int,
    source: str,
) -> CameraPath:
    if not isinstance(value, list):
        raise FlightPlanFormatError(f"{source}: camera keyframes must be an array")
    if len(value) < 2:
        raise FlightPlanFormatError(f"{source}: at least two camera keyframes are required")
    if len(value) > MAX_FLIGHT_PLAN_KEYFRAMES:
        raise FlightPlanFormatError(
            f"{source}: camera keyframes exceed {MAX_FLIGHT_PLAN_KEYFRAMES} entries"
        )

    frames: list[FlightKeyframe] = []
    expected = {
        "time_seconds",
        "center_x",
        "center_y",
        "view_width",
        "easing",
        "center_interpolation",
    }
    for index, item in enumerate(value):
        label = f"{source}: camera_keyframes[{index}]"
        frame_value = _require_object(item, label)
        _require_exact_keys(frame_value, expected, label)
        try:
            frame = FlightKeyframe(
                _require_string(frame_value["time_seconds"], f"{label}.time_seconds"),
                CameraState(
                    _require_string(frame_value["center_x"], f"{label}.center_x"),
                    _require_string(frame_value["center_y"], f"{label}.center_y"),
                    _require_string(frame_value["view_width"], f"{label}.view_width"),
                ),
                Easing(_require_string(frame_value["easing"], f"{label}.easing")),
                CenterInterpolation(
                    _require_string(
                        frame_value["center_interpolation"],
                        f"{label}.center_interpolation",
                    )
                ),
            )
        except ValueError as exc:
            raise FlightPlanFormatError(f"{label}: {exc}") from exc
        frames.append(frame)

    try:
        return CameraPath(tuple(frames), digits=digits)
    except ValueError as exc:
        raise FlightPlanFormatError(f"{source}: invalid camera path: {exc}") from exc


def _deserialize_render_track(
    value: Any,
    *,
    digits: int,
    source: str,
) -> RenderTrack:
    if not isinstance(value, list):
        raise FlightPlanFormatError(f"{source}: render_cues must be an array")
    if not value:
        raise FlightPlanFormatError(f"{source}: at least one render cue is required")
    if len(value) > MAX_FLIGHT_PLAN_RENDER_CUES:
        raise FlightPlanFormatError(
            f"{source}: render_cues exceed {MAX_FLIGHT_PLAN_RENDER_CUES} entries"
        )

    expected = {
        "time_seconds",
        "max_iterations",
        "reference_bits",
        "palette",
        "cycles",
        "palette_transition",
    }
    cues: list[RenderCue] = []
    for index, item in enumerate(value):
        label = f"{source}: render_cues[{index}]"
        cue_value = _require_object(item, label)
        _require_exact_keys(cue_value, expected, label)
        max_iterations = cue_value["max_iterations"]
        reference_bits = cue_value["reference_bits"]
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise FlightPlanFormatError(f"{label}.max_iterations must be an integer")
        if isinstance(reference_bits, bool) or not isinstance(reference_bits, int):
            raise FlightPlanFormatError(f"{label}.reference_bits must be an integer")
        try:
            profile = RenderProfile(
                max_iterations=max_iterations,
                reference_bits=reference_bits,
                palette=_require_string(cue_value["palette"], f"{label}.palette"),
                cycles_text=_require_string(cue_value["cycles"], f"{label}.cycles"),
            )
            cue = RenderCue(
                _require_string(cue_value["time_seconds"], f"{label}.time_seconds"),
                profile,
                PaletteTransition(
                    _require_string(
                        cue_value["palette_transition"],
                        f"{label}.palette_transition",
                    )
                ),
            )
        except ValueError as exc:
            raise FlightPlanFormatError(f"{label}: {exc}") from exc
        cues.append(cue)
    try:
        return RenderTrack(tuple(cues), digits=digits)
    except ValueError as exc:
        raise FlightPlanFormatError(f"{source}: invalid render track: {exc}") from exc


def _validated_digits(value: Any, *, source: str | None = None) -> int:
    label = "digits" if source is None else f"{source}: digits"
    if isinstance(value, bool) or not isinstance(value, int):
        raise FlightPlanFormatError(f"{label} must be an integer")
    if not 20 <= value <= _MAX_DIGITS:
        raise FlightPlanFormatError(f"{label} must be between 20 and {_MAX_DIGITS}")
    return value


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FlightPlanFormatError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise FlightPlanFormatError(f"invalid JSON constant {value!r}")


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise FlightPlanFormatError(f"{label}: expected a JSON object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise FlightPlanFormatError(f"{label} must be a string")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FlightPlanFormatError(f"{label} must be a number")
    return float(value)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(repr(item) for item in missing))
    if extra:
        details.append("unexpected " + ", ".join(repr(item) for item in extra))
    if details:
        raise FlightPlanFormatError(f"{label}: " + "; ".join(details))
