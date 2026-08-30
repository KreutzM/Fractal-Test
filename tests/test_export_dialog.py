from __future__ import annotations

from fractal_flight_studio.camera import CameraState
from fractal_flight_studio.export_warning_dialog import FlightExportDialog
from fractal_flight_studio.flight_path import CameraPath, FlightKeyframe
from fractal_flight_studio.models import RenderRequest
from fractal_flight_studio.preflight import PreflightSettings, run_path_preflight

from test_preflight import _Renderer


def _path() -> CameraPath:
    return CameraPath(
        (
            FlightKeyframe("0", CameraState("-0.5", "0", "3")),
            FlightKeyframe("1", CameraState("-0.75", "0.1", "0.1")),
        )
    )


def test_report_formats_visual_findings_as_non_blocking_warnings() -> None:
    report = run_path_preflight(
        _path(),
        RenderRequest(),
        _Renderer(mode="uniform"),
        PreflightSettings(width=32, height=24, sample_interval_seconds_text="1"),
    )

    text = FlightExportDialog._format_preflight_report(report)

    assert report.exportable
    assert "Ergebnis: exportierbar mit visuellen Warnungen" in text
    assert "[Warnung; visual]" in text
    assert "Export blockiert" not in text


def test_report_formats_numerical_findings_as_blocking() -> None:
    report = run_path_preflight(
        _path(),
        RenderRequest(),
        _Renderer(mode="unsafe-grid"),
        PreflightSettings(width=32, height=24, sample_interval_seconds_text="1"),
    )

    text = FlightExportDialog._format_preflight_report(report)

    assert not report.exportable
    assert "Ergebnis: Export blockiert" in text
    assert "[Blockierend; numerical]" in text
