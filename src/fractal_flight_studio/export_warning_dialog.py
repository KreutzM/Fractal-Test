from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from .export_controller import FlightExportJobKind
from .export_dialog import FlightExportDialog as BaseFlightExportDialog
from .preflight import PreflightIssueKind, PreflightReport
from .temporal_tonemapping import ToneStability


_TONE_STABILITY_LABELS = {
    "Zeitlich stabilisiert": ToneStability.TEMPORAL,
    "Automatisch pro Frame": ToneStability.PER_FRAME,
}


class FlightExportDialog(BaseFlightExportDialog):
    """Export dialog with visual-warning confirmation and video tone policy."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tone_stability_var = tk.StringVar(value="Zeitlich stabilisiert")
        self._install_tone_stability_controls()
        self.tone_stability_var.trace_add(
            "write", lambda *_args: self.refresh_path_summary()
        )
        self.refresh_path_summary()

    def _install_tone_stability_controls(self) -> None:
        outer = self.winfo_children()[0]
        settings = next(
            child for child in outer.winfo_children() if isinstance(child, ttk.Panedwindow)
        )
        video = settings.winfo_children()[0]
        overwrite = next(
            child for child in video.winfo_children() if isinstance(child, ttk.Checkbutton)
        )
        overwrite.grid_configure(row=7)
        ttk.Label(video, text="Tone Mapping").grid(
            row=6, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Combobox(
            video,
            textvariable=self.tone_stability_var,
            values=tuple(_TONE_STABILITY_LABELS),
            state="readonly",
        ).grid(row=6, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

    def _configuration(self):
        config = super()._configuration()
        variable = getattr(self, "tone_stability_var", None)
        mode = (
            _TONE_STABILITY_LABELS[variable.get()]
            if variable is not None
            else ToneStability.TEMPORAL
        )
        return replace(config, tone_stability=mode)

    def refresh_path_summary(self) -> None:
        super().refresh_path_summary()
        variable = getattr(self, "tone_stability_var", None)
        summary = self.plan_summary_var.get()
        if variable is not None and summary and not summary.startswith("Ungültige"):
            self.plan_summary_var.set(
                f"{summary.rstrip('.')}; Tone Mapping: {variable.get()}."
            )

    def _start_export(self) -> None:
        if not self._ensure_startable():
            return
        try:
            path, config, request, renderer, palette, cycles, fingerprint = self._context()
            if (
                self._preflight_report is None
                or not self._preflight_report.exportable
                or fingerprint != self._approved_preflight_fingerprint
            ):
                raise ValueError(
                    "Vor dem Export ist ein vollständiger Preflight ohne blockierende Fehler "
                    "mit den aktuellen Pfad- und Rendereinstellungen erforderlich."
                )
            if self._preflight_report.warnings and not messagebox.askyesno(
                "Visuelle Preflight-Warnungen",
                (
                    f"Der Preflight enthält {len(self._preflight_report.warnings)} "
                    "visuelle Warnung(en). Diese können legitime, strukturarme "
                    "Übergangsbilder sein, aber auch sichtbare Bildfehler anzeigen.\n\n"
                    "Trotzdem exportieren?"
                ),
                parent=self,
            ):
                self.status_var.set("MP4-Export wegen visueller Warnungen nicht gestartet.")
                return
            output_text = self.output_var.get().strip()
            if not output_text:
                raise ValueError("Wähle zuerst eine MP4-Zieldatei.")
            output = Path(output_text).expanduser()
            if output.suffix.casefold() != ".mp4":
                raise ValueError("Die Zieldatei muss die Erweiterung .mp4 verwenden.")
            offline_plan = config.build_offline_plan(path)
            future = self._controller.start_mp4(
                path,
                request,
                renderer,
                offline_plan,
                output,
                config.mp4_settings(),
                palette=palette,
                cycles=cycles,
                temporal_tone=config.temporal_tone_settings(),
            )
        except Exception as exc:
            self._show_error("MP4-Export", exc)
            return
        self._begin_job(future, FlightExportJobKind.MP4)

    def _finish_preflight(self, report: PreflightReport) -> None:
        self._preflight_report = report
        if report.safe:
            self._approved_preflight_fingerprint = self._pending_preflight_fingerprint
            self.status_var.set(
                f"Preflight bestanden: {len(report.samples)} Stichproben in "
                f"{report.total_elapsed_seconds:.2f} s Renderzeit."
            )
        elif report.exportable:
            self._approved_preflight_fingerprint = self._pending_preflight_fingerprint
            self.status_var.set(
                f"Preflight mit {len(report.warnings)} visuellen Warnung(en) abgeschlossen. "
                "Export ist nach Bestätigung möglich."
            )
        else:
            self._approved_preflight_fingerprint = None
            self.progress_var.set(0.0)
            self.status_var.set(
                f"Preflight blockiert den Export: {len(report.blocking_issues)} "
                f"schwerwiegende(s) Problem(e) in {len(report.samples)} Stichproben."
            )
        self._set_report(self._format_preflight_report(report))

    @staticmethod
    def _format_preflight_report(report: PreflightReport) -> str:
        lines = [
            f"Stichproben: {len(report.samples)}/{len(report.plan.sample_times_text)}",
            f"Auflösung: {report.plan.width}×{report.plan.height}",
            f"Zeitlich ausgedünnt: {'ja' if report.plan.decimated else 'nein'}",
            f"Früh beendet: {'ja' if report.stopped_early else 'nein'}",
        ]
        if report.safe:
            lines.append("Ergebnis: sicher")
        elif report.exportable:
            lines.append("Ergebnis: exportierbar mit visuellen Warnungen")
        else:
            lines.append("Ergebnis: Export blockiert")
        for issue in report.issues:
            severity = (
                "Warnung"
                if issue.kind is PreflightIssueKind.VISUAL
                else "Blockierend"
            )
            lines.append(
                f"- t={issue.time_seconds_text} s [{severity}; {issue.kind.value}]: "
                f"{issue.reason}"
            )
        return "\n".join(lines)
