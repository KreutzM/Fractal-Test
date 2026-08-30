from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from .export_controller import (
    FlightExportConfiguration,
    FlightExportController,
    FlightExportJobKind,
    flight_export_fingerprint,
)
from .ffmpeg_mp4 import Mp4ExportCancelled, Mp4ExportResult
from .flight_plan import FlightSource
from .models import RenderRequest
from .preflight import PreflightCancelled, PreflightReport


PathGetter = Callable[[], FlightSource | None]
RequestBuilder = Callable[[], RenderRequest]
RendererGetter = Callable[[], object]
StringGetter = Callable[[], str]
FloatGetter = Callable[[], float]
ReadyCheck = Callable[[], bool]
JobFinished = Callable[[], None]


class FlightExportDialog(tk.Toplevel):
    """Non-modal UI for path preflight and direct FFmpeg MP4 export."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        controller: FlightExportController,
        get_path: PathGetter,
        build_request: RequestBuilder,
        get_renderer: RendererGetter,
        get_palette: StringGetter,
        get_cycles: FloatGetter,
        ready_for_background_job: ReadyCheck,
        on_job_finished: JobFinished,
    ) -> None:
        super().__init__(parent)
        self.title("Flug-Preflight und MP4-Export")
        self.geometry("880x760")
        self.minsize(760, 620)
        self.transient(parent)

        self._controller = controller
        self._get_path = get_path
        self._build_request = build_request
        self._get_renderer = get_renderer
        self._get_palette = get_palette
        self._get_cycles = get_cycles
        self._ready_for_background_job = ready_for_background_job
        self._on_job_finished = on_job_finished
        self._future: Future | None = None
        self._job_kind: FlightExportJobKind | None = None
        self._pending_preflight_fingerprint: tuple[object, ...] | None = None
        self._approved_preflight_fingerprint: tuple[object, ...] | None = None
        self._preflight_report: PreflightReport | None = None

        self.width_var = tk.IntVar(value=1920)
        self.height_var = tk.IntVar(value=1080)
        self.frame_rate_var = tk.StringVar(value="30")
        self.preflight_width_var = tk.IntVar(value=320)
        self.preflight_height_var = tk.IntVar(value=180)
        self.preflight_interval_var = tk.StringVar(value="0.5")
        self.preflight_samples_var = tk.IntVar(value=240)
        self.ffmpeg_var = tk.StringVar(value="ffmpeg")
        self.codec_var = tk.StringVar(value="libx264")
        self.preset_var = tk.StringVar(value="medium")
        self.crf_var = tk.IntVar(value=18)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.output_var = tk.StringVar(value="")
        self.path_summary_var = tk.StringVar()
        self.plan_summary_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Bereit.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.refresh_path_summary()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        path_frame = ttk.LabelFrame(outer, text="Flugplan", padding=10)
        path_frame.pack(fill=tk.X)
        ttk.Label(
            path_frame,
            textvariable=self.path_summary_var,
            wraplength=820,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X)
        ttk.Button(
            path_frame,
            text="Plan aktualisieren",
            command=self.refresh_path_summary,
        ).pack(anchor="e", pady=(6, 0))

        settings = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        settings.pack(fill=tk.X, pady=(10, 0))
        video = ttk.LabelFrame(settings, text="Video", padding=10)
        preflight = ttk.LabelFrame(settings, text="Preflight", padding=10)
        settings.add(video, weight=1)
        settings.add(preflight, weight=1)

        self._labeled_entry(video, 0, "Breite", self.width_var)
        self._labeled_entry(video, 1, "Höhe", self.height_var)
        ttk.Label(video, text="Framerate").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            video,
            textvariable=self.frame_rate_var,
            values=("24", "25", "30", "50", "60", "30000/1001", "60000/1001"),
        ).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(video, text="Codec").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            video,
            textvariable=self.codec_var,
            values=("libx264", "libx265"),
        ).grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(video, text="Preset").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            video,
            textvariable=self.preset_var,
            values=("ultrafast", "fast", "medium", "slow", "veryslow"),
        ).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        self._labeled_entry(video, 5, "CRF", self.crf_var)
        ttk.Checkbutton(
            video,
            text="Vorhandene Datei überschreiben",
            variable=self.overwrite_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))
        video.columnconfigure(1, weight=1)

        self._labeled_entry(preflight, 0, "Breite", self.preflight_width_var)
        self._labeled_entry(preflight, 1, "Höhe", self.preflight_height_var)
        self._labeled_entry(preflight, 2, "Intervall (s)", self.preflight_interval_var)
        self._labeled_entry(preflight, 3, "Max. Stichproben", self.preflight_samples_var)
        ttk.Label(
            preflight,
            text=(
                "Der Preflight rendert kleine Stichproben entlang des exakten Pfads und "
                "prüft numerische sowie visuelle Ausfälle."
            ),
            foreground="#555",
            wraplength=350,
            justify=tk.LEFT,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        preflight.columnconfigure(1, weight=1)

        ffmpeg = ttk.LabelFrame(outer, text="FFmpeg und Zieldatei", padding=10)
        ffmpeg.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(ffmpeg, text="FFmpeg").grid(row=0, column=0, sticky="w")
        ttk.Entry(ffmpeg, textvariable=self.ffmpeg_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 6)
        )
        self.probe_button = ttk.Button(ffmpeg, text="Prüfen", command=self._start_probe)
        self.probe_button.grid(row=0, column=2)
        ttk.Label(ffmpeg, text="MP4-Datei").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ffmpeg, textvariable=self.output_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 6), pady=(6, 0)
        )
        ttk.Button(ffmpeg, text="Auswählen …", command=self._choose_output).grid(
            row=1, column=2, pady=(6, 0)
        )
        ffmpeg.columnconfigure(1, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, pady=(10, 0))
        self.preflight_button = ttk.Button(
            actions,
            text="Preflight starten",
            command=self._start_preflight,
        )
        self.preflight_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.export_button = ttk.Button(
            actions,
            text="MP4 exportieren",
            command=self._start_export,
        )
        self.export_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=6)
        self.cancel_button = ttk.Button(
            actions,
            text="Abbrechen",
            command=self._cancel,
            state=tk.DISABLED,
        )
        self.cancel_button.pack(side=tk.LEFT)

        ttk.Label(
            outer,
            textvariable=self.plan_summary_var,
            wraplength=820,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(10, 0))
        self.progress = ttk.Progressbar(
            outer,
            variable=self.progress_var,
            maximum=100.0,
        )
        self.progress.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            outer,
            textvariable=self.status_var,
            wraplength=820,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(6, 0))

        report_frame = ttk.LabelFrame(outer, text="Preflight-Bericht", padding=8)
        report_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.report_text = tk.Text(report_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        report_scroll = ttk.Scrollbar(
            report_frame,
            orient=tk.VERTICAL,
            command=self.report_text.yview,
        )
        self.report_text.configure(yscrollcommand=report_scroll.set)
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(footer, text="Schließen", command=self._close).pack(side=tk.RIGHT)

    @staticmethod
    def _labeled_entry(parent, row: int, label: str, variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(parent, textvariable=variable).grid(
            row=row,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=(6, 0),
        )

    def refresh_path_summary(self) -> None:
        path = self._get_path()
        if path is None:
            self.path_summary_var.set(
                "Kein gültiger Flugplan vorhanden. Erstelle zuerst mindestens zwei "
                "Keyframes im Flugplan-Editor."
            )
            self.plan_summary_var.set("")
            return
        self.path_summary_var.set(
            f"{len(path.keyframes)} Keyframes; Dauer {path.duration_text} s; "
            f"Pfadpräzision {path.digits} Dezimalstellen."
        )
        try:
            config = self._configuration()
            frames = config.export_frame_count(path)
            numerator, denominator = config.frame_rate
            self.plan_summary_var.set(
                f"Geplanter Export: {config.width}×{config.height}, "
                f"{numerator}/{denominator} fps, {frames} Frames."
            )
        except Exception as exc:
            self.plan_summary_var.set(f"Ungültige Exporteinstellungen: {exc}")

    def _configuration(self) -> FlightExportConfiguration:
        return FlightExportConfiguration(
            width=int(self.width_var.get()),
            height=int(self.height_var.get()),
            frame_rate_text=self.frame_rate_var.get(),
            preflight_width=int(self.preflight_width_var.get()),
            preflight_height=int(self.preflight_height_var.get()),
            preflight_interval_text=self.preflight_interval_var.get(),
            preflight_max_samples=int(self.preflight_samples_var.get()),
            ffmpeg_executable=self.ffmpeg_var.get().strip(),
            video_codec=self.codec_var.get().strip(),
            preset=self.preset_var.get().strip(),
            crf=int(self.crf_var.get()),
            overwrite=bool(self.overwrite_var.get()),
        )

    def _context(self):
        path = self._get_path()
        if path is None:
            raise ValueError("Es ist kein gültiger Flugplan vorhanden.")
        config = self._configuration()
        request = self._build_request()
        renderer = self._get_renderer()
        palette = self._get_palette()
        cycles = self._get_cycles()
        fingerprint = flight_export_fingerprint(
            path,
            request,
            getattr(renderer, "name", "unknown"),
            palette,
            cycles,
            config,
        )
        return path, config, request, renderer, palette, cycles, fingerprint

    def _ensure_startable(self) -> bool:
        if self._controller.busy:
            self.status_var.set("Es läuft bereits ein Exportauftrag.")
            return False
        if not self._ready_for_background_job():
            self.status_var.set(
                "Die interaktive Vorschau rendert noch. Warte kurz und starte den Auftrag erneut."
            )
            return False
        return True

    def _start_probe(self) -> None:
        if not self._ensure_startable():
            return
        try:
            config = self._configuration()
            future = self._controller.start_probe(config.ffmpeg_executable)
        except Exception as exc:
            self._show_error("FFmpeg-Prüfung", exc)
            return
        self._begin_job(future, FlightExportJobKind.PROBE)

    def _start_preflight(self) -> None:
        if not self._ensure_startable():
            return
        try:
            path, config, request, renderer, palette, cycles, fingerprint = self._context()
            future = self._controller.start_preflight(
                path,
                request,
                renderer,
                config.preflight_settings(),
                palette=palette,
                cycles=cycles,
            )
        except Exception as exc:
            self._show_error("Preflight", exc)
            return
        self._pending_preflight_fingerprint = fingerprint
        self._approved_preflight_fingerprint = None
        self._preflight_report = None
        self._set_report("Preflight läuft …")
        self._begin_job(future, FlightExportJobKind.PREFLIGHT)

    def _start_export(self) -> None:
        if not self._ensure_startable():
            return
        try:
            path, config, request, renderer, palette, cycles, fingerprint = self._context()
            if (
                self._preflight_report is None
                or not self._preflight_report.safe
                or fingerprint != self._approved_preflight_fingerprint
            ):
                raise ValueError(
                    "Vor dem Export ist ein erfolgreicher Preflight mit den aktuellen "
                    "Pfad- und Rendereinstellungen erforderlich."
                )
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
            )
        except Exception as exc:
            self._show_error("MP4-Export", exc)
            return
        self._begin_job(future, FlightExportJobKind.MP4)

    def _begin_job(self, future: Future, kind: FlightExportJobKind) -> None:
        self._future = future
        self._job_kind = kind
        self._set_running(True)
        self.progress_var.set(0.0)
        self.after(80, self._poll_job)

    def _poll_job(self) -> None:
        future = self._future
        if future is None:
            return
        progress = self._controller.progress
        if progress is not None:
            self.progress_var.set(progress.fraction * 100.0)
            self.status_var.set(progress.message)
        if not future.done():
            self.after(80, self._poll_job)
            return

        kind = self._job_kind
        try:
            result = future.result()
            if kind is FlightExportJobKind.PROBE:
                self.status_var.set(
                    f"FFmpeg bereit: {result.version_line}\n{result.executable}"
                )
            elif kind is FlightExportJobKind.PREFLIGHT:
                self._finish_preflight(result)
            elif kind is FlightExportJobKind.MP4:
                self._finish_export(result)
        except (PreflightCancelled, Mp4ExportCancelled) as exc:
            self.status_var.set(str(exc))
        except Exception as exc:
            self._show_error("Exportauftrag", exc)
        finally:
            self._controller.complete(future)
            self._future = None
            self._job_kind = None
            self._set_running(False)
            self._on_job_finished()

    def _finish_preflight(self, report: PreflightReport) -> None:
        self._preflight_report = report
        if report.safe:
            self._approved_preflight_fingerprint = self._pending_preflight_fingerprint
            self.status_var.set(
                f"Preflight bestanden: {len(report.samples)} Stichproben in "
                f"{report.total_elapsed_seconds:.2f} s Renderzeit."
            )
        else:
            self._approved_preflight_fingerprint = None
            self.status_var.set(
                f"Preflight nicht bestanden: {len(report.issues)} Problem(e) in "
                f"{len(report.samples)} Stichproben."
            )
        self._set_report(self._format_preflight_report(report))

    def _finish_export(self, result: Mp4ExportResult) -> None:
        self.progress_var.set(100.0)
        self.status_var.set(
            f"MP4 exportiert: {result.output_path}\n"
            f"{result.frame_count} Frames, {result.elapsed_seconds:.1f} s."
        )

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
        else:
            lines.append("Ergebnis: nicht sicher")
            for issue in report.issues:
                lines.append(
                    f"- t={issue.time_seconds_text} s [{issue.kind.value}]: {issue.reason}"
                )
        return "\n".join(lines)

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="MP4-Zieldatei",
            defaultextension=".mp4",
            filetypes=[("MP4-Video", "*.mp4")],
        )
        if path:
            self.output_var.set(path)

    def _cancel(self) -> None:
        if self._controller.busy:
            self._controller.cancel()

    def _set_running(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.probe_button.configure(state=state)
        self.preflight_button.configure(state=state)
        self.export_button.configure(state=state)
        self.cancel_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _set_report(self, text: str) -> None:
        self.report_text.configure(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert("1.0", text)
        self.report_text.configure(state=tk.DISABLED)

    def _show_error(self, title: str, error: Exception) -> None:
        self.status_var.set(str(error))
        messagebox.showerror(title, str(error), parent=self)

    def _close(self) -> None:
        if self._controller.busy:
            messagebox.showinfo(
                "Export läuft",
                "Brich den laufenden Auftrag zuerst ab, bevor du das Fenster schließt.",
                parent=self,
            )
            return
        self.destroy()
