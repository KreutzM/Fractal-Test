from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Sequence

from .camera import CameraState
from .deep_zoom_targets import DeepZoomTarget
from .flight_path import CameraPath, CenterInterpolation, Easing
from .flight_plan import PaletteTransition
from .flight_plan_session import FlightPlanSession
from .flight_transition import (
    TransitionMode,
    TransitionPlan,
    TransitionSettings,
    TransitionTarget,
    end_render_profile,
    plan_transition,
)
from .path_editor import CameraPathDraft
from .transition_dialog import TargetTransitionDialog

CameraGetter = Callable[[], CameraState]
CameraPreview = Callable[[CameraState, str], None]
AspectRatioGetter = Callable[[], float]
PlaybackStarter = Callable[[str], None]


class CameraPathEditorWindow(tk.Toplevel):
    """Non-modal editor for exact X/Y/zoom camera keyframes."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        get_current_camera: CameraGetter,
        targets: Sequence[DeepZoomTarget],
        on_preview: CameraPreview,
        session: FlightPlanSession,
        get_aspect_ratio: AspectRatioGetter | None = None,
        on_play_from: PlaybackStarter | None = None,
    ) -> None:
        super().__init__(parent)
        self.title("Flugplan / Keyframes")
        self.geometry("1120x680")
        self.minsize(860, 540)
        self.transient(parent)

        self._get_current_camera = get_current_camera
        self._targets = tuple(targets)
        self._targets_by_name = {target.name: target for target in self._targets}
        self._on_preview = on_preview
        self._session = session
        self._get_aspect_ratio = get_aspect_ratio or (lambda: 16 / 9)
        self._on_play_from = on_play_from
        self._transition_dialog: TargetTransitionDialog | None = None
        self._displayed_draft: CameraPathDraft | None = None

        self.time_var = tk.StringVar(value=self.draft.suggested_time_text())
        self.easing_var = tk.StringVar(value=Easing.SMOOTHSTEP.value)
        self.center_interpolation_var = tk.StringVar(
            value=CenterInterpolation.FOCUS.value
        )
        self.center_x_var = tk.StringVar()
        self.center_y_var = tk.StringVar()
        self.width_var = tk.StringVar()
        self.target_var = tk.StringVar(value=self._targets[0].name if self._targets else "")
        self.preview_time_var = tk.StringVar(value="0")
        self.status_var = tk.StringVar()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._fill_camera(self._get_current_camera())
        self._session.add_listener(self._on_session_changed, notify=True)

    @property
    def draft(self) -> CameraPathDraft:
        return self._session.camera_draft

    @property
    def _draft(self) -> CameraPathDraft:
        """Compatibility view; the session remains the only draft owner."""

        return self._session.camera_draft

    @_draft.setter
    def _draft(self, draft: CameraPathDraft) -> None:
        self._session.set_camera_draft(draft)

    @property
    def current_path(self) -> CameraPath | None:
        return self._session.camera_path

    def _close(self) -> None:
        self.destroy()

    def destroy(self) -> None:
        self._session.remove_listener(self._on_session_changed)
        dialog = self._transition_dialog
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()
        super().destroy()

    def _on_session_changed(self, session: FlightPlanSession) -> None:
        if not self.winfo_exists():
            return
        draft_changed = self._displayed_draft != session.camera_draft
        if draft_changed:
            selected_time = None
            index = session.selected_keyframe_index
            if index is not None and index < len(session.camera_draft.keyframes):
                selected_time = session.camera_draft.keyframes[index].time_seconds_text
            self._refresh_tree(select_time=selected_time)
        self.preview_time_var.set(session.playhead_time_text)
        self._publish_path_state()

    def _build_ui(self) -> None:
        content = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        list_frame = ttk.Frame(content)
        editor_frame = ttk.Frame(content, padding=(14, 0, 0, 0))
        content.add(list_frame, weight=3)
        content.add(editor_frame, weight=2)

        columns = ("time", "easing", "center", "x", "y", "width")
        self.tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        headings = {
            "time": "Zeit (s)",
            "easing": "Easing danach",
            "center": "Mittelpunkt danach",
            "x": "Zentrum X",
            "y": "Zentrum Y",
            "width": "Ansichtsbreite",
        }
        widths = {
            "time": 90,
            "easing": 115,
            "center": 130,
            "x": 170,
            "y": 170,
            "width": 145,
        }
        for name in columns:
            self.tree.heading(name, text=headings[name])
            self.tree.column(name, width=widths[name], minwidth=70, stretch=name in {"x", "y"})
        scroll_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scroll_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._on_selection_changed)
        self.tree.bind("<Double-Button-1>", lambda _event: self._preview_selected_keyframe())

        form = ttk.LabelFrame(editor_frame, text="Keyframe", padding=10)
        form.pack(fill=tk.X)
        ttk.Label(form, text="Zeit in Sekunden").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.time_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(form, text="Easing zum nächsten Frame").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            form,
            textvariable=self.easing_var,
            values=tuple(item.value for item in Easing),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(form, text="Mittelpunkt zum nächsten Frame").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Combobox(
            form,
            textvariable=self.center_interpolation_var,
            values=tuple(item.value for item in CenterInterpolation),
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))
        ttk.Label(form, text="Zentrum X").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.center_x_var).grid(
            row=3, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(form, text="Zentrum Y").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.center_y_var).grid(
            row=4, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        ttk.Label(form, text="Ansichtsbreite").grid(row=5, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.width_var).grid(
            row=5, column=1, sticky="ew", padx=(8, 0), pady=(6, 0)
        )
        form.columnconfigure(1, weight=1)

        source = ttk.LabelFrame(editor_frame, text="Kamera übernehmen", padding=10)
        source.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(
            source,
            text="Aktuelle Ansicht",
            command=lambda: self._fill_camera(self._get_current_camera()),
        ).grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Combobox(
            source,
            textvariable=self.target_var,
            values=tuple(target.name for target in self._targets),
            state="readonly" if self._targets else "disabled",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(source, text="Katalogziel", command=self._fill_target_camera).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0)
        )
        ttk.Button(
            source,
            text="Katalogziel mit Übergang …",
            command=self._open_transition_dialog,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        source.columnconfigure(0, weight=1)

        actions = ttk.Frame(editor_frame)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="Hinzufügen", command=self._add_keyframe).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )
        ttk.Button(actions, text="Auswahl ändern", command=self._update_keyframe).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=6
        )
        ttk.Button(actions, text="Löschen", command=self._remove_keyframe).pack(
            side=tk.LEFT, expand=True, fill=tk.X
        )

        preview = ttk.LabelFrame(editor_frame, text="Pfadvorschau", padding=10)
        preview.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(preview, text="Zeit in Sekunden").grid(row=0, column=0, sticky="w")
        ttk.Entry(preview, textvariable=self.preview_time_var, width=14).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Button(preview, text="Vorschau anzeigen", command=self._preview_path).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(
            preview,
            text="Ausgewählten Keyframe anzeigen",
            command=self._preview_selected_keyframe,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        preview.columnconfigure(1, weight=1)

        ttk.Label(
            editor_frame,
            textvariable=self.status_var,
            wraplength=390,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(12, 0))
        ttk.Label(
            editor_frame,
            text=(
                "Easing und Mittelpunktmodus gehören zum ausgehenden Segment. "
                "focus hält das nächste Ziel während starker Zooms im Bild; linear erzeugt "
                "eine geradlinige X/Y-Fahrt. Die Vorschau verändert nur die aktuelle Ansicht."
            ),
            foreground="#555",
            wraplength=390,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(8, 0))

        footer = ttk.Frame(self, padding=(12, 0, 12, 12))
        footer.pack(fill=tk.X)
        ttk.Button(footer, text="Schließen", command=self._close).pack(side=tk.RIGHT)

    def _transition_source(self) -> tuple[CameraState, str]:
        frames = self.draft.keyframes
        if not frames:
            raise ValueError("Lege zuerst einen Ausgangs-Keyframe an.")
        frame = frames[-1]
        return frame.camera, frame.time_seconds_text

    def _build_catalog_transition(
        self,
        target: DeepZoomTarget,
        mode: TransitionMode,
        palette_transition: PaletteTransition,
    ) -> TransitionPlan:
        source_camera, start_time = self._transition_source()
        state = self._session.render_track.evaluate(start_time)
        source_profile = end_render_profile(render_state=state)
        transition_target = TransitionTarget.from_deep_zoom_target(
            target,
            scene=self._session.scene,
            cycles_text=state.cycles_text,
        )
        aspect = max(0.1, float(self._get_aspect_ratio()))
        settings = TransitionSettings(aspect_ratio_text=repr(aspect))
        return plan_transition(
            source_camera,
            source_profile,
            transition_target,
            start_time_text=start_time,
            digits=self.draft.digits,
            requested_mode=mode,
            palette_transition=palette_transition,
            settings=settings,
        )

    def _apply_catalog_transition(
        self, plan: TransitionPlan, play: bool = False
    ) -> None:
        self._session.append_transition(plan)
        self._refresh_tree(select_time=plan.arrival_time_text)
        self._publish_path_state()
        if play and self._on_play_from is not None:
            self._on_play_from(plan.start_time_text)

    def _append_catalog_transition(
        self,
        target: DeepZoomTarget,
        mode: TransitionMode | str = TransitionMode.AUTO,
        palette_transition: PaletteTransition | str = PaletteTransition.BLEND,
        *,
        play: bool = False,
    ) -> TransitionPlan:
        """Testable non-dialog entry point used by the GUI and smoke tests."""

        plan = self._build_catalog_transition(
            target,
            TransitionMode(mode),
            PaletteTransition(palette_transition),
        )
        self._apply_catalog_transition(plan, play)
        return plan

    def _open_transition_dialog(self) -> None:
        if self._transition_dialog is not None and self._transition_dialog.winfo_exists():
            self._transition_dialog.deiconify()
            self._transition_dialog.lift()
            return
        self._transition_dialog = TargetTransitionDialog(
            self,
            targets=self._targets,
            initial_target_name=self.target_var.get(),
            build_plan=self._build_catalog_transition,
            apply_plan=self._apply_catalog_transition,
        )

    def _camera_from_form(self) -> CameraState:
        camera = CameraState(
            self.center_x_var.get().strip(),
            self.center_y_var.get().strip(),
            self.width_var.get().strip(),
        )
        camera.values(digits=self.draft.digits)
        return camera

    def _fill_camera(self, camera: CameraState) -> None:
        self.center_x_var.set(camera.center_x_text)
        self.center_y_var.set(camera.center_y_text)
        self.width_var.set(camera.view_width_text)

    def _fill_target_camera(self) -> None:
        target = self._targets_by_name.get(self.target_var.get())
        if target is None:
            return
        self._fill_camera(
            CameraState(target.center_x_text, target.center_y_text, target.view_width_text)
        )

    def _selected_index(self) -> int | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return int(selection[0])

    def _on_selection_changed(self, _event: tk.Event | None = None) -> None:
        index = self._selected_index()
        if index is None or index >= len(self._draft.keyframes):
            return
        frame = self._draft.keyframes[index]
        self._session.set_selected_keyframe(index)
        self.time_var.set(frame.time_seconds_text)
        self.easing_var.set(frame.easing.value)
        self.center_interpolation_var.set(frame.center_interpolation.value)
        self._fill_camera(frame.camera)
        self.preview_time_var.set(frame.time_seconds_text)

    def _add_keyframe(self) -> None:
        try:
            draft = self.draft.add_keyframe(
                self.time_var.get().strip(),
                self._camera_from_form(),
                self.easing_var.get(),
                self.center_interpolation_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Keyframe", str(exc), parent=self)
            return
        selected_time = self.time_var.get().strip()
        self._session.set_camera_draft(draft)
        self.time_var.set(draft.suggested_time_text())
        self._refresh_tree(select_time=selected_time)
        self._publish_path_state()

    def _update_keyframe(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showinfo("Keyframe", "Wähle zuerst einen Keyframe aus.", parent=self)
            return
        try:
            draft = self.draft.update_keyframe(
                index,
                time_seconds_text=self.time_var.get().strip(),
                camera=self._camera_from_form(),
                easing=self.easing_var.get(),
                center_interpolation=self.center_interpolation_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Keyframe", str(exc), parent=self)
            return
        selected_time = self.time_var.get().strip()
        self._session.set_camera_draft(draft)
        self._refresh_tree(select_time=selected_time)
        self._publish_path_state()

    def _remove_keyframe(self) -> None:
        index = self._selected_index()
        if index is None:
            messagebox.showinfo("Keyframe", "Wähle zuerst einen Keyframe aus.", parent=self)
            return
        self._session.set_camera_draft(self.draft.remove_keyframe(index))
        self._refresh_tree()
        self._publish_path_state()

    def _refresh_tree(self, *, select_time: str | None = None) -> None:
        self.tree.delete(*self.tree.get_children())
        selected_item: str | None = None
        draft = self.draft
        self._displayed_draft = draft
        for index, frame in enumerate(draft.keyframes):
            item = str(index)
            self.tree.insert(
                "",
                tk.END,
                iid=item,
                values=(
                    frame.time_seconds_text,
                    frame.easing.value,
                    frame.center_interpolation.value,
                    _compact(frame.camera.center_x_text),
                    _compact(frame.camera.center_y_text),
                    _compact(frame.camera.view_width_text),
                ),
            )
            if select_time is not None and frame.time_seconds_text == select_time:
                selected_item = item
        if selected_item is None and draft.keyframes:
            selected_index = self._session.selected_keyframe_index
            selected_item = str(selected_index if selected_index is not None else 0)
        if selected_item is not None:
            self.tree.selection_set(selected_item)
            self.tree.focus(selected_item)
            self.tree.see(selected_item)
            self._on_selection_changed()

    def _publish_path_state(self) -> None:
        draft = self.draft
        error = self._session.validation_error
        if error is None:
            path = self._session.camera_path
            assert path is not None
            self.status_var.set(
                f"Gültiger Flugplan: {len(path.keyframes)} Keyframes, Dauer {path.duration_text} s; "
                f"{len(self._session.render_track.cues)} Render-Cues."
            )
        else:
            self.status_var.set(f"Entwurf noch nicht exportierbar: {error}")

    def _preview_path(self) -> None:
        try:
            path = self.draft.build_path()
            time_text = self.preview_time_var.get().strip()
            camera = path.evaluate(time_text)
        except Exception as exc:
            messagebox.showerror("Pfadvorschau", str(exc), parent=self)
            return
        self._session.set_playhead(time_text)
        self._on_preview(camera, time_text)

    def _preview_selected_keyframe(self) -> None:
        index = self._selected_index()
        if index is None:
            return
        frame = self.draft.keyframes[index]
        self._session.set_playhead(frame.time_seconds_text)
        self._on_preview(frame.camera, frame.time_seconds_text)


def _compact(text: str, limit: int = 25) -> str:
    if len(text) <= limit:
        return text
    head = max(8, (limit - 1) // 2)
    tail = max(8, limit - head - 1)
    return text[:head] + "…" + text[-tail:]
