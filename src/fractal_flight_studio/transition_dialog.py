from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Sequence

from .deep_zoom_targets import DeepZoomTarget
from .flight_plan import PaletteTransition
from .flight_transition import TransitionMode, TransitionPlan


PlanBuilder = Callable[
    [DeepZoomTarget, TransitionMode, PaletteTransition], TransitionPlan
]
PlanConsumer = Callable[[TransitionPlan, bool], None]


class TargetTransitionDialog(tk.Toplevel):
    """Compact catalog-target dialog with a deterministic route preview."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        targets: Sequence[DeepZoomTarget],
        initial_target_name: str | None,
        build_plan: PlanBuilder,
        apply_plan: PlanConsumer,
    ) -> None:
        super().__init__(parent)
        self.title("Katalogziel mit Übergang hinzufügen")
        self.geometry("590x390")
        self.minsize(500, 340)
        self.transient(parent)
        self.resizable(True, False)

        self._targets = tuple(targets)
        self._targets_by_name = {target.name: target for target in self._targets}
        self._build_plan = build_plan
        self._apply_plan = apply_plan
        self._current_plan: TransitionPlan | None = None

        initial = initial_target_name if initial_target_name in self._targets_by_name else ""
        if not initial and self._targets:
            initial = self._targets[0].name
        self.target_var = tk.StringVar(value=initial)
        self.mode_var = tk.StringVar(value=TransitionMode.AUTO.value)
        self.palette_var = tk.StringVar(value=PaletteTransition.BLEND.value)
        self.summary_var = tk.StringVar(value="")
        self.details_var = tk.StringVar(value="")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _event: self.destroy())
        self._refresh_plan()
        self.grab_set()
        self.focus_set()

    @property
    def current_plan(self) -> TransitionPlan | None:
        return self._current_plan

    def _build_ui(self) -> None:
        content = ttk.Frame(self, padding=14)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(1, weight=1)

        ttk.Label(content, text="Neues Deep-Zoom-Ziel").grid(
            row=0, column=0, sticky="w"
        )
        target_box = ttk.Combobox(
            content,
            textvariable=self.target_var,
            values=tuple(target.name for target in self._targets),
            state="readonly" if self._targets else "disabled",
        )
        target_box.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        target_box.bind("<<ComboboxSelected>>", self._refresh_plan)

        ttk.Label(content, text="Übergang").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        mode_box = ttk.Combobox(
            content,
            textvariable=self.mode_var,
            values=tuple(mode.value for mode in TransitionMode),
            state="readonly",
        )
        mode_box.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))
        mode_box.bind("<<ComboboxSelected>>", self._refresh_plan)

        ttk.Label(content, text="Palette").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        palette_box = ttk.Combobox(
            content,
            textvariable=self.palette_var,
            values=(
                PaletteTransition.BLEND.value,
                PaletteTransition.HOLD.value,
                PaletteTransition.CUT.value,
            ),
            state="readonly",
        )
        palette_box.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))
        palette_box.bind("<<ComboboxSelected>>", self._refresh_plan)

        summary = ttk.LabelFrame(content, text="Geplanter Abschnitt", padding=12)
        summary.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Label(
            summary,
            textvariable=self.summary_var,
            font=("TkDefaultFont", 10, "bold"),
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X)
        ttk.Label(
            summary,
            textvariable=self.details_var,
            wraplength=520,
            justify=tk.LEFT,
            foreground="#555",
        ).pack(anchor="w", fill=tk.X, pady=(8, 0))

        ttk.Label(
            content,
            text=(
                "auto wählt den kleinsten sicheren Übergang. direct fliegt ohne "
                "Zwischenebene, bridge nutzt zwei Brücken-Keyframes, overview führt "
                "über die Gesamtansicht und cut schaltet die Kamera hart um."
            ),
            wraplength=550,
            justify=tk.LEFT,
            foreground="#555",
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(12, 0))

        buttons = ttk.Frame(content)
        buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        ttk.Button(buttons, text="Abbrechen", command=self.destroy).pack(
            side=tk.RIGHT
        )
        self.play_button = ttk.Button(
            buttons,
            text="Hinzufügen und abspielen",
            command=lambda: self._accept(play=True),
        )
        self.play_button.pack(side=tk.RIGHT, padx=(0, 8))
        self.add_button = ttk.Button(
            buttons,
            text="Hinzufügen",
            command=lambda: self._accept(play=False),
        )
        self.add_button.pack(side=tk.RIGHT, padx=(0, 8))

    def _selected_target(self) -> DeepZoomTarget | None:
        return self._targets_by_name.get(self.target_var.get())

    def _refresh_plan(self, _event: tk.Event | None = None) -> None:
        target = self._selected_target()
        if target is None:
            self._current_plan = None
            self.summary_var.set("Kein Ziel verfügbar.")
            self.details_var.set("")
            self.add_button.state(("disabled",))
            self.play_button.state(("disabled",))
            return
        try:
            plan = self._build_plan(
                target,
                TransitionMode(self.mode_var.get()),
                PaletteTransition(self.palette_var.get()),
            )
        except Exception as exc:
            self._current_plan = None
            self.summary_var.set(f"Planung nicht möglich: {exc}")
            self.details_var.set("")
            self.add_button.state(("disabled",))
            self.play_button.state(("disabled",))
            return

        self._current_plan = plan
        target_profile = plan.render_cues[-1].profile
        requested = (
            ""
            if plan.requested_mode is plan.mode
            else f" (auto → {plan.mode.value})"
        )
        self.summary_var.set(
            f"{target.name}: {plan.summary}{requested}"
        )
        self.details_var.set(
            f"Ankunft bei {plan.arrival_time_text} s; Ende bei {plan.end_time_text} s. "
            f"Zielprofil: {target_profile.max_iterations} Iterationen, "
            f"{target_profile.reference_bits} Bit, Palette {target_profile.palette}."
        )
        self.add_button.state(("!disabled",))
        self.play_button.state(("!disabled",))

    def _accept(self, *, play: bool) -> None:
        plan = self._current_plan
        if plan is None:
            return
        try:
            self._apply_plan(plan, play)
        except Exception as exc:
            messagebox.showerror("Übergang", str(exc), parent=self)
            return
        self.destroy()
