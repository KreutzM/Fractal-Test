from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from .flight_plan import PaletteTransition
from .flight_transition import FreeTargetValues, TransitionMode, TransitionPlan
from .palettes import palette_names



PlanBuilder = Callable[
    [FreeTargetValues, TransitionMode, PaletteTransition], TransitionPlan
]
PlanConsumer = Callable[[TransitionPlan, bool], None]


class FreeTargetTransitionDialog(tk.Toplevel):
    """Right-click confirmation dialog for appending a free flight target."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        initial_values: FreeTargetValues,
        build_plan: PlanBuilder,
        apply_plan: PlanConsumer,
    ) -> None:
        super().__init__(parent)
        self.title("Nächstes Flugziel hinzufügen")
        self.geometry("650x560")
        self.minsize(560, 500)
        self.transient(parent)
        self.resizable(True, False)

        self._build_plan = build_plan
        self._apply_plan = apply_plan
        self._current_plan: TransitionPlan | None = None

        self.center_x_var = tk.StringVar(value=initial_values.center_x_text)
        self.center_y_var = tk.StringVar(value=initial_values.center_y_text)
        self.width_var = tk.StringVar(value=initial_values.view_width_text)
        self.iterations_var = tk.StringVar(value=str(initial_values.max_iterations))
        self.reference_bits_var = tk.StringVar(value=str(initial_values.reference_bits))
        self.palette_var = tk.StringVar(value=initial_values.palette)
        self.cycles_var = tk.StringVar(value=initial_values.cycles_text)
        self.mode_var = tk.StringVar(value=TransitionMode.AUTO.value)
        self.palette_transition_var = tk.StringVar(value=PaletteTransition.HOLD.value)
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

        row = 0
        for label, variable in (
            ("Zentrum X", self.center_x_var),
            ("Zentrum Y", self.center_y_var),
            ("Zielbreite", self.width_var),
        ):
            ttk.Label(content, text=label).grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 8, 0))
            entry = ttk.Entry(content, textvariable=variable)
            entry.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=(0 if row == 0 else 8, 0))
            entry.bind("<FocusOut>", self._refresh_plan)
            row += 1

        ttk.Label(content, text="Übergang").grid(row=row, column=0, sticky="w", pady=(8, 0))
        mode_box = ttk.Combobox(
            content,
            textvariable=self.mode_var,
            values=tuple(mode.value for mode in TransitionMode),
            state="readonly",
        )
        mode_box.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        mode_box.bind("<<ComboboxSelected>>", self._refresh_plan)
        row += 1

        ttk.Separator(content).grid(row=row, column=0, columnspan=2, sticky="ew", pady=12)
        row += 1

        ttk.Label(content, text="Iterationen").grid(row=row, column=0, sticky="w")
        iterations = ttk.Spinbox(
            content, from_=20, to=100000, increment=20, textvariable=self.iterations_var
        )
        iterations.grid(row=row, column=1, sticky="ew", padx=(10, 0))
        iterations.bind("<FocusOut>", self._refresh_plan)
        row += 1

        ttk.Label(content, text="Referenzbits").grid(row=row, column=0, sticky="w", pady=(8, 0))
        bits = ttk.Combobox(
            content,
            textvariable=self.reference_bits_var,
            values=(128, 192, 256, 384, 512, 768, 1024),
            state="readonly",
        )
        bits.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        bits.bind("<<ComboboxSelected>>", self._refresh_plan)
        row += 1

        ttk.Label(content, text="Zielpalette").grid(row=row, column=0, sticky="w", pady=(8, 0))
        palette = ttk.Combobox(
            content,
            textvariable=self.palette_var,
            values=palette_names(),
            state="readonly",
        )
        palette.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        palette.bind("<<ComboboxSelected>>", self._refresh_plan)
        row += 1

        ttk.Label(content, text="Farbzyklen").grid(row=row, column=0, sticky="w", pady=(8, 0))
        cycles = ttk.Entry(content, textvariable=self.cycles_var)
        cycles.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        cycles.bind("<FocusOut>", self._refresh_plan)
        row += 1

        ttk.Label(content, text="Palettenwechsel").grid(row=row, column=0, sticky="w", pady=(8, 0))
        palette_mode = ttk.Combobox(
            content,
            textvariable=self.palette_transition_var,
            values=(
                PaletteTransition.HOLD.value,
                PaletteTransition.BLEND.value,
                PaletteTransition.CUT.value,
            ),
            state="readonly",
        )
        palette_mode.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=(8, 0))
        palette_mode.bind("<<ComboboxSelected>>", self._refresh_plan)
        row += 1

        summary = ttk.LabelFrame(content, text="Geplanter Abschnitt", padding=12)
        summary.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        ttk.Label(
            summary,
            textvariable=self.summary_var,
            font=("TkDefaultFont", 10, "bold"),
            wraplength=570,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X)
        ttk.Label(
            summary,
            textvariable=self.details_var,
            wraplength=570,
            justify=tk.LEFT,
            foreground="#555",
        ).pack(anchor="w", fill=tk.X, pady=(8, 0))
        row += 1

        ttk.Label(
            content,
            text=(
                "Der Rechtsklick erzeugt keinen separaten Endlosflug mehr. "
                "Er erweitert den gemeinsamen Flugplan; alle erzeugten Keyframes "
                "bleiben im Flugplan-Editor editierbar."
            ),
            wraplength=610,
            justify=tk.LEFT,
            foreground="#555",
        ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        row += 1

        buttons = ttk.Frame(content)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        ttk.Button(buttons, text="Abbrechen", command=self.destroy).pack(side=tk.RIGHT)
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
        ttk.Button(buttons, text="Vorschau aktualisieren", command=self._refresh_plan).pack(side=tk.LEFT)

    def _values(self) -> FreeTargetValues:
        return FreeTargetValues(
            self.center_x_var.get().strip(),
            self.center_y_var.get().strip(),
            self.width_var.get().strip(),
            int(self.iterations_var.get().strip()),
            int(self.reference_bits_var.get().strip()),
            self.palette_var.get().strip(),
            self.cycles_var.get().strip(),
        )

    def _refresh_plan(self, _event: tk.Event | None = None) -> None:
        try:
            plan = self._build_plan(
                self._values(),
                TransitionMode(self.mode_var.get()),
                PaletteTransition(self.palette_transition_var.get()),
            )
        except Exception as exc:
            self._current_plan = None
            self.summary_var.set(f"Planung nicht möglich: {exc}")
            self.details_var.set("")
            self.add_button.state(("disabled",))
            self.play_button.state(("disabled",))
            return

        self._current_plan = plan
        requested = "" if plan.requested_mode is plan.mode else f" (auto → {plan.mode.value})"
        profile = plan.render_cues[-1].profile
        self.summary_var.set(f"{plan.summary}{requested}")
        self.details_var.set(
            f"Ankunft bei {plan.arrival_time_text} s; Ende bei {plan.end_time_text} s. "
            f"Zielprofil: {profile.max_iterations} Iterationen, "
            f"{profile.reference_bits} Bit, Palette {profile.palette}."
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
            messagebox.showerror("Flugziel", str(exc), parent=self)
            return
        self.destroy()
