from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .models import FractalKind, Precision, RenderMode
from .palettes import palette_names
from .surface_lighting import (
    CUSTOM_SURFACE_LIGHTING_PRESET,
    surface_lighting_preset_names,
)


def build_ui(app) -> None:
    outer = ttk.Panedwindow(app.root, orient=tk.HORIZONTAL)
    outer.pack(fill=tk.BOTH, expand=True)

    controls = ttk.Frame(outer, padding=10)
    viewer = ttk.Frame(outer)
    outer.add(controls, weight=0)
    outer.add(viewer, weight=1)

    app.canvas = tk.Canvas(viewer, background="#080808", highlightthickness=0)
    app.canvas.pack(fill=tk.BOTH, expand=True)

    app.fractal_var = tk.StringVar(value=FractalKind.MANDELBROT.value)
    app.backend_var = tk.StringVar(value="auto")
    app.precision_var = tk.StringVar(value=Precision.FLOAT32.value)
    app.render_mode_var = tk.StringVar(value=RenderMode.AUTO.value)
    app.reference_bits_var = tk.IntVar(value=256)
    app.palette_var = tk.StringVar(value="inferno")
    app.iterations_var = tk.IntVar(value=400)
    app.cycles_var = tk.DoubleVar(value=1.0)
    app.surface_lighting_preset_var = tk.StringVar(
        value=CUSTOM_SURFACE_LIGHTING_PRESET
    )
    app.surface_lighting_enabled_var = tk.BooleanVar(value=False)
    app.surface_lighting_strength_var = tk.DoubleVar(value=1.5)
    app.surface_lighting_azimuth_var = tk.DoubleVar(value=315.0)
    app.surface_lighting_elevation_var = tk.DoubleVar(value=45.0)
    app.julia_real_var = tk.DoubleVar(value=-0.8)
    app.julia_imag_var = tk.DoubleVar(value=0.156)
    app.exponent_var = tk.IntVar(value=3)
    default_scale = 1.0 if app.cuda_status.available else 0.75
    app.preview_scale_var = tk.DoubleVar(value=default_scale)
    app.flight_scale_var = tk.DoubleVar(value=default_scale)
    app.flight_rate_var = tk.DoubleVar(value=1.035)
    app.status_var = tk.StringVar(value="Initialisierung …")
    app.position_var = tk.StringVar(value="")

    row = 0
    ttk.Label(controls, text="Fraktal", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        controls,
        textvariable=app.fractal_var,
        values=[f.value for f in FractalKind],
        state="readonly",
        width=20,
    ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
    row += 1

    ttk.Label(controls, text="Backend").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        controls,
        textvariable=app.backend_var,
        values=("auto", "cpu", "cuda"),
        state="readonly",
    ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
    row += 1

    ttk.Label(controls, text="Präzision").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        controls,
        textvariable=app.precision_var,
        values=[p.value for p in Precision],
        state="readonly",
    ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
    row += 1

    ttk.Label(controls, text="Berechnungsmodus").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        controls,
        textvariable=app.render_mode_var,
        values=[m.value for m in RenderMode],
        state="readonly",
    ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
    row += 1

    ttk.Label(controls, text="Referenzpräzision (Bits)").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        controls,
        textvariable=app.reference_bits_var,
        values=(128, 192, 256, 384, 512, 768, 1024),
        state="readonly",
    ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
    row += 1

    ttk.Label(controls, text="Vorschau-Skalierung").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        controls,
        textvariable=app.preview_scale_var,
        values=(0.5, 0.75, 1.0),
        state="readonly",
    ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
    row += 1

    ttk.Label(controls, text="Iterationen").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Spinbox(controls, from_=20, to=100000, textvariable=app.iterations_var, increment=20).grid(
        row=row, column=0, sticky="ew", pady=(2, 8)
    )
    row += 1

    ttk.Label(controls, text="Palette").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Combobox(
        controls,
        textvariable=app.palette_var,
        values=palette_names(),
        state="readonly",
    ).grid(row=row, column=0, sticky="ew", pady=(2, 8))
    row += 1

    ttk.Label(controls, text="Farbzyklen").grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Scale(controls, from_=0.25, to=8.0, variable=app.cycles_var, orient=tk.HORIZONTAL).grid(
        row=row, column=0, sticky="ew", pady=(2, 8)
    )
    row += 1

    lighting = ttk.LabelFrame(controls, text="2.5D-Beleuchtung", padding=6)
    lighting.grid(row=row, column=0, sticky="ew", pady=6)
    ttk.Checkbutton(
        lighting,
        text="Reliefbeleuchtung aktiv",
        variable=app.surface_lighting_enabled_var,
    ).grid(row=0, column=0, columnspan=3, sticky="w")
    ttk.Label(lighting, text="Preset").grid(row=1, column=0, sticky="w")
    app.surface_lighting_preset_combo = ttk.Combobox(
        lighting,
        textvariable=app.surface_lighting_preset_var,
        values=surface_lighting_preset_names(),
        state="readonly",
        width=14,
    )
    app.surface_lighting_preset_combo.grid(
        row=1, column=1, columnspan=2, sticky="ew", padx=(6, 0)
    )
    app.surface_lighting_preset_combo.bind(
        "<<ComboboxSelected>>", app.apply_surface_lighting_preset
    )
    ttk.Label(lighting, text="Stärke").grid(row=2, column=0, sticky="w")
    strength = ttk.Spinbox(
        lighting,
        from_=0.0,
        to=8.0,
        increment=0.1,
        textvariable=app.surface_lighting_strength_var,
        width=7,
    )
    strength.grid(row=2, column=1, sticky="ew", padx=(6, 0))
    ttk.Label(lighting, text="Azimut").grid(row=3, column=0, sticky="w")
    azimuth = ttk.Spinbox(
        lighting,
        from_=0.0,
        to=360.0,
        increment=5.0,
        textvariable=app.surface_lighting_azimuth_var,
        width=7,
    )
    azimuth.grid(row=3, column=1, sticky="ew", padx=(6, 0))
    ttk.Label(lighting, text="Höhe").grid(row=4, column=0, sticky="w")
    elevation = ttk.Spinbox(
        lighting,
        from_=1.0,
        to=90.0,
        increment=1.0,
        textvariable=app.surface_lighting_elevation_var,
        width=7,
    )
    elevation.grid(row=4, column=1, sticky="ew", padx=(6, 0))
    ttk.Label(lighting, text="°", foreground="#555").grid(row=3, column=2, sticky="w")
    ttk.Label(lighting, text="°", foreground="#555").grid(row=4, column=2, sticky="w")
    lighting.columnconfigure(1, weight=1)

    lighting_controls = (strength, azimuth, elevation)

    def update_lighting_controls(*_args) -> None:
        state = "normal" if app.surface_lighting_enabled_var.get() else "disabled"
        for control in lighting_controls:
            control.configure(state=state)

    app.surface_lighting_enabled_var.trace_add("write", update_lighting_controls)
    for variable in (
        app.surface_lighting_enabled_var,
        app.surface_lighting_strength_var,
        app.surface_lighting_azimuth_var,
        app.surface_lighting_elevation_var,
    ):
        variable.trace_add("write", app._refresh_surface_lighting_preset)
    update_lighting_controls()
    app._refresh_surface_lighting_preset()
    row += 1

    julia = ttk.LabelFrame(controls, text="Julia / Multibrot", padding=6)
    julia.grid(row=row, column=0, sticky="ew", pady=6)
    ttk.Label(julia, text="c real").grid(row=0, column=0, sticky="w")
    ttk.Entry(julia, textvariable=app.julia_real_var, width=12).grid(row=0, column=1)
    ttk.Label(julia, text="c imag").grid(row=1, column=0, sticky="w")
    ttk.Entry(julia, textvariable=app.julia_imag_var, width=12).grid(row=1, column=1)
    ttk.Label(julia, text="Exponent").grid(row=2, column=0, sticky="w")
    ttk.Spinbox(julia, from_=2, to=8, textvariable=app.exponent_var, width=10).grid(row=2, column=1)
    row += 1

    flight = ttk.LabelFrame(controls, text="Flugplan-Vorschau", padding=6)
    flight.grid(row=row, column=0, sticky="ew", pady=6)
    ttk.Label(flight, text="Render-Skalierung").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        flight,
        textvariable=app.flight_scale_var,
        values=(0.5, 0.75, 1.0),
        state="readonly",
        width=8,
    ).grid(row=0, column=1)
    ttk.Label(
        flight,
        text="Rechtsklick fügt ein Ziel zum Flugplan hinzu",
        foreground="#555",
        wraplength=210,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
    # Compatibility attributes for the retired instant-flight controller.  They
    # stay ungridded so older programmatic integrations fail gracefully while
    # the normal UI exposes only the unified flight-plan workflow.
    app.flight_button = ttk.Button(flight, text="Explorationsflug", command=app.toggle_flight)
    row += 1

    buttons = ttk.Frame(controls)
    buttons.grid(row=row, column=0, sticky="ew", pady=(8, 4))
    ttk.Button(buttons, text="Rendern", command=app.request_render).pack(side=tk.LEFT, expand=True, fill=tk.X)
    ttk.Button(buttons, text="Reset", command=app.reset_view).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
    row += 1
    ttk.Button(controls, text="PNG exportieren", command=app.export_png).grid(row=row, column=0, sticky="ew")
    row += 1
    ttk.Button(controls, text="GPU-Diagnose", command=app.show_gpu_diagnostics).grid(
        row=row, column=0, sticky="ew", pady=(4, 0)
    )
    row += 1

    ttk.Separator(controls).grid(row=row, column=0, sticky="ew", pady=8)
    row += 1
    ttk.Label(controls, textvariable=app.status_var, wraplength=240).grid(row=row, column=0, sticky="w")
    row += 1
    ttk.Label(controls, textvariable=app.position_var, wraplength=240).grid(row=row, column=0, sticky="w", pady=(4, 0))
    controls.columnconfigure(0, weight=1)

    for variable in (
        app.fractal_var,
        app.backend_var,
        app.precision_var,
        app.render_mode_var,
        app.reference_bits_var,
        app.palette_var,
        app.iterations_var,
        app.cycles_var,
        app.surface_lighting_enabled_var,
        app.surface_lighting_strength_var,
        app.surface_lighting_azimuth_var,
        app.surface_lighting_elevation_var,
        app.julia_real_var,
        app.julia_imag_var,
        app.exponent_var,
        app.preview_scale_var,
        app.flight_scale_var,
    ):
        variable.trace_add("write", lambda *_: app.root.after_idle(app.request_render))
