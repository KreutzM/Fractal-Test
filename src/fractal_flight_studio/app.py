from __future__ import annotations

from concurrent.futures import Future
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import mpmath as mp
from PIL import Image, ImageTk

from .app_ui import build_ui
from .camera import CameraState
from .deep_zoom import digits_for_bits, effective_direct_precision, minimum_safe_view_width
from .flight_controller import FlightController
from .gpu_info import inspect_cuda
from .models import FractalKind, Precision, RenderMode, RenderRequest
from .render_controller import RenderController
from .renderers import available_renderers, select_renderer
from .surface_lighting import (
    CUSTOM_SURFACE_LIGHTING_PRESET,
    SurfaceLightingSettings,
    surface_lighting_preset_for,
    surface_lighting_settings_for_preset,
)


class FractalStudioApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fractal Flight Studio")
        self.root.geometry("1260x820")
        self.root.minsize(900, 600)

        self.render_controller = RenderController()
        self.flight_controller = FlightController()
        self.last_rgb = None
        self.photo: ImageTk.PhotoImage | None = None
        self.drag_start: tuple[int, int] | None = None
        self.last_frame_time = time.perf_counter()
        self.cuda_status = inspect_cuda()

        self.camera = CameraState()
        lighting_defaults = SurfaceLightingSettings()
        if not hasattr(self, "_surface_lighting_ambient"):
            self._surface_lighting_ambient = lighting_defaults.ambient
        if not hasattr(self, "_surface_lighting_diffuse"):
            self._surface_lighting_diffuse = lighting_defaults.diffuse
        self._applying_surface_lighting_preset = False

        self._build_ui()
        self._bind_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self.request_render)

    def _build_ui(self) -> None:
        build_ui(self)

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda _event: self.request_render())
        self.canvas.bind("<ButtonPress-1>", self._drag_begin)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.canvas.bind("<Button-3>", self._set_flight_target)
        self.canvas.bind("<Motion>", self._show_coordinate)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_event(event, 1.35))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_event(event, 1 / 1.35))

    def _work_digits(self) -> int:
        return digits_for_bits(int(self.reference_bits_var.get()))

    def _format_precise(self, value: mp.mpf) -> str:
        return mp.nstr(value, n=self._work_digits())

    @staticmethod
    def _format_display(value: mp.mpf, digits: int = 14) -> str:
        return mp.nstr(value, n=digits)

    def _pixel_to_complex_hp(self, px: float, py: float) -> tuple[mp.mpf, mp.mpf]:
        return self.camera.pixel_to_complex(
            px,
            py,
            max(1, self.canvas.winfo_width()),
            max(1, self.canvas.winfo_height()),
            digits=self._work_digits(),
        )

    def _zoom_view_hp(self, px: float, py: float, factor: float) -> None:
        if factor <= 0:
            return
        self.camera = self.camera.zoom_at(
            px,
            py,
            max(1, self.canvas.winfo_width()),
            max(1, self.canvas.winfo_height()),
            factor,
            digits=self._work_digits(),
        )

    def _pan_view_hp(self, dx_pixels: float, dy_pixels: float) -> None:
        self.camera = self.camera.pan_pixels(
            dx_pixels,
            dy_pixels,
            max(1, self.canvas.winfo_width()),
            digits=self._work_digits(),
        )

    def _request(self, scale: float | None = None) -> RenderRequest:
        scale = self.preview_scale_var.get() if scale is None else scale
        canvas_width = max(64, self.canvas.winfo_width())
        canvas_height = max(64, self.canvas.winfo_height())
        width = max(64, int(canvas_width * scale))
        height = max(64, int(canvas_height * scale))
        proxy_viewport = self.camera.proxy_viewport(digits=self._work_digits())
        return RenderRequest(
            width=width,
            height=height,
            viewport=proxy_viewport,
            fractal=FractalKind(self.fractal_var.get()),
            max_iterations=int(self.iterations_var.get()),
            julia_c_real=float(self.julia_real_var.get()),
            julia_c_imag=float(self.julia_imag_var.get()),
            exponent=int(self.exponent_var.get()),
            precision=Precision(self.precision_var.get()),
            render_mode=RenderMode(self.render_mode_var.get()),
            reference_bits=int(self.reference_bits_var.get()),
            center_x_text=self.camera.center_x_text,
            center_y_text=self.camera.center_y_text,
            view_width_text=self.camera.view_width_text,
        )

    def _render_palette(self):
        return self.palette_var.get()

    def _render_cycles(self) -> float:
        return float(self.cycles_var.get())

    def _surface_lighting_settings(self) -> SurfaceLightingSettings:
        return SurfaceLightingSettings(
            enabled=bool(self.surface_lighting_enabled_var.get()),
            strength=float(self.surface_lighting_strength_var.get()),
            azimuth_degrees=float(self.surface_lighting_azimuth_var.get()),
            elevation_degrees=float(self.surface_lighting_elevation_var.get()),
            ambient=float(self._surface_lighting_ambient),
            diffuse=float(self._surface_lighting_diffuse),
        )

    def apply_surface_lighting_preset(self, _event=None) -> None:
        name = self.surface_lighting_preset_var.get()
        if name == CUSTOM_SURFACE_LIGHTING_PRESET:
            return
        settings = surface_lighting_settings_for_preset(name)
        previous_flight_guard = getattr(
            self, "_applying_flight_plan_settings", None
        )
        self._applying_surface_lighting_preset = True
        if previous_flight_guard is not None:
            self._applying_flight_plan_settings = True
        try:
            self._surface_lighting_ambient = settings.ambient
            self._surface_lighting_diffuse = settings.diffuse
            self.surface_lighting_enabled_var.set(settings.enabled)
            self.surface_lighting_strength_var.set(settings.strength)
            self.surface_lighting_azimuth_var.set(settings.azimuth_degrees)
            self.surface_lighting_elevation_var.set(settings.elevation_degrees)
        finally:
            self._applying_surface_lighting_preset = False
            if previous_flight_guard is not None:
                self._applying_flight_plan_settings = previous_flight_guard
        self._refresh_surface_lighting_preset()
        sync = getattr(self, "_sync_primary_flight_settings", None)
        if sync is not None:
            sync()

    def _refresh_surface_lighting_preset(self, *_args) -> None:
        if self._applying_surface_lighting_preset:
            return
        try:
            name = surface_lighting_preset_for(self._surface_lighting_settings())
        except (AttributeError, ValueError, tk.TclError):
            name = CUSTOM_SURFACE_LIGHTING_PRESET
        if self.surface_lighting_preset_var.get() != name:
            self._applying_surface_lighting_preset = True
            try:
                self.surface_lighting_preset_var.set(name)
            finally:
                self._applying_surface_lighting_preset = False

    def _interactive_render_scale(self) -> float:
        return (
            self.flight_scale_var.get()
            if self.flight_controller.running
            else self.preview_scale_var.get()
        )

    def request_render(self) -> None:
        self.render_controller.invalidate()
        if self.render_controller.busy:
            return
        try:
            request = self._request(self._interactive_render_scale())
            backend_name = self.backend_var.get()
            renderer = select_renderer(backend_name)
            palette = self._render_palette()
            cycles = self._render_cycles()
            surface_lighting = self._surface_lighting_settings()
        except Exception as exc:
            self.status_var.set(str(exc))
            return

        submission = self.render_controller.submit(
            lambda: renderer.render_frame(
                request,
                palette,
                cycles,
                0.0,
                surface_lighting=surface_lighting,
            )
        )
        if submission is None:
            return
        generation, future = submission
        self.status_var.set(
            f"Rendere mit {renderer.name} ({request.width}×{request.height}, {request.precision.value}, {request.render_mode.value}) …"
        )
        self.root.after(10, self._poll_render, future, generation, request)

    def _poll_render(self, future: Future, generation: int, request: RenderRequest) -> None:
        if future.done():
            self._finish_render(future, generation, request)
            return
        self.root.after(20, self._poll_render, future, generation, request)

    def _finish_render(self, future: Future, generation: int, request: RenderRequest) -> None:
        render_invalidated = self.render_controller.complete(generation)
        try:
            result = future.result()
            rgb = result.rgb
            self.last_rgb = rgb
            display_started = time.perf_counter()
            image = Image.fromarray(rgb, mode="RGB")
            canvas_size = (
                max(1, self.canvas.winfo_width()),
                max(1, self.canvas.winfo_height()),
            )
            if image.size != canvas_size:
                image = image.resize(canvas_size, Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image)
            if getattr(self, "canvas_image_id", None) is None:
                self.canvas_image_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
            else:
                self.canvas.itemconfigure(self.canvas_image_id, image=self.photo)
            display_seconds = time.perf_counter() - display_started
            total_seconds = result.elapsed_seconds + display_seconds
            fps = 1.0 / total_seconds if total_seconds > 0 else float("inf")
            device = result.details.get("device")
            device_line = f"\nGPU: {device}" if device else ""
            fallback_line = ""
            if result.backend == "cpu-numba" and self.backend_var.get() == "auto" and not self.cuda_status.available:
                fallback_line = f"\nAuto-Fallback: {self.cuda_status.reason}"
            optimized = "GPU-RGB" if result.details.get("optimized_frame_path") else "CPU-RGB"
            allocation = float(result.details.get("allocation_seconds", 0.0))
            reference_upload = float(result.details.get("reference_upload_seconds", 0.0))
            allocation_line = f"; Puffer {allocation * 1000:.1f} ms" if allocation > 0 else ""
            upload_line = f"; Ref {reference_upload * 1000:.1f} ms" if reference_upload > 0 else ""
            lighting_line = ""
            if result.details.get("surface_lighting_enabled"):
                lighting_seconds = float(
                    result.details.get("surface_lighting_seconds", 0.0)
                )
                lighting_line = f"; Licht {lighting_seconds * 1000:.1f} ms"
            render_mode = result.details.get("render_mode", "direct")
            requested_precision = request.precision.value
            if render_mode == "perturbation":
                effective_precision = Precision.FLOAT64.value
            else:
                effective_precision = effective_direct_precision(request).value
            precision_line = effective_precision
            if requested_precision != effective_precision:
                precision_line = f"{requested_precision}→{effective_precision}"
            ref_bits = int(result.details.get("reference_bits", 0) or 0)
            repair_line = ""
            if result.details.get("rebasing_enabled"):
                if "rebase_pixels" in result.details:
                    repair_line += f"; Rebases {int(result.details['rebase_pixels'])}"
                else:
                    repair_line += "; Rebasing aktiv"
            if result.details.get("glitch_detection_enabled"):
                if "glitch_pixels" in result.details:
                    repair_line += f"; Glitch-Reparaturen {int(result.details['glitch_pixels'])}"
                else:
                    repair_line += "; Glitch-Reparatur aktiv"
            reference_line = ""
            if render_mode == "perturbation":
                reference_line = "; Referenz wiederverwendet" if result.details.get("reference_reused") else "; Referenz neu"
            deep_line = (
                f"; Modus {render_mode}"
                + (f"; Präzision {precision_line}" if precision_line else "")
                + (f"; RefBits {ref_bits}" if ref_bits else "")
                + reference_line
                + repair_line
            )
            self.status_var.set(
                f"{result.backend}/{optimized}: Rechnen+Transfer {result.elapsed_seconds * 1000:.1f} ms; "
                f"Anzeige {display_seconds * 1000:.1f} ms; ca. {fps:.1f} FPS"
                f"{allocation_line}{upload_line}{lighting_line}{deep_line}"
                f"{device_line}{fallback_line}\n"
                f"Rendergröße: {rgb.shape[1]}×{rgb.shape[0]}; Ansichtsbreite: {self.camera.view_width_text}"
            )
        except Exception as exc:
            self.status_var.set(f"Renderfehler: {exc}")

        if render_invalidated:
            self.request_render()

    def _mousewheel(self, event: tk.Event) -> None:
        self._zoom_event(event, 1.35 if event.delta > 0 else 1 / 1.35)

    def _zoom_event(self, event: tk.Event, factor: float) -> None:
        self._zoom_view_hp(event.x, event.y, factor)
        self.request_render()

    def _drag_begin(self, event: tk.Event) -> None:
        self.drag_start = (event.x, event.y)

    def _drag_move(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self._pan_view_hp(dx, dy)
        self.drag_start = (event.x, event.y)
        self.request_render()

    def _drag_end(self, _event: tk.Event) -> None:
        self.drag_start = None

    def _set_flight_target(self, event: tk.Event) -> None:
        x, y = self._pixel_to_complex_hp(event.x, event.y)
        self.flight_controller.set_target(
            self._format_precise(x),
            self._format_precise(y),
        )
        self.position_var.set(
            f"Flugziel: {self._format_display(x, 16)}, {self._format_display(y, 16)}"
        )

    def _show_coordinate(self, event: tk.Event) -> None:
        x, y = self._pixel_to_complex_hp(event.x, event.y)
        target = " gesetzt" if self.flight_controller.target_text else " nicht gesetzt"
        self.position_var.set(
            f"Cursor: {self._format_display(x, 12)}, {self._format_display(y, 12)}\n"
            f"Flugziel:{target}"
        )

    def _stop_flight(self, message: str | None = None) -> None:
        numerical_limit = bool(message and "numerische Präzisionsgrenze" in message)
        self.flight_controller.stop(numerical_limit=numerical_limit)
        self.flight_button.configure(text="Flug starten")
        if message:
            self.position_var.set(message)

    def toggle_flight(self) -> None:
        if self.flight_controller.running:
            self._stop_flight()
            return
        if not self.flight_controller.start(self.camera, self.last_rgb):
            messagebox.showinfo("Flugziel", "Setze zuerst mit Rechtsklick ein Flugziel.")
            return
        self.flight_button.configure(text="Flug stoppen")
        self.last_frame_time = time.perf_counter()
        self._flight_step()

    def _flight_step(self) -> None:
        if not self.flight_controller.running or self.flight_controller.target_text is None:
            return
        try:
            rate = max(1.0001, float(self.flight_rate_var.get()))
        except ValueError:
            rate = 1.035
        request = self._request(self.flight_scale_var.get())
        minimum_width = minimum_safe_view_width(request)
        step = self.flight_controller.step(
            self.camera,
            zoom_rate=rate,
            minimum_width=minimum_width,
            digits=self._work_digits(),
        )
        self.camera = step.camera
        self.request_render()
        if step.reached_limit:
            self._stop_flight(
                "Flug automatisch gestoppt: numerische Präzisionsgrenze erreicht.\n"
                f"Mindestbreite: {mp.nstr(minimum_width, 8)}; "
                f"Referenz: {int(self.reference_bits_var.get())} Bit"
            )
            return
        self.root.after(33, self._flight_step)

    def reset_view(self) -> None:
        self.camera = CameraState.for_fractal(FractalKind(self.fractal_var.get()))
        self.request_render()

    def show_gpu_diagnostics(self) -> None:
        self.cuda_status = inspect_cuda()
        messagebox.showinfo("GPU-Diagnose", self.cuda_status.report())

    def export_png(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            renderer = select_renderer(self.backend_var.get())
            request = self._request(1.0)
            result = renderer.render_frame(
                request,
                self.palette_var.get(),
                float(self.cycles_var.get()),
                surface_lighting=self._surface_lighting_settings(),
            )
            Image.fromarray(result.rgb, mode="RGB").save(path)
            self.status_var.set(f"Exportiert: {path}")
        except Exception as exc:
            messagebox.showerror("Exportfehler", str(exc))

    def _on_close(self) -> None:
        self.flight_controller.stop()
        self.render_controller.shutdown()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = FractalStudioApp(root)
    backends = ", ".join(available_renderers())
    app.status_var.set(f"Verfügbare Backends: {backends}\n{app.cuda_status.summary}")
    root.mainloop()


if __name__ == "__main__":
    main()
