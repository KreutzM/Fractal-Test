from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

from .flight_plan import FlightPlanDocument
from .flight_plan_playback import (
    FlightPlanPlaybackController,
    PlaybackSample,
    PlaybackState,
)


SampleListener = Callable[[PlaybackSample, str], None]
RenderBusy = Callable[[], bool]
RenderRequest = Callable[[], None]
KeyframeTimes = Callable[[], Sequence[float]]


class FlightPlanPlaybackPanel(ttk.Frame):
    """Tk playback controls around the wall-clock flight-plan controller.

    The panel owns scheduling, seeking and render-request coalescing. The host
    application remains responsible for applying an evaluated sample to its
    camera/render state and for actually starting a render.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_sample: SampleListener,
        render_busy: RenderBusy,
        request_render: RenderRequest,
        keyframe_times: KeyframeTimes,
    ) -> None:
        super().__init__(parent, padding=(8, 4))
        self.controller = FlightPlanPlaybackController()
        self._on_sample = on_sample
        self._render_busy = render_busy
        self._request_render = request_render
        self._keyframe_times = keyframe_times
        self._after_id: str | None = None
        self._render_pending = False
        self._updating_ui = False

        self.time_var = tk.DoubleVar(value=0.0)
        self.time_label_var = tk.StringVar(value="0,00 / 0,00 s")
        self.rate_var = tk.StringVar(value="1×")
        self._build_ui()
        self._update_controls()

    @property
    def loaded(self) -> bool:
        return self.controller.loaded

    @property
    def playing(self) -> bool:
        return self.controller.playing

    @property
    def paused(self) -> bool:
        return self.controller.paused

    @property
    def state(self) -> PlaybackState:
        return self.controller.state

    @property
    def playhead_seconds(self) -> float:
        return self.controller.playhead_seconds

    @property
    def duration_seconds(self) -> float:
        return self.controller.duration_seconds

    def load(
        self,
        document: FlightPlanDocument | None,
        *,
        playhead_seconds: float = 0.0,
    ) -> None:
        self._cancel_tick()
        self._render_pending = False
        if document is None:
            self.controller.unload()
        else:
            self.controller.load(document, playhead_seconds=playhead_seconds)
        self._update_controls()

    def play(self) -> PlaybackSample:
        sample = self.controller.play()
        self._emit(sample, "Flugplan-Wiedergabe")
        self._schedule_tick()
        return sample

    def pause(self, *, request_render: bool = True) -> PlaybackSample | None:
        if not self.loaded:
            return None
        self._cancel_tick()
        sample = self.controller.pause() if self.playing else self.controller.sample()
        self._emit(sample, "Flugplan pausiert", request_render=request_render)
        return sample

    def stop(self, *, request_render: bool = True) -> PlaybackSample | None:
        if not self.loaded:
            return None
        self._cancel_tick()
        sample = self.controller.stop(reset=True)
        self._emit(sample, "Flugplan gestoppt", request_render=request_render)
        return sample

    def preview(self, time_seconds: float) -> PlaybackSample:
        if self.playing:
            self.controller.pause()
        self._cancel_tick()
        sample = self.controller.seek(time_seconds)
        self._emit(sample, "Flugplan-Vorschau")
        return sample

    def seek(self, time_seconds: float) -> PlaybackSample:
        sample = self.controller.seek(time_seconds)
        self._emit(sample, "Flugplan-Position")
        return sample

    def interrupt(self) -> None:
        if not self.loaded:
            return
        self._cancel_tick()
        self._render_pending = False
        if self.playing:
            sample = self.controller.pause()
            self._emit(sample, "Flugplan pausiert", request_render=False)
        else:
            self._update_controls()

    def render_completed(self) -> None:
        """Start one newest-frame render after an older frame completed."""

        if self._render_pending and not self._render_busy():
            self._render_pending = False
            self._request_render()

    def close(self) -> None:
        self._cancel_tick()
        self._render_pending = False

    def _build_ui(self) -> None:
        ttk.Label(self, text="Flugplan-Wiedergabe:").pack(side=tk.LEFT)
        self.start_button = ttk.Button(self, text="|◀", width=4, command=self._seek_start)
        self.start_button.pack(side=tk.LEFT, padx=(8, 2))
        self.previous_button = ttk.Button(
            self, text="◀ Keyframe", command=self._seek_previous_keyframe
        )
        self.previous_button.pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(self, text="▶ Abspielen", command=self.play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        self.pause_button = ttk.Button(self, text="⏸ Pause", command=self.pause)
        self.pause_button.pack(side=tk.LEFT, padx=2)
        self.stop_button = ttk.Button(self, text="■ Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=2)
        self.next_button = ttk.Button(
            self, text="Keyframe ▶", command=self._seek_next_keyframe
        )
        self.next_button.pack(side=tk.LEFT, padx=(2, 8))

        self.scale = ttk.Scale(
            self,
            from_=0.0,
            to=1.0,
            variable=self.time_var,
            command=self._seek_from_scale,
        )
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(self, textvariable=self.time_label_var, width=17).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        self.rate_box = ttk.Combobox(
            self,
            textvariable=self.rate_var,
            values=("0,5×", "1×", "2×"),
            state="readonly",
            width=6,
        )
        self.rate_box.pack(side=tk.LEFT)
        self.rate_box.bind("<<ComboboxSelected>>", self._change_rate)

    @staticmethod
    def _format_seconds(value: float) -> str:
        return f"{value:.2f}".replace(".", ",")

    def _update_controls(self, sample: PlaybackSample | None = None) -> None:
        duration = self.duration_seconds
        playhead = self.playhead_seconds if sample is None else sample.playhead_seconds
        state = self.state if sample is None else sample.state
        loaded = self.loaded

        self._updating_ui = True
        try:
            self.scale.configure(to=max(duration, 1.0))
            self.time_var.set(playhead)
            self.time_label_var.set(
                f"{self._format_seconds(playhead)} / {self._format_seconds(duration)} s"
            )
        finally:
            self._updating_ui = False

        common_state = ("!disabled",) if loaded else ("disabled",)
        for widget in (
            self.start_button,
            self.previous_button,
            self.stop_button,
            self.next_button,
            self.scale,
        ):
            widget.state(common_state)
        self.play_button.state(
            ("!disabled",)
            if loaded and state is not PlaybackState.PLAYING
            else ("disabled",)
        )
        self.pause_button.state(
            ("!disabled",) if state is PlaybackState.PLAYING else ("disabled",)
        )
        self.rate_box.configure(state="readonly" if loaded else "disabled")
        self.play_button.configure(
            text="▶ Fortsetzen" if state is PlaybackState.PAUSED else "▶ Abspielen"
        )

    def _emit(
        self,
        sample: PlaybackSample,
        status_prefix: str,
        *,
        request_render: bool = True,
    ) -> None:
        self._update_controls(sample)
        self._on_sample(sample, status_prefix)
        if not request_render:
            self._render_pending = False
            return
        if self._render_busy():
            # Keep only the newest playhead sample. The host calls
            # render_completed() after the active frame is displayed.
            self._render_pending = True
        else:
            self._render_pending = False
            self._request_render()

    def _schedule_tick(self) -> None:
        if self._after_id is None and self.playing:
            self._after_id = self.after(15, self._tick)

    def _cancel_tick(self) -> None:
        after_id = self._after_id
        self._after_id = None
        if after_id is not None:
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass

    def _tick(self) -> None:
        self._after_id = None
        if not self.playing:
            self._update_controls()
            return
        sample = self.controller.sample()
        self._emit(sample, "Flugplan-Wiedergabe")
        if self.playing:
            self._schedule_tick()

    def _seek_from_scale(self, value: str) -> None:
        if self._updating_ui or not self.loaded:
            return
        try:
            self.seek(float(value))
        except ValueError:
            return

    def _seek_start(self) -> None:
        if self.loaded:
            self.seek(0.0)

    def _seek_previous_keyframe(self) -> None:
        if not self.loaded:
            return
        current = self.playhead_seconds
        candidates = [value for value in self._keyframe_times() if value < current - 1e-9]
        self.seek(candidates[-1] if candidates else 0.0)

    def _seek_next_keyframe(self) -> None:
        if not self.loaded:
            return
        current = self.playhead_seconds
        candidates = [value for value in self._keyframe_times() if value > current + 1e-9]
        self.seek(candidates[0] if candidates else self.duration_seconds)

    def _change_rate(self, _event: tk.Event | None = None) -> None:
        text = self.rate_var.get().replace("×", "").replace(",", ".")
        sample = self.controller.set_rate(float(text))
        self._update_controls(sample)
