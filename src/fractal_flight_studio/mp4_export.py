from __future__ import annotations

from pathlib import Path

from .ffmpeg_mp4 import (
    CancellationCheck,
    Mp4ExportPlan,
    Mp4ExportProgress,
    Mp4ExportResult,
    Mp4ExportSettings,
    ProgressCallback,
    encode_mp4_frames,
)
from .flight_plan import FlightSource, flight_plan_fingerprint, surface_lighting_for
from .models import RenderRequest
from .palettes import PaletteInput
from .offline_render import OfflineFramePlan, render_offline_frames
from .surface_lighting import SurfaceLightingSettings
from .temporal_tonemapping import (
    TemporalToneSettings,
    ToneAnalysisCallback,
    ToneStability,
    analyze_offline_tone_states,
    offline_tone_scene_key,
)


def build_mp4_export_plan(offline_plan: OfflineFramePlan) -> Mp4ExportPlan:
    """Convert an inclusive offline sampling plan to constant-rate video frames."""

    if offline_plan.endpoint_appended:
        raise ValueError(
            "MP4 export requires a cadence-only offline plan; rebuild it with "
            "append_endpoint=False"
        )
    frame_count = offline_plan.frame_count - int(offline_plan.endpoint_included)
    if frame_count < 1:
        raise ValueError("MP4 export requires at least one cadence frame")
    return Mp4ExportPlan(
        width=offline_plan.width,
        height=offline_plan.height,
        fps_numerator=offline_plan.fps_numerator,
        fps_denominator=offline_plan.fps_denominator,
        duration_text=offline_plan.duration_text,
        digits=offline_plan.digits,
        frame_count=frame_count,
        source_frame_count=offline_plan.frame_count,
    )


def export_path_to_mp4(
    source: FlightSource,
    request_template: RenderRequest,
    renderer,
    offline_plan: OfflineFramePlan,
    output_path: str | Path,
    settings: Mp4ExportSettings = Mp4ExportSettings(),
    *,
    palette: PaletteInput = "inferno",
    cycles: float = 1.0,
    phase: float = 0.0,
    tone_mapping: str = "auto",
    temporal_tone: TemporalToneSettings = TemporalToneSettings(
        mode=ToneStability.PER_FRAME
    ),
    tone_analysis_progress: ToneAnalysisCallback | None = None,
    progress: ProgressCallback | None = None,
    cancellation_requested: CancellationCheck | None = None,
    surface_lighting: SurfaceLightingSettings | None = None,
) -> Mp4ExportResult:
    """Render a complete cadence-only path and stream it directly into FFmpeg."""

    mp4_plan = build_mp4_export_plan(offline_plan)
    lighting = surface_lighting_for(source, surface_lighting)
    tone_states = None
    tone_state_locked = False
    tone_scene_key = None
    if temporal_tone.mode is ToneStability.TEMPORAL and tone_mapping != "linear":
        tone_states = analyze_offline_tone_states(
            source,
            request_template,
            renderer,
            offline_plan,
            stop_index=mp4_plan.frame_count,
            settings=temporal_tone,
            palette=palette,
            cycles=cycles,
            phase=phase,
            tone_mapping=tone_mapping,
            progress=tone_analysis_progress,
            cancellation_requested=cancellation_requested,
            surface_lighting=lighting,
        )
        tone_state_locked = any(state is not None for state in tone_states)
        if tone_state_locked:
            tone_scene_key = (
                offline_tone_scene_key(
                    request_template,
                    tone_mapping,
                    palette,
                    cycles,
                    phase,
                ),
                flight_plan_fingerprint(source),
                lighting,
            )

    frames = render_offline_frames(
        source,
        request_template,
        renderer,
        offline_plan,
        stop_index=mp4_plan.frame_count,
        palette=palette,
        cycles=cycles,
        phase=phase,
        tone_mapping=tone_mapping,
        tone_states=tone_states,
        tone_scene_key=tone_scene_key,
        tone_state_locked=tone_state_locked,
        surface_lighting=lighting,
    )
    return encode_mp4_frames(
        frames,
        mp4_plan,
        output_path,
        settings,
        progress=progress,
        cancellation_requested=cancellation_requested,
    )
