# Architecture

Fractal Flight Studio separates numerical rendering, application state and Tk presentation so that interactive preview, path editing and offline video rendering can reuse the same core behavior.

## Main components

```text
Tk GUI (`app.py`, `app_ui.py`, `flight_app.py`)
    ├── CameraState
    ├── CameraPath / FlightKeyframe
    ├── CameraPathDraft / timeline editor
    ├── path preflight
    ├── offline frame planning / rendering
    ├── direct FFmpeg MP4 encoding
    ├── preflight / export workflow controller
    ├── RenderController
    ├── FlightController
    ├── target catalog
    └── renderer selection
            ├── CPU renderer
            ├── CUDA renderer
            └── adaptive precision / perturbation
```

### `CameraState`

`CameraState` is the canonical camera representation. It stores the center and view width as decimal text and performs navigation with `mpmath` precision selected from the configured reference bits.

The floating-point `Viewport` inside `RenderRequest` is only a compatibility proxy for direct FP32/FP64 rendering. Deep-zoom setup must continue to use the exact text fields carried by the request.

Responsibilities:

- preserve exact center and width text;
- convert to high-precision values for pan, zoom and flight movement;
- produce a bounded float proxy for direct render paths;
- construct camera snapshots from completed render requests;
- define fractal-specific reset views.

### `CameraPath` and `FlightKeyframe`

`FlightKeyframe` stores an exact decimal timeline position, a `CameraState`, the easing curve and the center-interpolation mode for its outgoing segment. `CameraPath` validates an immutable sequence of at least two keyframes that starts at zero seconds and has strictly increasing times.

Path evaluation is independent of Tk, rendering speed and wall-clock time:

- X and Y are interpolated with the path's configured `mpmath` precision;
- `linear` center interpolation follows a straight X/Y path for deliberate pans;
- `focus` center interpolation ties center movement to the current zoom scale, keeping the next keyframe destination approximately fixed on screen during large zoom changes;
- view width is interpolated in logarithmic space for visually uniform zoom speed;
- `linear`, `smoothstep` and `smootherstep` easing are available per segment;
- evaluation before or after the timeline returns the exact endpoint camera;
- camera text is never reduced to absolute FP64 values.

This component is the common camera source for a timeline editor, low-resolution preflight and deterministic offline frame renderer.

### Timeline and keyframe editing

`path_editor.py` provides the Tk-independent `CameraPathDraft`. A draft preserves exact decimal camera text and exact timeline positions while allowing temporary states that are not yet exportable, such as a single keyframe. Mutations return new draft values, automatically keep keyframes in timeline order and reject duplicate times. Only `build_path()` crosses the boundary into an immutable, fully validated `CameraPath`.

`timeline_editor.py` is a non-modal Tk adapter over that model. It can capture the current camera, copy a packaged catalog target, edit time/easing/center-interpolation/camera text, remove keyframes and preview either a selected keyframe or an interpolated timeline position. New GUI segments default to `focus`; `linear` remains available for explicit X/Y camera moves. The window reports valid paths back to the application through a callback; it does not own rendering, preflight or export.

### Path preflight

`preflight.py` performs a bounded diagnostic render pass over a `CameraPath` without depending on Tk or wall-clock scheduling.

- The path is sampled at exact decimal times and always includes both endpoints.
- A configurable sample cap evenly decimates very long paths without first expanding an unbounded time list.
- Every sample derives a low-resolution `RenderRequest` from an immutable request template while retaining exact camera text.
- Tone state is local to the preflight run and is not stored in widgets or the renderer's implicit interactive session.
- Renderer pixel-grid metadata, explicit grid exhaustion, runtime failures and the shared RGB visual-quality classifier produce structured issues.
- The result is an immutable `PreflightReport`; RGB frames are deliberately not retained.

This layer diagnoses a path before expensive offline rendering. It does not own timeline editing, full-resolution frame production or video encoding.

### Offline frame rendering

`offline_render.py` turns a validated `CameraPath` into deterministic full-resolution frame jobs and rendered RGB frames without depending on Tk, timers or output files.

- Frame cadence is represented exactly by an integer numerator and denominator, including rates such as `30000/1001`.
- `OfflineFramePlan` stores only the duration, cadence and frame count; individual times are calculated lazily by frame index instead of expanding a large timeline.
- A configurable frame cap rejects accidental oversized jobs before rendering begins.
- An exact endpoint may be appended when the path duration is not on the regular cadence. Consumers that require cadence-only output can disable this behavior.
- `iter_offline_frame_jobs` supports deterministic start/stop index ranges for resumable or chunked work.
- Every job derives an immutable `RenderRequest` from the template while retaining exact camera text.
- A single frame can be rendered independently by index. Automatic tone mapping is intentionally stateless at this stage so results do not depend on previously rendered frames or chunk boundaries.
- `OfflineFrame` owns a copied 8-bit RGB array and compact scalar metadata. The iterator does not retain earlier frames.
- Renderer failures are wrapped with the exact frame index and timeline time.

This layer produces frames only. File publication and video process ownership remain separate responsibilities. Its default remains stateless for random-access and chunked callers; an export session may explicitly supply a planned per-frame `ToneMapState` sequence.

### Temporal production tone mapping

`temporal_tonemapping.py` owns the video-only analysis and stabilization policy. It does not retain full-resolution RGB frames.

- a low-resolution analysis pass evaluates every exact cadence frame used by the MP4 plan;
- each analysis frame produces an independent robust automatic `ToneMapState`;
- frames without usable outside samples borrow neighboring states before smoothing;
- a deterministic forward/backward exponential filter reduces parameter jumps without directional exposure lag;
- the final full-resolution pass receives one locked state per frame and performs no new tone analysis;
- CPU and CUDA renderers accept the same locked-state contract, and the CUDA final pass avoids the tone-sample readback;
- cancellation and progress cover both analysis and final rendering.

The planned states belong to one immutable export session. They are not stored in Tk widgets, FFmpeg, or a renderer's implicit interactive session, and they do not change the stateless random-access default of `offline_render.py`.

### Direct FFmpeg MP4 export

`ffmpeg_mp4.py` owns FFmpeg discovery, version probing and the lifetime of one raw-RGB-to-MP4 encoder process. `mp4_export.py` adapts a complete offline frame plan and connects its frame iterator to that process.

- RGB24 frame bytes are streamed directly to FFmpeg through binary standard input; no PNG sequence is required.
- The exact numerator/denominator frame rate is passed to FFmpeg without converting it to a binary floating-point rate.
- The initial production profile is software H.264 (`libx264`) with CRF quality control and `yuv420p` compatibility output.
- `yuv420p` exports reject odd dimensions before the encoder starts.
- The inclusive offline sampling plan is converted to constant-rate video semantics: an explicitly appended endpoint is rejected, while an endpoint exactly on cadence is omitted so a one-second, 30-fps path produces 30 rather than 31 video frames.
- Frame index, exact frame time, RGB shape and byte depth are validated before each write.
- Encoder standard error is drained concurrently into a bounded diagnostic buffer so long jobs cannot deadlock on a full pipe.
- Cancellation, broken pipes, renderer failures and encoder failures terminate the child process and remove the incomplete file.
- FFmpeg writes to a unique temporary file in the destination directory. The final `.mp4` path is replaced atomically only after a successful encoder exit.
- Progress callbacks report completed frames and streamed bytes without moving state into Tk widgets.

This layer does not own camera interpolation, preflight decisions or timeline editing. `mp4_export.py` may request a temporal tone plan before it hands the final RGB stream to the encoder; FFmpeg itself remains unaware of tone state. Hardware encoding, PNG checkpoint sequences and higher-bit-depth video remain later extensions.

### Preflight and export workflow

`export_controller.py` owns one background worker, cooperative cancellation and thread-safe progress snapshots for FFmpeg probing, path preflight and MP4 export. `export_dialog.py` is a Tk adapter over that controller.

- the dialog captures an immutable render request and exact `CameraPath` before each job;
- interactive preview submissions are paused while the export worker owns the shared renderer;
- preflight reports progress per completed sample and can be cancelled between samples;
- stabilized MP4 jobs expose one combined progress range for low-resolution tone analysis and final rendering/encoding without updating Tk from a worker thread;
- a successful preflight is fingerprinted against the exact path, renderer and render settings; changing relevant settings requires another preflight before export;
- constant-rate frame plans use `append_endpoint=False` and show the planned frame count before work starts;
- FFmpeg discovery, output selection, overwrite policy, codec preset, CRF and per-frame versus temporally stabilized tone mapping are explicit user choices;
- closing the application requests cancellation and shuts down the workflow worker without blocking Tk.

The controller does not own widgets and the dialog does not render frames itself. Project persistence and hardware encoders remain separate extensions.

### `RenderController`

`RenderController` owns the interactive single-render worker, request generations and invalidation coalescing. Tk remains responsible only for scheduling a non-blocking poll and displaying the completed image.

A new UI change invalidates the current generation. If a render is already active, no second worker job is queued. When the active job completes, the GUI submits one render for the newest state.

This boundary remains specific to interactive rendering. Offline rendering follows its deterministic frame plan directly and does not inherit GUI invalidation or coalescing behavior.

### `FlightController`

`FlightController` owns:

- the exact flight target;
- running/stopped state;
- the logarithmic zoom step toward the target;
- numerical final-frame state;
- the last accepted camera and RGB frame.

The Tk adapter decides when to schedule the next step. `flight_app.py` applies the numerical and visual quality gates to completed frames and asks the controller to accept or reject them.

### GUI adapters

`app_ui.py` builds the control widgets. `app.py` translates widget values into `RenderRequest`, displays images and formats status text. It must not implement deep-zoom arithmetic or worker lifecycle state directly.

`flight_app.py` adds the curated target strip and the flight-result quality gate. It should remain an adapter over the controllers rather than a second application state store.

### Renderers

Renderer modules remain responsible for numerical values, perturbation, CUDA execution, tone mapping and RGB production. They receive immutable `RenderRequest` values and must not depend on Tk or controller classes.

### Surface-lighting reference core

`surface_lighting.py` defines the renderer-independent reference behavior for optional 2.5D relief shading. It consumes an already normalized floating-point value field, the matching inside mask and an RGB8 frame. Central differences in the image interior and one-sided border differences form a screen-space normal field; a directional ambient-plus-Lambert model then modulates only outside pixels. Inside pixels remain byte-identical, all inputs remain immutable and disabled lighting returns a byte-identical copy.

The reference core does not own fractal iteration, tone mapping, palette selection, GUI state, flight-plan persistence or CUDA routing. The generic host `Renderer.render_frame` path accepts optional immutable `SurfaceLightingSettings` and applies the reference behavior only after tone mapping and palette lookup. This integrates the feature into the CPU renderer without changing `RenderRequest`, renderer selection or any disabled frame bytes. The optimized CUDA frame path remains unchanged; a later CUDA implementation must preserve the same contract while keeping value, mask and RGB data on the device, with physical GPU parity and performance validation in that dedicated integration PR.

## Extension rules

Future PRs should follow these boundaries:

- target previews and search belong to separate UI components backed by the existing catalog;
- timeline windows edit immutable `CameraPathDraft` values and publish only validated `CameraPath` objects;
- preflight evaluates a path through `run_path_preflight` and reports diagnostics without retaining frame images;
- full-resolution work is planned with `build_offline_frame_plan`, evaluated with `iter_offline_frame_jobs` and rendered through the stateless offline-frame API;
- constant-rate MP4 export uses `build_mp4_export_plan` and `export_path_to_mp4`, consumes completed RGB frames and does not own camera interpolation;
- temporal tone analysis and zero-phase smoothing belong to `temporal_tonemapping.py`; locked states belong to one export session, not to widgets, FFmpeg or implicit renderer state;
- persistent projects serialize camera/keyframe text without reducing it to absolute `float` values.

## Testing strategy

- camera, path drafts, path interpolation, preflight, offline planning and controller behavior is tested without Tk;
- preflight workload planning and diagnostics use deterministic fake renderers;
- offline frame cadence, random-access jobs, RGB ownership and contextual failures use deterministic fake renderers;
- FFmpeg command construction, process cleanup, error propagation, atomic publication and cancellation use deterministic fake processes;
- export workflow configuration, fingerprints, two-phase progress and cooperative cancellation are tested without Tk;
- temporal tone planning is tested for determinism, missing-state handling, locked CPU/CUDA application and reduced adjacent parameter jumps;
- renderer precision and CPU/CUDA parity remain covered by numerical tests;
- the Xvfb smoke test verifies only the assembled GUI wiring;
- physical CUDA performance, native FFmpeg encoder availability and Windows packaging remain separate target-system checks.

## Target browser

`target_browser.py` owns target search, category filtering, packaged preview loading and the browser window. It receives callbacks for applying a target; it
does not mutate camera, renderer or flight state directly. Preview generation
is reproducible through `scripts/generate_target_thumbnails.py`.
