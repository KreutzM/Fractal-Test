# Changelog

## Unreleased

### 2.5D surface lighting

- Added compact GUI controls for enabling relief lighting and adjusting strength,
  azimuth and elevation.
- Added deterministic presets for soft, dramatic, side, rim and top lighting;
  manual changes remain available as a custom configuration.
- Applied the current lighting settings to interactive preview, flight-plan
  playback and PNG export while keeping the default disabled output unchanged.
- Added schema-3 persistence with backward-compatible schema-1/schema-2 migration.
- Synchronized loaded plans back into the GUI and reused one immutable lighting
  value for preflight, temporal tone analysis and final MP4 rendering.
- Exposed lighting time in the GUI status line and extended GUI smoke coverage.
- Corrected the relief model to derive normals from the tone-mapped height field,
  preserve flat-surface exposure and apply a calibrated screen-space slope scale;
  this fixes the previous mostly uniform color darkening at high iteration counts.
- Extended CPU/CUDA parity and physical RTX validation with a representative
  auto-tone Seahorse view that must show measurable directional relief.


## 0.10.0 — 2026-07-31

### CUDA Double-Single

- Added a guarded internal Double-Single tier for eligible Mandelbrot `auto` direct rendering after FP32 promotion and before perturbation.
- Added a guarded Double-Single perturbation-delta kernel while keeping the arbitrary-precision reference orbit on the CPU.
- Preserved explicit native FP64 direct and perturbation modes as reference and safety paths.
- Added conservative fallbacks for unsupported fractals, unsafe coordinate grids, FP32 exponent overflow or underflow, reference-orbit split loss and unsafe reference magnitudes.
- Added diagnostic metadata for selected arithmetic, Double-Single mode and fallback reason.

### Flight-plan workflow

- Added schema-2 scene and render tracks with exact decimal camera values, time-dependent quality cues and palette transitions.
- Added deterministic real-time playback, seeking, playback rates and slow-render coalescing.
- Added automatic `direct`, `bridge`, `overview` and `cut` transitions for catalog, browser and right-click targets.
- Unified interactive playback, preflight and MP4 export around the same flight-plan evaluation path.
- Added six directly loadable Mandelbrot example plans, including three extended flights lasting 3:30, 4:20 and 4:58.

### Validation

- Validated direct Double-Single production routing on an RTX 3060 with substantial speedups across exterior, interior, boundary and deep-zoom targets and a clean handoff to perturbation.
- Validated guarded Double-Single perturbation on an RTX 3060 across six eligible deep-zoom targets at 1.87× to 2.99× native-FP64 speed, with identical inside, glitch and rebase decisions.
- Confirmed the intentional exponent-range case falls back to native FP64 with the expected diagnostic reason.
- Expanded automated coverage to 265 tests and kept Windows/Ubuntu Python 3.11–3.13, GUI smoke, pan stability, wheel build and repository snapshot green.

## 0.9.0 — 2026-07-26

### Visual deep-zoom target browser

- Added a searchable target dialog with category filters, rendered previews,
  descriptions, exact coordinates, recommended settings and source links.
- Added a compressed atlas of 48×30 XPM previews for every curated target and a reproducible
  CPU thumbnail-generation script.
- Kept the compact quick selector and existing free right-click target workflow.

### Validation

- Added filtering, category and packaged-atlas regressions.
- Extended the Xvfb GUI smoke test to open and filter the visual browser.

## 0.8.0 — 2026-07-25

### Curated deep-zoom targets

- Added a packaged, schema-validated JSON catalog with ten curated Mandelbrot
  locations, exact text coordinates, initial view widths, descriptions, tags,
  source links and recommended render settings.
- Added compact controls that can set a catalog entry as the flight target or
  load its complete view while preserving the existing free right-click target.
- Applying a target also selects its recommended iteration count, reference
  precision and palette without converting deep coordinates to absolute floats.

### Validation

- Added catalog schema, exact-coordinate and GUI-application regressions.
- Extended the Xvfb GUI smoke test to exercise the compact target selector and
  exact flight-target application.
- Rendered every packaged target with its recommendations at low resolution and
  confirmed that all ten pass the existing visual-quality classifier.

## 0.7.2 — 2026-07-25

### Visual flight-quality stop

- Added a renderer-level pixel-grid quality check that measures whether
  neighbouring X/Y samples still map to distinct FP32/FP64 coordinates with a
  safety margin of at least four ULPs.
- Deep Mandelbrot flights now rebuild a stale perturbation reference before its
  relative FP64 grid can collapse into repeated coordinates.
- Each completed flight candidate is checked before display for near-uniform
  colour fields, missing edge structure, repeated rows/columns and repeated
  two-dimensional pixel blocks.
- If either the coordinate grid or the rendered image loses useful resolution,
  the GUI rejects that candidate, restores the last good view and stops before
  block or colour-field artifacts are displayed.
- Direct auto-render frames expose grid uniqueness and ULP margin in renderer
  metadata; an exhausted fresh perturbation grid is reported as an explicit
  render boundary.

### Validation

- Added regressions for collapsed coordinate grids, automatic reference
  re-anchoring, uniform colour fields, repeated pixel blocks and last-good-frame
  restoration.

## 0.7.1 — 2026-07-25

### Flight and precision stability

- Added an automatic direct-precision ladder in `auto` mode: fast `float32`
  frames are promoted to `float64` before FP32 coordinate quantization becomes
  visible, followed by Mandelbrot perturbation when direct FP64 becomes unsafe.
- `direct` mode continues to honor the explicitly selected precision without
  hidden promotion.
- Added a conservative minimum viewport width derived from reference bits,
  render width and the FP64 perturbation-delta floor.
- Interactive flights now clamp to that width and stop automatically instead of
  continuing into blocky or single-colour frames.
- The GUI status reports effective precision transitions such as
  `float32→float64`.

### Validation

- Added regression coverage for FP32 promotion, FP64-to-perturbation switching,
  strict direct mode, finite reference-bit flight floors and automatic flight
  stopping.

## 0.7.0 — 2026-07-24

### Automatic tone mapping

- Added automatic contrast handling as the default presentation mode.
- Uses a deterministic, image-wide sample of at most 4096 outside pixels and
  robust percentile clipping instead of letting sparse extrema dominate the
  available palette range.
- Combines an adaptive `asinh` highlight compression curve with a bounded gamma
  correction derived from the sampled value distribution.
- Tone parameters are temporally smoothed across related pan and zoom frames;
  flights use stronger damping to avoid exposure pumping.
- Large scene changes converge faster, while viewport, resolution, palette and
  color-cycle changes do not unnecessarily reset exposure.
- Newton rendering keeps automatic mode linear so its three root regions remain
  correctly encoded.

### CUDA presentation path

- Adaptive tone mapping keeps iteration values and the inside mask on the GPU.
- CUDA transfers only a small stratified sample for parameter analysis and then
  performs tone curve, palette lookup and color cycling on the GPU.
- The optimized path still returns only the final RGB image, plus the small
  sample used for automatic exposure.

### Interface and tooling

- The GUI uses automatic tone mapping by default without requiring manual exposure controls.
- Added `--tone-mapping` to single-image, flight and benchmark CLI commands.
- CLI flights preserve and smooth tone state across their frame sequence.
- Reused renderer instances preserve temporally smoothed tone state across GUI frames.

### Validation

- Added automatic-curve, outlier-resistance, temporal-smoothing, scene-reset,
  Newton-preservation and large-frame sampling tests.
- Added CUDA-simulator parity coverage for the optimized automatic tone path.

## 0.6.0 — 2026-07-22

### Perturbation correctness and stability

- Replaced the previous per-frame reference anchoring with a persistent
  high-precision reference orbit cache.
- Related pan and zoom frames now reuse the same reference anchor; pixel
  offsets are computed relative to that anchor with arbitrary precision.
- Replaced the previous "repair by direct absolute FP64 continuation" with
  true perturbation rebasing. The deep parameter `c` is never reconstructed as
  an absolute float64 value.
- Added a cancellation-based glitch criterion and repair by rebasing the orbit
  representation to `Z_0 = 0`.
- Reference candidates are sampled across the current viewport and the
  longest-lived candidate is preferred, avoiding rapidly escaping reference
  orbits where possible.
- Escaped reference orbits have an explicit rebase limit, so kernels do not
  continue through overflowing reference values.
- CUDA keeps reused reference orbits resident on the device instead of
  uploading them for every frame.
- The GUI reports whether a perturbation reference was newly created or reused.

### Validation

- Added integer-pixel pan overlap regression tests; overlapping regions are
  bit-identical with a reused reference.
- Added `scripts/check_pan_stability.py` for reproducible CPU/CUDA validation on
  the target system.
- Expanded the test suite to 31 tests.

## 0.5.0 — 2026-07-22

### Deep zoom stability

- Added conservative rebasing / repair heuristics to the Mandelbrot
  perturbation renderer.
- Added glitch detection for unstable perturbation pixels.
- Unstable pixels switched from the perturbation path to a continued direct
  FP64 orbit from their current absolute state. This implementation was later
  replaced in 0.6.0 because it lost the deep parameter precision.
- CPU and CUDA backends report rebase/glitch activity.

## 0.4.0 — 2026-07-22

- Added Mandelbrot perturbation deep zoom with high-precision viewport storage.
- Added render modes `auto`, `direct`, and `perturbation`.
- Added configurable reference precision and CPU/CUDA perturbation kernels.

## 0.3.0 — 2026-07-22

- Added persistent CUDA buffers, true FP32/FP64 kernels, GPU palette mapping,
  and one RGB readback per interactive frame.
