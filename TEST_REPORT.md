# Test report

Date: 2026-07-31
Release candidate: 0.10.0
Repository head before this documentation PR: `02f7f4989d0bba8e5c153debfa5cc8eee30394c8`

## Scope

This report summarizes the durable validation state after the unified flight-plan
workflow, production CUDA Double-Single direct rendering, guarded Double-Single
perturbation and the six bundled example flight plans.

## Automated validation

The most recent feature branch completed:

- Python source compilation;
- 265 tests across focused and remaining deterministic groups;
- Windows and Ubuntu CI on Python 3.11, 3.12 and 3.13;
- Tk/Xvfb GUI smoke coverage;
- CPU pan-stability validation;
- wheel build and package-data checks;
- repository snapshot validation.

The example-flight tests load every bundled schema-2 document through the
production parser, require deterministic serialization, evaluate complete
timelines, check distinct deep endpoints and constrain the extended examples to
180–300 seconds.

## Physical RTX 3060 validation

### Direct Double-Single

The production `auto` path promoted public precision from FP32 to FP64 and used
internal Double-Single arithmetic for eligible Mandelbrot direct frames. The
initial production-shaped Seahorse Valley measurement reported 8.111 ms versus
39.517 ms for explicit native FP64, a 4.87× median speedup.

The broader 1280×720 matrix covered exterior, interior, cusp, Seahorse Valley,
Seahorse satellite, a near-direct-coordinate floor and the transition to
perturbation. Compute-heavy cases measured approximately 3.4× to 5.0× faster
than native FP64; the short exterior case measured 1.87×. Stable regions were
byte-identical or nearly identical. Chaotic boundary cases retained small
frame-wide deltas and at most 0.1993% inside-mask divergence against FP64. The
validated deeper case switched to perturbation before direct-coordinate
uniqueness failed.

### Double-Single perturbation

The production Seahorse deep-zoom check at 1280×720, 1200 iterations and a
384-bit CPU reference orbit measured 149.735 ms for Double-Single and 443.391 ms
for explicit native FP64, a 2.961× speedup. All 921,600 pixels matched exactly
for normalized values, inside decisions, glitch flags and rebase flags.

The seven-case physical matrix validated six eligible targets plus one guarded
fallback. Eligible targets measured 1.872× to 2.988× faster than native FP64.
All cases had zero inside, glitch and rebase mismatches. Five Double-Single cases
had zero value delta; the short Misiurewicz `-2` case had maximum normalized
delta `1.863e-9` while preserving identical classification and flags. The
intentional tiny-reference case correctly selected native FP64 with fallback
reason `reference-magnitude-range`.

## Numerical invariants retained

- Deep viewport centers, widths, reference anchors and offsets remain exact text
  or arbitrary-precision values until renderer setup.
- The high-precision perturbation reference orbit remains on the CPU.
- Rebasing never reconstructs the deep parameter as an absolute FP64 value.
- Explicit native FP64 remains available as a reference and safety path.
- Double-Single does not extend the FP32 exponent range; conservative guards
  delegate unsupported cases to native FP64.
- Integer-pixel pan overlap remains stable with a reused reference orbit.

## Flight-plan and visual validation

Six example plans are bundled under `examples/flight_plans/`. Their final views
were selected from structure-rich subregions and checked with the existing
visual-quality classifier. The extended examples last 3:30, 4:20 and 4:58 and
use active camera movement, target correction, zooming and relocation rather
than long static holds.

## Remaining validation boundaries

- Physical performance results currently target the RTX 3060 and should not be
  generalized to all NVIDIA architectures without measurement.
- CUDA-simulator tests establish functional execution, not physical performance
  or driver compatibility.
- Subjective visual ranking of palettes and routes remains display- and
  preference-dependent.
- AMD, Intel and Apple GPU backends are not implemented.
