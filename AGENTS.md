# AGENTS.md

## Scope

These instructions apply to the entire repository.

Repository and GitHub state are the source of truth for agent work; prior chat or
session context is not authoritative when it conflicts with current evidence.
Hermes must also follow the repo-local operating policy in `.hermes.md`. The
end-to-end issue lifecycle is documented in `docs/AGENT_OPERATING_MODEL.md`, and
Git transport and publication details are documented in
`docs/AGENT_GIT_WORKFLOW.md`.

## Environment

- Primary target: Windows 11 with PowerShell 7.
- Supported Python versions: 3.11, 3.12, and 3.13.
- CPU rendering must remain available without CUDA.
- CUDA work must remain compatible with the optional `numba-cuda[cu12]` setup used by the Windows scripts.

## Setup and validation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . pytest
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_pan_stability.py --backend cpu
```

On Linux, use `.venv/bin/python` instead. For GUI changes also run:

```bash
xvfb-run -a env PYTHONPATH=src python scripts/gui_smoke.py
```

## Numerical invariants

- Never reduce deep-zoom viewport centers, widths, reference anchors, or reference offsets to absolute Python `float` values before perturbation setup.
- Preserve high-precision text/`mpmath` coordinates through navigation, flight interpolation, and reference selection.
- Related pan and zoom frames should reuse a valid reference orbit; avoid per-frame reference changes that make overlapping pixels unstable.
- Rebasing may change the decomposition `z = Z + dz`, but must not reconstruct the deep parameter `c` as an absolute FP64 value.
- CPU and CUDA paths must agree within the tolerances established by the tests.
- Any perturbation, reference-cache, viewport, or CUDA-kernel change must run the full test suite and `scripts/check_pan_stability.py`.
- Do not claim physical CUDA performance or driver compatibility from CUDA-simulator results alone.

## Repository acquisition and GitHub publishing

- Prefer a normal up-to-date local clone and normal `git push`; probe an unavailable transport only once per task.
- When normal push is unavailable, prefer direct GitHub connector file operations for ordinary UTF-8 text changes.
- Create the feature branch from the current target branch before connector edits. Read existing files first and pass their current blob SHA to updates.
- Direct connector edits are the default for small and medium changes. Several focused connector commits are acceptable because pull requests are squash-merged.
- Use the Git-data blob/tree/commit workflow only when exact byte preservation, binary files, executable modes, generated large payloads, or an atomic many-file commit materially require it.
- Do not use the Git-data workflow merely to reduce the number of commits.
- After connector writes, compare the feature branch with the target and require `behind_by == 0` plus only expected paths.
- Before publishing, read `docs/AGENT_GIT_WORKFLOW.md` and select the least complex safe path described there.

## Development workflow

- Substantive work starts from a GitHub issue and normally produces one dedicated
  feature branch and one reviewable pull request.
- Work on a feature branch and open a pull request; do not commit directly to `main`.
- Do not merge a pull request to `main` without explicit user approval.
- Keep commits small, focused, and reviewable. Multiple connector-generated commits are acceptable when the PR will be squash-merged.
- Add or update tests for behavioral and numerical changes.
- Update `CHANGELOG.md` for user-visible release behavior. Update `README.md` only when user workflows materially change, and update `TEST_REPORT.md` only when validation strategy or durable results change; avoid rewriting large documents mechanically in every PR.
- Avoid committing generated images, benchmark output, virtual environments, build artifacts, or caches.
