# Agent Git workflow

This document defines the least complex safe workflow available when normal Git transport may be unavailable. The mandatory summary is in the repository-root `AGENTS.md`.

## Goals

- Work from the exact current target branch.
- Use normal Git transport whenever it works.
- Prefer direct connector file edits for ordinary source and documentation changes.
- Accept several focused branch commits when squash merge will produce one clean commit on `main`.
- Reserve the Git-data blob/tree/commit workflow for changes that genuinely require byte-exact or atomic publication.
- Avoid repeated capability probes, unnecessary snapshot downloads, oversized documentation churn, and manual payload construction.

## Decision flow

```text
Normal authenticated Git push works?
├─ Yes: develop, commit and push normally.
└─ No: use the GitHub connector.

Connector publication needed?
├─ Ordinary UTF-8 text files:
│    create branch, read files, update/create/delete files directly.
└─ Binary, byte-critical, executable-mode, large generated, or strictly atomic change:
     use the Git-data blob/tree/commit workflow.
```

File count alone does not require Git-data publication. A small or medium multi-file pull request may use several direct connector commits and be squash-merged.

## 1. Probe capabilities once

At the start of a task, perform one bounded capability check:

1. inspect any available local repository and current branch;
2. try normal Git/CLI transport once when available;
3. verify connector access once;
4. select the working path and do not repeatedly retry an unavailable transport during the same task.

A failed DNS lookup, missing `gh`, unavailable credentials, or an inaccessible remote should immediately select the connector path. Repeated probes add latency without increasing safety.

## 2. Establish the current base

### Normal local path

Use an up-to-date local clone when available:

```powershell
git fetch origin --prune
git switch main
git pull --ff-only
git switch -c agent/<description>
```

### Connector-only path

When no usable local transport exists:

1. read repository metadata and identify the current default branch;
2. create `agent/<description>` directly from the current target branch;
3. fetch every existing file that will be updated and record its blob SHA;
4. create new files directly without inventing a prior SHA;
5. do not commit directly to `main`.

Direct file reads are acceptable for the files being intentionally changed. Do not reconstruct an entire source tree from unrelated individual downloads and then claim it was locally validated.

### Snapshot use

A repository snapshot is optional, not the default connector prerequisite. Use the newest successful snapshot artifact only when local execution or broad repository inspection requires a complete checkout and no verified current checkout is available.

When used, verify its checksum manifest and Git bundle before development. A previously verified checkout may be reused when its tree SHA exactly matches the current target tree.

## 3. Develop and validate

### With a local checkout

Work normally in Git:

```powershell
git status --short
git diff --check
git diff --stat
git add -- <intended paths>
git commit -m "<focused description>"
```

Run the checks required by `AGENTS.md` and report only checks actually executed.

### Connector-only edits

For ordinary UTF-8 text changes:

1. fetch the current file from the feature branch or target branch;
2. replace it through the contents API using its current blob SHA;
3. create new UTF-8 files through the contents API;
4. delete files only with their current blob SHA;
5. perform dependent edits sequentially when they affect the same file or branch state.

Each contents-API write creates a commit. This is acceptable. Prefer focused commit messages and rely on squash merge to keep `main` history clean.

Do not spend more time constructing a one-commit Git-data transaction than the change itself warrants.

Keep documentation proportional:

- update `CHANGELOG.md` for user-visible behavior;
- update `README.md` only when the user workflow materially changes;
- update `TEST_REPORT.md` only when validation strategy or durable results change;
- do not rewrite large historical sections mechanically.

## 4. Preferred publication paths

### A. Normal Git push

```powershell
git push -u origin HEAD
```

Then open a draft pull request through the GitHub connector. This remains the preferred route when available.

### B. Direct connector file operations

Use this as the default fallback for ordinary Python, JSON, YAML, TOML, Markdown, and other UTF-8 text files.

Typical sequence:

1. create a feature branch from the current target branch;
2. fetch existing files and their blob SHAs;
3. apply `update_file`, `create_file`, or `delete_file` operations;
4. compare the branch with the target;
5. open a draft pull request.

Several commits are acceptable for one logical pull request. Do not add an extra blob/tree consolidation step solely to force the branch to one commit.

### C. Git-data blob/tree/commit workflow

Use `scripts/prepare_connector_publish.py` only when at least one of these conditions applies:

- binary files must be preserved exactly;
- CRLF, non-UTF-8 encoding, executable mode, or another byte-level property is material;
- a large generated payload is safer to transfer from committed Git objects;
- a many-file change must appear atomically on the feature branch;
- exact local-tree identity is a stated validation requirement.

Then generate and publish exact committed objects:

```powershell
python scripts\prepare_connector_publish.py `
  --base-ref <verified-local-base-ref> `
  --repository KreutzM/fractal-flight-studio `
  --remote-base-commit <current-remote-main-sha> `
  --expected-base-tree <current-remote-main-tree-sha> `
  --branch agent/<description> `
  --output-dir .agent-publish
```

Required sequence:

1. create and verify every blob;
2. create one tree and verify its expected SHA;
3. create one commit with the current remote target commit as parent;
4. create the feature branch only after the commit exists;
5. compare target and feature branch;
6. open a draft pull request.

Stop at the first blob or tree SHA mismatch. Do not retry by manual copying or re-encoding.

## 5. Pre-PR verification

Before opening the pull request, verify:

- the feature branch is based on the intended current target;
- `behind_by == 0`;
- changed paths are exactly the intended paths;
- additions and deletions are plausible;
- generated payload directories, caches, coverage files, build output, and local artifacts are not committed;
- validation claims distinguish local checks, CI checks, simulator checks, and physical GPU checks.

For direct connector edits, a branch may contain several commits. One-commit-ahead is required only for the explicit Git-data atomic path, not for the normal contents-API path.

If the target branch moves during a connector-only documentation or small source change, reassess whether the branch can be safely updated or recreated. Do not silently overwrite concurrent changes.

## 6. Pull request and merge

Open a draft pull request unless the user explicitly requests otherwise. State:

- what changed and why;
- which publication path was used;
- checks actually executed;
- checks delegated to CI;
- remaining hardware or platform validation.

Use squash merge by default for connector-created multi-commit branches unless preserving individual commits has a specific value. Merge only after required checks pass and the user authorizes it.

## Prohibited shortcuts

- repeatedly retrying a transport already known to be unavailable;
- committing directly to `main`;
- editing from a stale or unknown base;
- claiming a collection of downloaded files is a fully verified current checkout;
- using Git-data publication solely to avoid several harmless feature-branch commits;
- manually copying large binary or byte-critical data into text file operations;
- accepting a blob or tree SHA mismatch in the Git-data path;
- describing CI as local validation or simulator results as physical GPU validation.
