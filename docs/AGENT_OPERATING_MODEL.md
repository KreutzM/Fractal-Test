# Agent operating model

This document describes the durable issue-to-merge lifecycle. `AGENTS.md` is the
binding repository overview, `.hermes.md` is the canonical Hermes operating
policy, and `AGENT_GIT_WORKFLOW.md` contains detailed transport and publication
fallbacks.

## Lifecycle

```text
GitHub issue
    -> current main
    -> agent/<issue>-<short-description>
    -> implementation and relevant validation
       -> optional scoped leaf subagents
       -> parent verification/integration
    -> committed review target + pushed branch
    -> draft pull request
    -> fresh independent reviewer leaf
       -> parent evaluates/fixes valid findings
       -> fresh re-review until clean verdict
    -> final CI/status gate
       -> fix + re-review again if the code changes
    -> review-ready pull request
    -> explicit user-authorized merge
    -> durable issue/PR handoff
```

### 1. Establish current truth

Fetch `origin`, read the repository instructions and relevant architecture, then
read the complete issue, comments, existing branch, pull request, discussion,
diff, history, and recorded checks. Current repository and GitHub evidence
supersedes historical chat context.

### 2. Select the work item

Substantive work begins with an issue. Continue its existing branch and pull
request when they exist. Otherwise, branch from current `origin/main` using
`agent/<issue-number>-<short-description>`. An issue normally maps to one
reviewable pull request; tracking issues may coordinate multiple child issues and
pull requests.

### 3. Implement within scope

Follow the issue contract, `AGENTS.md`, and the architectural boundaries. Small
necessary fixes may remain in the same change. Record independently useful or
out-of-scope work as a linked follow-up issue rather than silently enlarging the
pull request.

Hermes may use leaf subagents for genuinely separable reasoning-heavy work such
as independent review, architecture analysis, test-gap discovery, investigations,
or isolated implementation leaves. The parent remains responsible for the issue,
main feature branch, final integration, validation, commits, push, PR, and
handoff. Child summaries are inputs to verify, not authoritative project state.

Read-only children may share the parent workspace. Parallel code-writing children
must use isolated worktrees when that runtime feature is available; otherwise
keep implementation sequential or delegate analysis-only work. Review child diffs
and test evidence before merging or cherry-picking them into the parent branch.
Routine development stays flat: parent -> leaf children.

### 4. Validate and commit

Run the checks relevant to the changed area and report only actual results. Keep
local tests, CI, simulator, physical hardware, and manual checks distinct. Inspect
the complete diff and `git status`, run `git diff --check`, and create focused
commits that reference the issue where practical.

For a substantive product-code change, commit the exact state that will be sent
to an independent reviewer before dispatching the reviewer. This is especially
important with Hermes child worktree isolation because each child worktree is
branched from the parent's current `HEAD`.

### 5. Publish and autonomously review

Push the branch normally and open or update the linked pull request. Keep it
draft while implementation or autonomous review is incomplete. Follow the
required PR structure in `.hermes.md`.

A substantive product-code PR must then pass an autonomous review-convergence
loop inside the same parent Hermes session:

1. dispatch a fresh independent reviewer leaf with the issue contract, relevant
   repository constraints, current commit/PR, actual diff and validation target;
2. reviewer work is read-only by default and covers spec compliance, correctness,
   regressions, architecture/invariants, test gaps, edge cases, relevant platform
   concerns, and misleading claims;
3. the parent evaluates every finding independently, fixes valid in-scope
   BLOCKER/MAJOR and actionable MINOR findings on the primary feature branch,
   validates, commits, and pushes;
4. dispatch a fresh reviewer after each fix round rather than asking the prior
   reviewer context to confirm itself;
5. repeat until a fresh reviewer reports no unresolved actionable findings and a
   merge-ready verdict from the review perspective.

This loop is intended to replace manual user orchestration between separate
implementer and reviewer sessions. The parent owns the issue lifecycle and moves
between implementation, review and repair itself.

Use at most three review/fix rounds without convergence. If a genuine blocker or
semantic disagreement remains after that, keep the PR not-ready, record the exact
unresolved point in GitHub, and escalate only that decision to the user.

### 6. Final CI gate and handoff

Check final GitHub CI/status before declaring the PR review-ready. Passing CI is
required evidence but does not replace independent review. If a change-driven CI
failure requires another code modification, validate, commit and push the fix,
then independently review the new delta again before handoff.

Physical CUDA or manual checks remain distinct. When `AGENTS.md` permits them to
be unavailable, record them explicitly as pending rather than blocking unrelated
CPU/CI evidence.

Once the independent review has converged and final CI is acceptable, mark the PR
review-ready and update its handoff with the current commit, decisions, actual
checks and any permitted pending hardware/manual validation.

Merge into `main` only after explicit user authorization. Passing checks or an
approval does not itself grant Hermes merge permission.

### 7. Handoff

Before stopping, ensure the branch is pushed and the issue/PR records the current
commit, decisions, actual checks, remaining work, and next action. This durable
state replaces chat-memory handoffs.

Start a fresh chat with only:

> Work on issue #N. Treat repository and GitHub state as source of truth. Read
> AGENTS.md, .hermes.md, relevant docs, the issue, existing branch/PR/comments,
> then continue from the current repository state.

## Recommended Hermes delegation runtime

For local Git development in this repository, the recommended user-level Hermes
configuration is:

```yaml
delegation:
  max_concurrent_children: 3
  worktree_isolation: true
  max_spawn_depth: 1
```

This belongs in `~/.hermes/config.yaml`, not in the repository policy file.
Worktree isolation prevents concurrent coding children from clobbering the same
checkout. The flat depth keeps ordinary development understandable and avoids
unnecessary agent trees. If isolation is unavailable, the repo policy requires
sequential code integration or analysis-only delegation instead.

## Model selection

The workflow is model-independent and model selection happens at invocation time
or in user-level Hermes configuration. A team may configure
`delegation.provider` / `delegation.model` so fresh reviewer children use a
different model family from the parent. This is a useful implementation/reviewer
separation, but concrete model identifiers remain an operational choice rather
than durable repository architecture.
