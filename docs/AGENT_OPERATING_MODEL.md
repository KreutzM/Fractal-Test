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
    -> focused commits and pushed branch
    -> draft or review-ready pull request
    -> review and authorized merge
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

### 5. Publish and review

Push the branch normally and open or update the linked pull request. Use a draft
while work is incomplete and a review-ready pull request when implementation and
meaningful local validation are complete. Follow the required PR structure in
`.hermes.md`. Review may request further commits on the same branch.

Merge into `main` only after explicit user authorization. Passing checks or an
approval does not itself grant Hermes merge permission.

### 6. Handoff

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

The workflow is model-independent and model selection happens at invocation time.
If a team chooses multiple models, it may use fast implementer candidates and an
independent reviewer with a different model family, but current model preferences
are operational choices rather than durable repository architecture.
