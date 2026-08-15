# Repository Guidelines

These instructions apply to every task in this repository unless the user explicitly overrides them.
Prefer the smallest correct change and avoid adding structure before it is needed.

## Project Structure & Module Organization

The repository is intentionally minimal. Create directories and modules only when a task requires them. Keep related
logic together until a clear, repeated need justifies splitting it.

## Build, Test, and Development Commands

Use Python, matching `.python-version` and `requires-python`. Prefix shell commands with `rtk` as required by the
workspace tooling.

If `src/` or `tests/` does not exist yet, run tools only against paths that exist. Do not change configured Python or
tool versions unless the task requires it.

## Coding Style & Naming Conventions

Prefer clear names and direct implementations. Add docstrings when behavior or a public interface is not obvious. Do
not add wrappers, base classes, configuration layers, or dependencies for a single use case.

## Testing Guidelines

Add the narrowest test that proves the requested behavior, favoring deterministic inputs and real collaborators over
broad mocks.

Do not add ignores such as `# noqa` or `# type: ignore` without explaining the specific violation and why a local fix
is unsuitable.

## Git & Merge Request Guidelines

This repository is hosted on GitHub and uses four task branches:

- `main`: externally released, verified development results.
- `dev`: normal feature development.
- `research`: web research, source validation, and information synthesis.
- `experiments`: experiments for a specific feature.

Before starting repository work, confirm the task type and corresponding branch with the user. Ensure the branch is up
to date before making changes:

- Merge the latest `main` into `dev` before feature development.
- Merge the latest `dev` into `research` or `experiments` before research or experimental work.

Perform branch-based work in a Git worktree. Confirm whether to use an existing worktree or create a dedicated one.
When creating a new worktree, place it under `.worktrees/<branch>` inside the project directory so it survives
devcontainer rebuilds; `.worktrees/` is git-ignored. Keep the main working directory free from unrelated task states. Preserve existing user changes and do not revert,
overwrite, or clean unrelated work.

All commit messages must follow Conventional Commits using `<type>(<optional scope>): <imperative description>`. Use
the types configured in `.releaserc.yaml`: `breaking`,
`build`, `chore`, `ci`, `docs`, `example`, `feat`, `fix`, `perf`, `refactor`, `revert`, `style`, or `test`. Mark breaking
changes with `!`, a `BREAKING CHANGE:` footer, or the `breaking` type. Ask before creating commits. Merge requests
should state the change, verification performed, and any known limitations; keep each merge request focused.

## Communication Rules

Ask and reply in 繁體中文 (Traditional Chinese). Keep repository-facing content, commands, code, commit messages, and internal
work output in English.

## Agents Rules

These rules apply to every task in this project unless explicitly overridden. Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

### Rule 1 - Think Before Coding

State assumptions explicitly. If uncertain, ask rather than guess. Present multiple interpretations when ambiguity
exists. Push back when a simpler approach exists. Stop when confused and name what is unclear.

### Rule 2 - Simplicity First

Write the minimum code that solves the problem. Add nothing speculative, no unrequested features, and no abstractions
for single-use code. If a senior engineer would call the solution overcomplicated, simplify it.

### Rule 3 - Surgical Changes

Touch only what is necessary. Clean up only your own mess. Do not improve adjacent code, comments, or formatting. Do not
refactor what is not broken. Match the existing style.

### Rule 4 - Goal-Driven Execution

Define success criteria, then loop until verified. Do not follow steps blindly; define success and iterate independently.

### Rule 5 - Use the Model Only for Judgment Calls

Use the model for classification, drafting, summarization, and extraction. Do not use it for routing, retries, or
deterministic transforms. If code can answer, code answers.

### Rule 6 - Token Budgets Are Not Advisory

Per-task budget: 32k tokens. Per-session budget: 256k tokens. If approaching a budget, summarize and start fresh.
Surface the breach; do not silently overrun.

### Rule 7 - Surface Conflicts, Do Not Average Them

If two patterns contradict, pick one, usually the more recent or more tested option. Explain why and flag the other for
cleanup. Do not blend conflicting patterns.

### Rule 8 - Read Before You Write

Before adding code, read exports, immediate callers, and shared utilities. If unsure why code is structured a certain
way, ask.

### Rule 9 - Tests Verify Intent

Tests must encode why behavior matters, not only what it does. A test that cannot fail when business logic changes is
wrong.

### Rule 10 - Checkpoint After Significant Steps

Summarize what was done, what is verified, and what remains. Do not continue from a state you cannot describe. If you
lose track, stop and restate.

### Rule 11 - Match Codebase Conventions

Conformance is more important than taste inside the codebase. If a convention appears harmful, surface it. Do not fork
silently.

### Rule 12 - Fail Loud

Do not claim completion if anything was skipped silently. Do not say tests pass if any were skipped. Surface uncertainty
instead of hiding it.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `f6ra07nk14/structcast`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, using the default label strings. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
