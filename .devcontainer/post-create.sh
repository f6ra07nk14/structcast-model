#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.codex"
mkdir -p "$HOME/.claude"
mkdir -p "$HOME/.cache/uv"
mkdir -p "$HOME/.local/bin"
mkdir -p /commandhistory

touch /commandhistory/.bash_history

if [ -d "$HOME/.claude.json" ]; then
  echo "$HOME/.claude.json is a directory. Ensure the host ~/.claude.json exists as a file before rebuilding." >&2
  exit 1
fi

if ! grep -q "HISTFILE=/commandhistory/.bash_history" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'EOF'

# Persist shell history across devcontainer rebuilds.
export HISTFILE=/commandhistory/.bash_history
export HISTSIZE=10000
export HISTFILESIZE=20000
shopt -s histappend
PROMPT_COMMAND="history -a; history -c; history -r; ${PROMPT_COMMAND:-}"
EOF
fi

if ! grep -q "alias codex=" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<'EOF'

# Default codex to full access inside this container. Interactive shells only:
# scripts and the Claude codex plugin invoke the real binary unmodified, and
# the host's shared ~/.codex/config.toml is untouched.
alias codex='codex --sandbox danger-full-access --ask-for-approval never'
alias codex-safe='command codex --sandbox workspace-write --ask-for-approval on-request'
EOF
fi

if ! codex --version >/dev/null 2>&1; then
  echo "Codex CLI is not runnable. Rebuild the devcontainer image to reinstall it." >&2
  exit 1
fi

if [ -f "pyproject.toml" ]; then
  echo "pyproject.toml found. Running uv sync..."
  uv sync
else
  echo "No pyproject.toml found. Skipping uv sync."
fi

if command -v rtk >/dev/null 2>&1; then
  rtk telemetry disable || echo "Warning: rtk telemetry disable failed; continuing."
  rtk init -g --auto-patch || echo "Warning: rtk auto-patch init failed; continuing."
  rtk init -g --codex || echo "Warning: rtk Codex init failed; continuing."
  rtk init -g --copilot || echo "Warning: rtk Copilot init failed; continuing."
else
  echo "rtk not found. Skipping rtk initialization."
fi

# Expose the project's agent-agnostic skills (.agents/skills, versioned with the
# repo) to Claude Code's skill loader, which only looks under <repo>/.claude/skills.
# This is a project-local path (workspaceFolder), unrelated to $CLAUDE_HOME.
# It must be (re)created here rather than in the Dockerfile because the repo
# isn't checked out/mounted yet at image build time.
mkdir -p .claude
if [ -L .claude/skills ]; then
  if [ "$(readlink .claude/skills)" != "../.agents/skills" ]; then
    ln -sfn ../.agents/skills .claude/skills
  fi
elif [ ! -e .claude/skills ]; then
  ln -s ../.agents/skills .claude/skills
fi

# Default Claude Code to bypassing permission prompts (equivalent to running
# with --dangerously-skip-permissions), matching the trust level codex gets via
# the codex-full alias. Written to the project-level settings file, NOT the
# user-level ~/.claude/settings.json: ~/.claude is bind-mounted from the host,
# and a user-level default would silently disable permission prompts for every
# project on the host. skipDangerousModePermissionPrompt suppresses the
# one-time acceptance prompt so fresh containers work without interaction.
# Re-running post-create re-asserts these two keys but preserves everything
# else in the file.
CLAUDE_SETTINGS=".claude/settings.local.json"
mkdir -p .claude
if [ ! -s "$CLAUDE_SETTINGS" ]; then
  printf '{}\n' > "$CLAUDE_SETTINGS"
fi
if command -v jq >/dev/null 2>&1; then
  CLAUDE_SETTINGS_TMP="$(mktemp)"
  if jq '.permissions.defaultMode = "bypassPermissions" | .skipDangerousModePermissionPrompt = true' \
    "$CLAUDE_SETTINGS" > "$CLAUDE_SETTINGS_TMP"; then
    mv "$CLAUDE_SETTINGS_TMP" "$CLAUDE_SETTINGS"
    echo "Claude Code set to bypass permission prompts by default."
  else
    rm -f "$CLAUDE_SETTINGS_TMP"
    echo "Warning: could not update $CLAUDE_SETTINGS (invalid JSON?); continuing."
  fi
else
  echo "Warning: jq not found; skipping Claude permission defaults."
fi

# Wire Claude Code plugins via the official plugin marketplace. $CLAUDE_HOME is
# bind-mounted from the host's ~/.claude, so plugin state is intentionally
# shared with the host and survives rebuilds.
if ! command -v claude >/dev/null 2>&1; then
  echo "Claude CLI not found. Skipping Claude plugin setup."
else
  claude plugin marketplace add DietrichGebert/ponytail && claude plugin install ponytail@ponytail
fi

echo "Devcontainer post-create setup complete."
echo "Codex path: $(command -v codex || echo 'not found')"
echo "UV path: $(command -v uv || echo 'not found')"
