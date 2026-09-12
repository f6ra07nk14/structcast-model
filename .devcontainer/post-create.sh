#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.codex"
mkdir -p "$HOME/.claude"
mkdir -p "$HOME/.cache/uv"
mkdir -p "$HOME/.local/bin"
mkdir -p "$HOME/.gemini/config"
mkdir -p .claude
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
claude plugin marketplace add DietrichGebert/ponytail && claude plugin install ponytail@ponytail
claude plugin marketplace add JuliusBrussee/caveman && claude plugin install caveman@caveman
# claude plugin marketplace add openai/codex-plugin-cc && claude plugin install codex@openai-codex

# Install session integrations after persistent user configuration is mounted.
# These commands update the host-shared hook settings for each agent.
herdr integration install codex
herdr integration install claude
herdr integration install antigravity-cli
herdr integration status

# Give each agent the Herdr skill so they can address each other by name
# (herdr agent prompt <name> ...). `herdr --skill` emits the SKILL.md for the
# installed herdr version, so this re-syncs on every rebuild. These dirs are
# host bind-mounts for claude/codex, so the skill lands on the host too.
for skill_dir in "$HOME/.claude/skills" "$HOME/.codex/skills" "$HOME/.gemini/config/skills"; do
  mkdir -p "$skill_dir/herdr"
  # Write via temp file: a plain redirect would truncate a good SKILL.md into an
  # empty one if `herdr --skill` failed.
  if herdr --skill > "$skill_dir/herdr/SKILL.md.tmp"; then
    mv "$skill_dir/herdr/SKILL.md.tmp" "$skill_dir/herdr/SKILL.md"
  else
    rm -f "$skill_dir/herdr/SKILL.md.tmp"
    echo "Warning: could not write $skill_dir/herdr/SKILL.md; continuing."
  fi
done

# Install the herdr-crew launcher. It is NOT run here: post-create runs headless
# with no Herdr session, and starting eight agents is a deliberate, billable act.
# Run it by hand from inside a Herdr pane once the container is up.
cat > "$HOME/.local/bin/herdr-crew" <<'HERDR_CREW_EOF'
#!/usr/bin/env bash
# Spawn the standard 8-agent crew in the current Herdr session.
# Named agents are addressable: herdr agent prompt worker_opus "..." --wait
# Usage: herdr-crew [project-dir]
set -euo pipefail
CWD="${1:-$PWD}"

# tab-label|agent-name|kind|native args
CREW=(
  'lead|project_leader|claude|--model claude-fable-5-1 --effort low --dangerously-skip-permissions'
  'workers|worker_opus|claude|--model claude-opus-5 --effort ultracode --dangerously-skip-permissions'
  'workers|worker_sonnet|claude|--model claude-sonnet-5 --effort max --dangerously-skip-permissions'
  'workers|worker_astra|codex|-m gpt-6-astra -c model_reasoning_effort="xhigh"'
  'workers|worker_sol|codex|-m gpt-5.6-sol -c model_reasoning_effort="xhigh"'
  'reviewers|reviewer_opus|claude|--model claude-opus-5 --effort xhigh --dangerously-skip-permissions'
  'reviewers|reviewer_astra|codex|-m gpt-6-astra -c model_reasoning_effort="low"'
  'reviewers|reviewer_sol|codex|-m gpt-5.6-sol -c model_reasoning_effort="xhigh"'
  'websearch|worker_gemini|agy|--model gemini-3.8-flash-high --dangerously-skip-permissions'
)

cur_tab=""; last_pane=""; n=0
for entry in "${CREW[@]}"; do
  IFS='|' read -r tab name kind args <<<"$entry"
  if [[ $tab != "$cur_tab" ]]; then
    pane=$(herdr tab create --label "$tab" --cwd "$CWD" | jq -r '.result.root_pane.pane_id')
    cur_tab=$tab; n=0
  else
    # ponytail: alternate right/down chain split; hand-drag borders if you want exact 2x2
    (( n % 2 )) && dir=down || dir=right
    pane=$(herdr pane split "$last_pane" --direction "$dir" --cwd "$CWD" --no-focus \
             | jq -r '.result.pane.pane_id')
  fi
  herdr pane rename "$pane" "$name" >/dev/null
  # shellcheck disable=SC2086
  herdr agent start "$name" --kind "$kind" --pane "$pane" --timeout 120000 -- $args \
    || echo "WARN: $name did not report ready; check with: herdr agent get $name" >&2
  last_pane=$pane; n=$((n+1))
done
herdr agent list | jq -r '.result.agents[] | "\(.agent_status)\t\(.pane_id)"'
HERDR_CREW_EOF
chmod +x "$HOME/.local/bin/herdr-crew"

echo "Devcontainer post-create setup complete."
echo "Codex path: $(command -v codex || echo 'not found')"
echo "UV path: $(command -v uv || echo 'not found')"
