#!/usr/bin/env bash
# Launch Claude Code against the Modal sglang endpoint.
# Env-only override for this process — nothing written to ~/.claude/.
#
# Usage:
#   ./scripts/run_claude.sh
#   ./scripts/run_claude.sh --continue
#   ./scripts/run_claude.sh https://<workspace>--glm-5-2-fp8-marathon.modal.run
#   ./scripts/run_claude.sh https://... --continue

set -euo pipefail

MODEL="glm-5.2[1m]"

ENDPOINT="https://subconscious-systems--glm-5-2-fp8-marathon.modal.run"
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == http://* || "$arg" == https://* ]]; then
    ENDPOINT="$arg"
  else
    ARGS+=("$arg")
  fi
done
ENDPOINT="${ENDPOINT%/}"
ENDPOINT="${ENDPOINT%/v1}"

export ANTHROPIC_BASE_URL="$ENDPOINT"
export ANTHROPIC_AUTH_TOKEN="dummy"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="1000000"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"
export CLAUDE_CODE_ATTRIBUTION_HEADER="0"
export API_TIMEOUT_MS="3000000"

if ! command -v claude >/dev/null 2>&1; then
  echo "error: \`claude\` not on PATH. Install Claude Code first:" >&2
  echo "  curl -fsSL https://claude.ai/install.sh | bash" >&2
  exit 127
fi

exec claude ${ARGS[@]+"${ARGS[@]}"}
