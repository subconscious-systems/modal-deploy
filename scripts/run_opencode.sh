#!/usr/bin/env bash
# Launch OpenCode against the Modal sglang endpoint.
# Env-only override for this process — nothing written to ~/.opencode/.
#
# Usage:
#   ./scripts/run_opencode.sh
#   ./scripts/run_opencode.sh --continue
#   ./scripts/run_opencode.sh https://<workspace>--glm-5-2-fp8-marathon.modal.run
#   ./scripts/run_opencode.sh https://... --continue

set -euo pipefail

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

export OPENCODE_API_KEY="dummy"
export OPENCODE_CONFIG_CONTENT=$(cat <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "modal": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Modal GLM-5.2",
      "options": {
        "baseURL": "${ENDPOINT}/v1",
        "apiKey": "{env:OPENCODE_API_KEY}"
      },
      "models": {
        "glm-5.2-fp8": {
          "name": "GLM-5.2",
          "tools": true,
          "limit": { "context": 1000000, "output": 65536 }
        }
      }
    }
  },
  "model": "modal/glm-5.2-fp8"
}
EOF
)

if ! command -v opencode >/dev/null 2>&1; then
  echo "error: \`opencode\` not on PATH. Install OpenCode first:" >&2
  echo "  npm i -g opencode-ai" >&2
  exit 127
fi

exec opencode ${ARGS[@]+"${ARGS[@]}"}
