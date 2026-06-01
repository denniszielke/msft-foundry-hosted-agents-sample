#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ───────────────────────────────────────────────
# Agent name: pass as first argument or defaults to travel-concierge
AGENT_NAME="${1:-travel-concierge}"

# Load environment from .env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${AZURE_AI_PROJECT_ENDPOINT:-}" ]]; then
  echo "ERROR: AZURE_AI_PROJECT_ENDPOINT is not set." >&2
  exit 1
fi

BASE_URL="${AZURE_AI_PROJECT_ENDPOINT%/}"
TOKEN=$(az account get-access-token --resource https://ai.azure.com \
  --query accessToken -o tsv)

A2A_BASE="$BASE_URL/agents/$AGENT_NAME/endpoint/protocols/a2a"
CARD_URL="$A2A_BASE/agentCard/v0.3"
API_VERSION="v1"

echo "Agent:         $AGENT_NAME"
echo "A2A base path: $A2A_BASE"
echo "Agent card URL: $CARD_URL"
echo ""

# ── 2. GET — fetch the agent card ──────────────────────────────
echo ">>> GET agent card..."
curl -s -w "\nHTTP %{http_code}\n" \
  -X GET "$CARD_URL" \
  -H "Authorization: Bearer $TOKEN" | jq . 2>/dev/null || true
echo ""

# ── 3. POST — send a test A2A message ──────────────────────────
echo ">>> POST test A2A message..."
curl -s -w "\nHTTP %{http_code}\n" \
  -X POST "$A2A_BASE" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test-1",
    "method": "message/send",
    "params": {
      "message": {
        "kind": "message",
        "messageId": "msg-test-1",
        "role": "user",
        "parts": [
          { "kind": "text", "text": "Hello, are you there?" }
        ]
      }
    }
  }' | jq . 2>/dev/null || true