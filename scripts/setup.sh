#!/usr/bin/env bash
# vault-mcp installer — idempotent, safe to re-run.
#
# Usage:
#   VAULT_MCP_VAULT="/path/to/your/vault" bash scripts/setup.sh
#   bash scripts/setup.sh /path/to/your/vault
#
# Does: python check → .venv → editable install → Ollama + model check
# (pulls the model if missing) → full index → prints the exact Claude Code
# registration command with absolute paths resolved.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_URL="${VAULT_MCP_OLLAMA:-http://localhost:11434}"
MODEL="${VAULT_MCP_MODEL:-nomic-embed-text}"
VAULT="${VAULT_MCP_VAULT:-${1:-}}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- vault path ---
[ -n "$VAULT" ] || die "No vault path given. Run: VAULT_MCP_VAULT=\"/path/to/vault\" bash scripts/setup.sh"
VAULT="${VAULT/#\~/$HOME}"
VAULT="$(cd "$VAULT" 2>/dev/null && pwd)" || die "Vault directory not found: $VAULT"

# --- python >= 3.11 ---
say "Checking Python (need 3.11+)"
PY=""
for cand in python3.13 python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1 \
     && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$(command -v "$cand")"
    break
  fi
done
if [ -n "$PY" ]; then
  echo "Using $PY ($("$PY" --version 2>&1))"
elif command -v uv >/dev/null 2>&1; then
  # uv can download a managed CPython — no system Python needed
  echo "No system Python 3.11+; using a uv-managed Python 3.12."
else
  die "Python 3.11+ not found. Either: brew install python@3.12 — or install uv (curl -LsSf https://astral.sh/uv/install.sh | sh; open a new shell), which provisions Python itself. Then re-run."
fi

# --- venv + install ---
say "Installing vault-mcp into $REPO_DIR/.venv"
cd "$REPO_DIR"
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv --python "${PY:-3.12}"
  uv pip install -q -e .
else
  [ -d .venv ] || "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e .
fi

# --- Ollama running? ---
say "Checking Ollama at $OLLAMA_URL"
if ! curl -sf --max-time 3 "$OLLAMA_URL/api/tags" >/dev/null; then
  if command -v ollama >/dev/null 2>&1 || [ -d /Applications/Ollama.app ]; then
    die "Ollama is installed but not running. Start Ollama.app (menu-bar icon) or run 'ollama serve' in another terminal, then re-run this script."
  fi
  die "Ollama not found. Install it (brew install ollama, or download from https://ollama.com), start it, then re-run this script."
fi

# --- model pulled? ---
if ! curl -sf "$OLLAMA_URL/api/tags" | grep -q "\"$MODEL"; then
  say "Pulling embedding model $MODEL (~270 MB, one-time)"
  ollama pull "$MODEL"
else
  echo "Model $MODEL already pulled."
fi

# --- full index ---
say "Indexing $VAULT (full — a large vault takes a few minutes)"
VAULT_MCP_VAULT="$VAULT" .venv/bin/python -m vault_mcp index --full

# --- registration ---
say "Done. Register with Claude Code (user scope — available in every session):"
cat <<EOF

  claude mcp add-json vault --scope user '{"command":"$REPO_DIR/.venv/bin/python","args":["-m","vault_mcp"],"env":{"VAULT_MCP_VAULT":"$VAULT"}}'

Then restart Claude Code and verify by asking it to run vault_stats.
EOF
