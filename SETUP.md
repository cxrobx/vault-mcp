# vault-mcp — Setup (agent-executable)

**Who this is for:** an AI coding agent (e.g. Claude Code) installing vault-mcp
on a user's Mac, pointed at that user's markdown vault.

**What the user gets when done:** four semantic-search tools inside every
Claude Code session — `search_vault`, `related_notes`, `reindex_vault`,
`vault_stats` — running entirely locally (Ollama embeddings, SQLite index, no
cloud calls, no API cost).

---

## Prerequisites

1. **macOS** (Linux works too) with **Claude Code** installed (`claude --version` works).
2. **Python 3.11+** available as `python3`, `python3.11`, `python3.12`, or
   `python3.13`. If missing: `brew install python@3.12`.
3. About **1 GB free disk** for Ollama + the embedding model.
4. The path to the vault — any folder of `.md` files. Referred to below as
   `<VAULT>`. Use an absolute path.

## Step 1 — Install and start Ollama

```bash
# Either the Homebrew CLI:
brew install ollama
ollama serve &          # leave running; or use `brew services start ollama`

# Or the desktop app: download from https://ollama.com, open it once
# (menu-bar icon = server is running).
```

Verify the server is up:

```bash
curl -sf http://localhost:11434/api/tags >/dev/null && echo OK
```

## Step 2 — Clone and run the installer

```bash
git clone https://github.com/cxrobx/vault-mcp ~/Projects/vault-mcp
cd ~/Projects/vault-mcp
VAULT_MCP_VAULT="<VAULT>" bash scripts/setup.sh
```

The script is idempotent (safe to re-run). It creates `.venv`, installs the
package, pulls `nomic-embed-text` if missing, and builds the full index. On a
few hundred notes that's well under a minute; thousands of notes take a few
minutes.

## Step 3 — Register the MCP server (user scope)

Run the command `setup.sh` printed at the end — it looks like this, with the
absolute paths filled in:

```bash
claude mcp add-json vault --scope user '{"command":"/ABS/PATH/vault-mcp/.venv/bin/python","args":["-m","vault_mcp"],"env":{"VAULT_MCP_VAULT":"<VAULT>"}}'
```

Confirm it registered:

```bash
claude mcp get vault
```

### Fallback — manual config

If the CLI route fails, merge this under `mcpServers` in `~/.claude.json`
(don't overwrite existing entries):

```jsonc
"vault": {
  "type": "stdio",
  "command": "/ABS/PATH/vault-mcp/.venv/bin/python",
  "args": ["-m", "vault_mcp"],
  "env": { "VAULT_MCP_VAULT": "<VAULT>" }
}
```

## Step 4 — Restart and verify

> **Agent:** MCP servers load at session start — the user must restart Claude
> Code (or start a new session) before the tools appear.

After restart, in Claude Code:

1. `vault_stats` → `files_indexed` should match the vault's note count and
   `vault` should be `<VAULT>`.
2. `search_vault` with a phrase you know is in some note → that note should be
   a top hit.

Both pass = setup complete.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ERROR: Ollama not found / not running` | Step 1 skipped. Install/start Ollama; verify with the `curl` check above. |
| `ERROR: Python 3.11+ not found` | `brew install python@3.12`, open a new shell, re-run `setup.sh`. |
| `vault_stats` shows 0 files or the wrong `vault` path | `VAULT_MCP_VAULT` wrong or unset in the registration `env`. Fix the registration, restart, run `reindex_vault(full=True)`. |
| Vault tools don't appear after restart | `claude mcp list` — if `vault` is missing, redo Step 3. If listed but failing, run the command manually to see the error: `/ABS/PATH/.venv/bin/python -m vault_mcp` (Ctrl-C to exit). |
| Model pull is slow / fails | Network issue; re-run `ollama pull nomic-embed-text` — it resumes. |
| Notes changed but search is stale | A delta reindex runs at every server start; force one with `reindex_vault()`. |

## Reference

- Env vars: `VAULT_MCP_VAULT` (vault path), `VAULT_MCP_DB` (index location,
  default `<repo>/data/index.db`), `VAULT_MCP_OLLAMA`, `VAULT_MCP_MODEL`.
- The index DB is local and disposable — deleting `data/index.db` just forces
  a full re-embed on next index.
- Freshness: every MCP server start runs a background delta reindex (changed
  files re-embedded, deleted files dropped). No cron needed.
