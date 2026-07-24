# vault-mcp

Semantic search MCP server for a local markdown vault (Obsidian or any folder
of `.md` files). Embeds every note locally with Ollama (`nomic-embed-text`,
768-dim) and exposes meaning-based retrieval to Claude Code (or any MCP
client). Fully local, zero API cost, no cloud calls.

Point it at a vault and your agent can `search_vault("how did we decide on
the deployment setup")` instead of grepping for keywords.

## Quickstart

Requirements: macOS or Linux, **Python 3.11+**, [Ollama](https://ollama.com).

```bash
# 1. Ollama + the embedding model
brew install ollama            # or download Ollama.app from https://ollama.com
ollama serve &                 # skip if Ollama.app is already running (menu-bar icon)
ollama pull nomic-embed-text   # ~270 MB, one-time

# 2. Clone + run the installer (creates .venv, installs, builds the index)
git clone https://github.com/cxrobx/vault-mcp && cd vault-mcp
VAULT_MCP_VAULT="$HOME/path/to/your/vault" bash scripts/setup.sh
```

The installer finishes by printing the exact `claude mcp add-json` command
(with absolute paths resolved) to register the server user-scope. Note the
registration embeds the absolute path to `.venv/bin/python` inside the clone —
if you move the repo, re-run the registration.

> **`VAULT_MCP_VAULT` is required** unless your vault happens to live at the
> author's default, `~/Documents/CX`. Set it both when indexing and in the MCP
> registration's `env` block.

Setting up by hand instead of `setup.sh`:

```bash
uv venv && uv pip install -e .                       # or: python3 -m venv .venv && .venv/bin/pip install -e .
VAULT_MCP_VAULT="$HOME/path/to/your/vault" .venv/bin/python -m vault_mcp index --full
claude mcp add-json vault --scope user "{\"command\":\"$PWD/.venv/bin/python\",\"args\":[\"-m\",\"vault_mcp\"],\"env\":{\"VAULT_MCP_VAULT\":\"$HOME/path/to/your/vault\"}}"
```

Restart Claude Code, then verify: ask it to run `vault_stats` (should show your
file count) and `search_vault` for something you know is in a note.

**[SETUP.md](SETUP.md)** is a step-by-step version of the above written for an
AI coding agent to execute — hand it to Claude Code and it can do the whole
install.

## Tools

| Tool | What it does |
|------|--------------|
| `search_vault(query, k=8, folder=None)` | Semantic top-k over all chunks; `folder` prefix-filters (e.g. `"topics"` or `"Projects/alpha"`). |
| `related_notes(note_path, k=8)` | Mean of the note's chunk vectors → nearest other notes (best-chunk score per note). |
| `reindex_vault(full=False)` | Delta (or full) reindex now; returns files scanned/changed, chunks embedded, duration. |
| `vault_stats()` | File/chunk counts, model, last index time, DB size, startup-reindex status. |

## How it works

- **Embeddings**: Ollama `POST /api/embed` with nomic task prefixes
  (`search_document: ` at index time, `search_query: ` at query time — the
  prefixes matter for retrieval quality).
- **Store**: plain SQLite (`data/index.db`, gitignored) with float32 BLOB
  embeddings, L2-normalized at write. Search is a numpy dot product over the
  whole chunk matrix — at ~7k chunks that's ~21 MB and <20 ms, no vector
  extension needed.
- **Chunking**: split on H1–H3 headings (outside code fences); each chunk is
  embedded as `"{note title} > {heading path}\n{body}"`; oversized sections
  split on paragraph boundaries at ~2000 chars; chunks under 80 chars are
  skipped; frontmatter is parsed for tags, not embedded.
- **Scope**: all `*.md` under the vault, **following symlinks** (symlinked
  folders index like real ones), with a realpath cycle guard. Excluded:
  dot-dirs (`.obsidian`, `.smart-env`, `.trash`, `.git`,
  `.SynologyWorkingDirectory`) and `Other/Templates`.
- **Freshness**: every server start runs a delta reindex in a background
  thread (mtime+size sweep → content-hash confirm → re-embed changed, delete
  removed). Searches during a reindex use the current index. Per-file
  transactions — a crash never half-indexes a file.

## Config (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `VAULT_MCP_VAULT` | `~/Documents/CX` | Path to your vault — set this. |
| `VAULT_MCP_DB` | `<repo>/data/index.db` | One DB per instance; don't share it between two servers. |
| `VAULT_MCP_OLLAMA` | `http://localhost:11434` | |
| `VAULT_MCP_MODEL` | `nomic-embed-text` | |

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot reach Ollama" / connection refused | Start Ollama.app (menu-bar icon) or `ollama serve`. Verify: `curl -s localhost:11434/api/tags`. |
| Error mentioning the model / 404 from Ollama | `ollama pull nomic-embed-text`. |
| `vault_stats` shows 0 files | `VAULT_MCP_VAULT` points at the wrong path (check the `vault` field in the output), or the index never ran — run `reindex_vault(full=True)` or the CLI `index --full`. |
| Vault tools missing in Claude Code | Registration is user-scope in `~/.claude.json` — check `claude mcp list`, then restart the session. |
| Stale results after editing notes | Delta reindex runs on every server start; force one anytime with `reindex_vault()`. |

## Layout

```
src/vault_mcp/
  __main__.py    # `python -m vault_mcp` → server; `index [--full]` → CLI indexer
  server.py      # FastMCP("vault"), 4 tools, background startup delta reindex
  indexer.py     # vault walk, chunking, delta logic
  embeddings.py  # Ollama /api/embed client (batched, prefixed, clear errors)
  store.py       # SQLite schema + numpy cosine search
scripts/
  setup.sh       # idempotent installer: venv → deps → Ollama check → full index
```

## License

MIT
