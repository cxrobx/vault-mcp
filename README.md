# vault-mcp

Semantic search MCP server for the CX Obsidian vault (`~/Documents/CX`).
Embeds every note locally with Ollama (`nomic-embed-text`, 768-dim) and
exposes meaning-based retrieval to every Claude Code session — the AI-side
counterpart to Smart Connections' in-app Connections pane. Fully local,
zero recurring cost.

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
- **Scope**: all `*.md` under the vault, **following symlinks** (the Finances
  Wiki and XRF client folders are symlinks into other repos), with a realpath
  cycle guard. Excluded: dot-dirs (`.obsidian`, `.smart-env`, `.trash`,
  `.git`, `.SynologyWorkingDirectory`) and `Other/Templates`.
- **Freshness**: every server start runs a delta reindex in a background
  thread (mtime+size sweep → content-hash confirm → re-embed changed, delete
  removed). Searches during a reindex use the current index. Per-file
  transactions — a crash never half-indexes a file.

## Tools

| Tool | What it does |
|------|--------------|
| `search_vault(query, k=8, folder=None)` | Semantic top-k over all chunks; `folder` prefix-filters (e.g. `"Projects/CXVentures"`). |
| `related_notes(note_path, k=8)` | Mean of the note's chunk vectors → nearest other notes (best-chunk score per note). |
| `reindex_vault(full=False)` | Delta (or full) reindex now; returns files scanned/changed, chunks embedded, duration. |
| `vault_stats()` | File/chunk counts, model, last index time, DB size, startup-reindex status. |

## Setup

```bash
cd ~/Projects/vault-mcp
uv venv && uv pip install -e .

# initial full index (requires Ollama running with nomic-embed-text pulled)
.venv/bin/python -m vault_mcp index --full

# register user-scope
claude mcp add --scope user vault -- ~/Projects/vault-mcp/.venv/bin/python -m vault_mcp
```

## Config (env vars)

| Var | Default |
|-----|---------|
| `VAULT_MCP_VAULT` | `~/Documents/CX` |
| `VAULT_MCP_DB` | `<repo>/data/index.db` |
| `VAULT_MCP_OLLAMA` | `http://localhost:11434` |
| `VAULT_MCP_MODEL` | `nomic-embed-text` |

## Layout

```
src/vault_mcp/
  __main__.py    # `python -m vault_mcp` → server; `index [--full]` → CLI indexer
  server.py      # FastMCP("vault"), 4 tools, background startup delta reindex
  indexer.py     # vault walk, chunking, delta logic
  embeddings.py  # Ollama /api/embed client (batched, prefixed, clear errors)
  store.py       # SQLite schema + numpy cosine search
```
