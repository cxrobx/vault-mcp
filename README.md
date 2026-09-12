# vault-mcp

Hybrid search MCP server for a local markdown vault (Obsidian or any folder
of `.md` files), including the HTML pages in it and in any folder you mount
beside it. Embeds every note locally with Ollama (`nomic-embed-text`,
768-dim), indexes it for keyword search with SQLite FTS5/BM25, and exposes
both to Claude Code (or any MCP client). Fully local, zero API cost, no cloud
calls.

Point it at a vault and your agent can `search_vault("how did we decide on
the deployment setup")` instead of grepping for keywords — and still find
`PostGIS` or `nas-tunnel` by exact name, which pure embedding search is
notoriously bad at.

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
| `search_vault(query, k=8, folder=None)` | Hybrid (dense + BM25) top-k over all chunks; `folder` prefix-filters (e.g. `"topics"` or `"Projects/alpha"`). |
| `related_notes(note_path, k=8)` | Mean of the note's chunk vectors → nearest other notes (best-chunk score per note). Dense only — see below. |
| `reindex_vault(full=False)` | Delta (or full) reindex now; returns files scanned/changed, chunks embedded, duration. |
| `vault_stats()` | File/chunk counts, model, last index time, DB size, FTS health, startup-reindex status. |

Each hit carries `retrieval` (`dense` / `lexical` / `hybrid`) saying which leg
found it, and the response carries `scoring` naming the scale `score` is on.

## How it works

### Retrieval

Two legs, fused by **Reciprocal Rank Fusion** (`1/(60+rank)` summed per doc):

- **Dense** — cosine over the embedding matrix. Handles meaning, paraphrase,
  and "notes about X" questions.
- **Lexical** — SQLite FTS5 `bm25()` over chunk text + tags (weights 1.0 /
  0.5). Handles exact literals: identifiers, product names, hyphenated slugs.

RRF rather than a weighted sum of the two scores, because BM25 is unbounded
and corpus-dependent while cosine is [-1, 1] — summing them is just BM25 with
rounding noise, and the dense leg stops mattering. RRF only compares rank
positions, so the scales never have to be reconciled.

The lexical leg **ANDs** the query terms, which makes it self-gating: a literal
lookup (`PostGIS`) matches precisely, while a discursive query has no chunk
containing all its terms, so the leg returns nothing and search falls back to
pure dense — the regime where dense was already right.

`scripts/eval_retrieval.py` measures both halves of that trade — run it before
and after any ranking change. On the author's vault (~450 notes / ~7.9k chunks),
over 14 auto-discovered rare identifiers:

|  | literal hit@1 |
|---|---|
| dense only | 2/14 |
| **hybrid** | **14/14** |

The choice of connector is what makes or breaks the natural-language side. On a
curated probe set (14 rare literals / 10 NL queries):

| lexical mode | literal hit@1 | NL top-1 agreement |
|---|---|---|
| OR | 13/14 | 1/10 ✗ wrecks NL |
| AND → OR fallback | 14/14 | 1/10 ✗ fallback fires exactly where OR is worst |
| **AND** | **14/14** | **9/10** |

An OR over six common words drags in every chunk that merely says "leaving" or
"retainer"; with the two candidate lists barely overlapping, RRF ties then hand
rank 1 to that noise.

Query text is never passed to `MATCH` raw. FTS5's query parser only accepts
bareword alphanumerics, so `nas-tunnel` raises `fts5: syntax error` and
anything with `:` is read as a column filter; terms are re-quoted into a safe
expression, which doubles as the injection guard. The **raw** query goes to
BM25 — not the `search_query: `-prefixed string, which would poison every
lexical query with two junk tokens.

`related_notes` stays deliberately dense-only: it is note-to-note similarity
driven by a mean embedding, with no query string to give BM25. Synthesizing a
pseudo-query from the note's own terms mostly retrieves notes sharing its
boilerplate (same client, same template, same tag header) rather than its
meaning.

### Storage

- **Embeddings**: Ollama `POST /api/embed` with nomic task prefixes
  (`search_document: ` at index time, `search_query: ` at query time — the
  prefixes matter for retrieval quality).
- **Store**: plain SQLite (`data/index.db`, gitignored) with float32 BLOB
  embeddings, L2-normalized at write. The dense leg is a numpy dot product over
  the whole chunk matrix — at ~8k chunks that's ~24 MB and <20 ms, no vector
  extension needed.
- **FTS index**: an external-content FTS5 table (`chunks_fts`) — postings only,
  column values read back from `chunks`, so no text is duplicated. Kept in
  lockstep by insert/update/delete triggers, so the delta reindex maintains it
  without knowing it exists. Missing or drifted indexes are rebuilt on open;
  an existing DB upgrades in place in well under a second and **never
  re-embeds**. `vault_stats()` reports FTS health via the `chunks_fts_docsize`
  shadow table and FTS5's own `integrity-check` — note that
  `SELECT count(*) FROM chunks_fts` is a false green, since on an
  external-content table it is answered from the content table and matches
  `chunks` even when the index is empty.
- **Chunking**: split on H1–H3 headings (outside code fences); each chunk is
  embedded as `"{note title} > {heading path}\n{body}"`; oversized sections
  split on paragraph boundaries at ~2000 chars; chunks under 80 chars are
  skipped; frontmatter is parsed for tags, not embedded. HTML pages split the
  same way on `<h1>`–`<h3>`, over the text a reader sees (`<script>`,
  `<style>`, `<head>`, `<nav>` and comments dropped, each block a paragraph),
  titled by their `<title>`; a page over 4 MB is kept with no chunks.
- **Scope**: all `*.md` under the vault, plus its `*.html`/`*.htm` notes
  except one sitting beside a same-name `.md` (that note's rendering),
  **following symlinks** (symlinked folders index like real ones), with a
  realpath cycle guard. Then each mount's HTML pages under its name, listed
  the way Onyx's Artifacts sidebar lists them: below the project level, a
  folder holding `index.html` is one page. A page a mount reaches that the
  vault already gave is skipped. Excluded: dot-dirs (`.obsidian`,
  `.smart-env`, `.trash`, `.git`, `.SynologyWorkingDirectory`),
  `node_modules`, `__pycache__`, and `Other/Templates`.
- **Freshness**: every server start runs a delta reindex in a background
  thread (mtime+size sweep → content-hash confirm → re-embed changed, delete
  removed). Searches during a reindex use the current index. Per-file
  transactions — a crash never half-indexes a file.

## Config (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `VAULT_MCP_VAULT` | `~/Documents/CX` | Path to your vault — set this. |
| `VAULT_MCP_MOUNTS` | `Artifacts=~/Documents/Artifacts` | Extra folders indexed beside the vault: `Name=/path` pairs joined by `:`. Their HTML pages appear under `Name/…` (so `folder="Name"` searches just them). A missing folder is skipped; `""` turns mounts off. |
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
  store.py       # SQLite schema, FTS5 index, dense + BM25 legs, RRF fusion
scripts/
  setup.sh           # idempotent installer: venv → deps → Ollama check → full index
  eval_retrieval.py  # literal-recall + NL-regression eval; run around ranking changes
```

## License

MIT
