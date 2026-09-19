"""FastMCP server: semantic search tools over a local markdown/HTML vault.

Startup kicks off a delta reindex in a background thread, so the server is
responsive immediately; searches during a reindex use the current index.
"""

import threading

import numpy as np
from mcp.server.fastmcp import FastMCP

from . import DB_PATH, VAULT_PATH
from .embeddings import EmbeddingsUnavailable, OllamaEmbedder
from .indexer import mount_status, reindex
from .store import Store

mcp = FastMCP("vault")
store = Store(DB_PATH)
embedder = OllamaEmbedder()

SNIPPET_CHARS = 400

# Filled in by the startup delta-reindex thread; surfaced via vault_stats.
_startup = {"status": "not started"}


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …"


def _format_hits(hits: list[dict]) -> list[dict]:
    out = []
    for h in hits:
        hit = {
            "path": h["path"],
            "heading": h["heading_path"],
            "snippet": _snippet(h["text"]),
            "score": round(h["score"], 4),
            "tags": h["tags"],
        }
        # Present only on hybrid search: which leg(s) surfaced the hit, and the
        # cosine behind it. `score` is an RRF score there — a small number on a
        # different scale from cosine, comparable only within one result set.
        if "retrieval" in h:
            hit["retrieval"] = h["retrieval"]
        if "cosine" in h:
            hit["cosine"] = round(h["cosine"], 4)
        out.append(hit)
    return out


@mcp.tool()
def search_vault(query: str, k: int = 8, folder: str | None = None) -> dict:
    """Hybrid search over the configured vault.

    Covers the vault's markdown and HTML notes, plus any mounted folder, whose
    paths start with the mount's name rather than a vault folder's (e.g.
    "Artifacts/Learnings/…"). A "pages" mount gives HTML pages; a "notes" mount
    is a folder of documents outside the vault — a repo's docs, a client
    folder — indexed the way the vault is. vault_stats lists the mounts, their
    kind and where they live; pass a mount's name as `folder` to search only
    it, or a vault folder to keep mounted documents out of the results.

    Runs two retrieval legs and fuses them, so it handles both ends of the
    query spectrum: natural-language questions ("pricing strategy for service
    deals", "how we decided on the deployment setup") are answered by
    embeddings, and rare literal tokens (identifiers, product names,
    hyphenated slugs like "nas-tunnel") are answered by BM25 keyword match.
    Returns the top-k chunks with note path, heading, snippet, score, and tags.

    Args:
        query: Search query — natural language or a literal term.
        k: Number of results to return (default 8).
        folder: Optional folder prefix to restrict the search, e.g. "topics",
            "Projects/alpha", or a mount name like "Artifacts".
    """
    try:
        qvec = embedder.embed_query(query)
    except EmbeddingsUnavailable as exc:
        return {"error": str(exc)}
    # The RAW query goes to the lexical leg. embed_query() internally prepends
    # nomic-embed-text's "search_query: " task prefix; that prefix is an
    # artifact of the embedding model and would poison every BM25 query with
    # two junk high-frequency tokens.
    hits = store.search(qvec, k=k, folder=folder, query_text=query)
    formatted = _format_hits(hits)
    # `score` changes scale depending on whether the lexical leg fired, so say
    # which it is. RRF scores are ~1/60 and only comparable within one result
    # set — without this an agent reads 0.016 as "weak hit" and discards a
    # perfect keyword match.
    fused = any(h.get("retrieval") in ("hybrid", "lexical") for h in formatted)
    result: dict = {
        "results": formatted,
        "scoring": "rrf (rank-fused; compare only within this result set)"
        if fused
        else "cosine (0-1; lexical leg found no match, dense only)",
    }
    if not hits and folder:
        result["note"] = f"No indexed notes under folder '{folder}' — check the prefix (case-sensitive)."
    return result


@mcp.tool()
def related_notes(note_path: str, k: int = 8) -> dict:
    """Find vault notes semantically related to a given note (the DIY Connections pane).

    Averages the note's chunk embeddings and returns the k nearest other
    notes (best-chunk score per note). Accepts a vault-relative path
    ("Projects/alpha/launch-plan.md", "Artifacts/…/page.html") or just a
    filename — ambiguous names return candidates.

    Args:
        note_path: Vault-relative path or filename of the source note.
        k: Number of related notes to return (default 8).
    """
    resolved, candidates = store.resolve_note(note_path)
    if resolved is None:
        if candidates:
            return {"error": f"Ambiguous note '{note_path}'.", "candidates": candidates}
        return {"error": f"No indexed note matches '{note_path}'."}
    vectors = store.file_vectors(resolved)
    if vectors is None or vectors.shape[0] == 0:
        return {"error": f"Note '{resolved}' has no indexed chunks (too short or empty)."}
    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm > 0:
        mean = mean / norm
    hits = store.related_notes(mean.astype(np.float32), exclude_path=resolved, k=k)
    return {"note": resolved, "related": _format_hits(hits)}


@mcp.tool()
def reindex_vault(full: bool = False) -> dict:
    """Reindex the vault now. Delta by default (fast — only changed/removed
    files are re-embedded); full=True re-embeds everything (takes minutes).

    Args:
        full: Re-embed every note instead of only changed ones.
    """
    try:
        return reindex(store, embedder, full=full)
    except (EmbeddingsUnavailable, RuntimeError) as exc:
        return {"error": str(exc)}


@mcp.tool()
def vault_stats() -> dict:
    """Index health: file/chunk counts, mounted folders, embedding model, last
    index time, DB size, and the status of the startup delta reindex."""
    stats = store.stats()
    return {
        "vault": str(VAULT_PATH),
        "mounts": mount_status(),
        "files_indexed": stats["files"],
        "chunks": stats["chunks"],
        "db_size_mb": round(stats["db_bytes"] / 1e6, 1),
        "fts": stats["fts"],
        "model": store.get_meta("model"),
        "last_index_time": store.get_meta("last_index_time"),
        "startup_reindex": dict(_startup),
    }


def _startup_reindex() -> None:
    _startup.clear()
    _startup["status"] = "running"
    try:
        result = reindex(store, embedder, full=False)
        _startup.clear()
        _startup.update(status="ok", **result)
    except Exception as exc:  # surfaced via vault_stats, never kills the server
        _startup.clear()
        _startup.update(status="error", error=str(exc))


def run() -> None:
    """Run the MCP server over stdio."""
    threading.Thread(target=_startup_reindex, daemon=True, name="vault-startup-reindex").start()
    mcp.run()
