"""FastMCP server: semantic search tools over a local markdown vault.

Startup kicks off a delta reindex in a background thread, so the server is
responsive immediately; searches during a reindex use the current index.
"""

import threading

import numpy as np
from mcp.server.fastmcp import FastMCP

from . import DB_PATH, VAULT_PATH
from .embeddings import EmbeddingsUnavailable, OllamaEmbedder
from .indexer import reindex
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
    return [
        {
            "path": h["path"],
            "heading": h["heading_path"],
            "snippet": _snippet(h["text"]),
            "score": round(h["score"], 4),
            "tags": h["tags"],
        }
        for h in hits
    ]


@mcp.tool()
def search_vault(query: str, k: int = 8, folder: str | None = None) -> dict:
    """Semantic search over the configured markdown vault.

    Finds notes by meaning, not keywords — use natural-language queries
    ("pricing strategy for service deals", "how we decided on the deployment
    setup"). Returns the top-k chunks with note path, heading, snippet,
    similarity score, and tags.

    Args:
        query: Natural-language search query.
        k: Number of results to return (default 8).
        folder: Optional vault-relative folder prefix to restrict the search,
            e.g. "topics" or "Projects/alpha".
    """
    try:
        qvec = embedder.embed_query(query)
    except EmbeddingsUnavailable as exc:
        return {"error": str(exc)}
    hits = store.search(qvec, k=k, folder=folder)
    result: dict = {"results": _format_hits(hits)}
    if not hits and folder:
        result["note"] = f"No indexed notes under folder '{folder}' — check the prefix (case-sensitive)."
    return result


@mcp.tool()
def related_notes(note_path: str, k: int = 8) -> dict:
    """Find vault notes semantically related to a given note (the DIY Connections pane).

    Averages the note's chunk embeddings and returns the k nearest other
    notes (best-chunk score per note). Accepts a vault-relative path
    ("Projects/alpha/launch-plan.md") or just a filename —
    ambiguous names return candidates.

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
    """Index health: file/chunk counts, embedding model, last index time,
    DB size, and the status of the startup delta reindex."""
    stats = store.stats()
    return {
        "vault": str(VAULT_PATH),
        "files_indexed": stats["files"],
        "chunks": stats["chunks"],
        "db_size_mb": round(stats["db_bytes"] / 1e6, 1),
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
