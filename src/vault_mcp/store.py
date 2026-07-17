"""SQLite persistence + in-memory numpy cosine search.

Embeddings are stored as float32 BLOBs, L2-normalized at write time, so
search is a single matrix–vector dot product. At vault scale (~2–3k chunks,
~9 MB matrix) brute force is <10 ms — no vector extension needed.

Every operation opens its own connection (WAL mode), so the background
reindex thread and tool calls never share a connection. The chunk matrix is
cached in memory and invalidated on any write that touches chunks.
"""

import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path         TEXT PRIMARY KEY,
    mtime        REAL NOT NULL,
    size         INTEGER NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path    TEXT NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    tags         TEXT NOT NULL DEFAULT '',
    embedding    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict | None = None
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None

    # ---- writes (indexer) ----------------------------------------------

    def replace_file(self, path: str, mtime: float, size: int, content_hash: str, chunks: list[dict]) -> None:
        """Upsert a file row and atomically replace its chunks."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """INSERT INTO files(path, mtime, size, content_hash) VALUES(?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET
                     mtime=excluded.mtime, size=excluded.size, content_hash=excluded.content_hash""",
                (path, mtime, size, content_hash),
            )
            conn.execute("DELETE FROM chunks WHERE file_path=?", (path,))
            conn.executemany(
                "INSERT INTO chunks(file_path, heading_path, text, tags, embedding) VALUES(?,?,?,?,?)",
                [
                    (path, c["heading_path"], c["text"], c["tags"],
                     np.asarray(c["embedding"], dtype=np.float32).tobytes())
                    for c in chunks
                ],
            )
        self.invalidate()

    def touch_file(self, path: str, mtime: float, size: int) -> None:
        """Stat changed but content hash didn't — refresh stat, keep chunks."""
        with closing(self._connect()) as conn, conn:
            conn.execute("UPDATE files SET mtime=?, size=? WHERE path=?", (mtime, size, path))

    def delete_files(self, paths: list[str]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executemany("DELETE FROM chunks WHERE file_path=?", [(p,) for p in paths])
            conn.executemany("DELETE FROM files WHERE path=?", [(p,) for p in paths])
        self.invalidate()

    def all_file_stats(self) -> dict[str, tuple[float, int, str]]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT path, mtime, size, content_hash FROM files").fetchall()
        return {r[0]: (r[1], r[2], r[3]) for r in rows}

    # ---- meta -----------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, value))

    # ---- reads (search) -------------------------------------------------

    def _get_cache(self) -> dict:
        with self._lock:
            if self._cache is None:
                self._cache = self._load()
            return self._cache

    def _load(self) -> dict:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT file_path, heading_path, text, tags, embedding FROM chunks ORDER BY id"
            ).fetchall()
        if not rows:
            return {"paths": [], "headings": [], "texts": [], "tags": [],
                    "matrix": np.zeros((0, 1), dtype=np.float32)}
        matrix = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        return {
            "paths": [r[0] for r in rows],
            "headings": [r[1] for r in rows],
            "texts": [r[2] for r in rows],
            "tags": [r[3] for r in rows],
            "matrix": matrix,
        }

    def search(self, qvec: np.ndarray, k: int = 8, folder: str | None = None) -> list[dict]:
        cache = self._get_cache()
        n = cache["matrix"].shape[0]
        if n == 0:
            return []
        scores = cache["matrix"] @ qvec.astype(np.float32)
        if folder:
            prefix = folder.strip("/") + "/"
            keep = np.fromiter(
                (p.startswith(prefix) for p in cache["paths"]), dtype=bool, count=n
            )
            scores = np.where(keep, scores, -np.inf)
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            {
                "path": cache["paths"][i],
                "heading_path": cache["headings"][i],
                "text": cache["texts"][i],
                "tags": cache["tags"][i],
                "score": float(scores[i]),
            }
            for i in top
            if np.isfinite(scores[i])
        ]

    def related_notes(self, qvec: np.ndarray, exclude_path: str, k: int = 8) -> list[dict]:
        """Nearest notes (aggregated per note, best-chunk score), excluding the source note."""
        cache = self._get_cache()
        if cache["matrix"].shape[0] == 0:
            return []
        scores = cache["matrix"] @ qvec.astype(np.float32)
        best: dict[str, tuple[float, int]] = {}
        for i, path in enumerate(cache["paths"]):
            if path == exclude_path:
                continue
            s = float(scores[i])
            if path not in best or s > best[path][0]:
                best[path] = (s, i)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        return [
            {
                "path": path,
                "heading_path": cache["headings"][i],
                "text": cache["texts"][i],
                "tags": cache["tags"][i],
                "score": s,
            }
            for path, (s, i) in ranked
        ]

    def file_vectors(self, path: str) -> np.ndarray | None:
        cache = self._get_cache()
        idx = [i for i, p in enumerate(cache["paths"]) if p == path]
        if not idx:
            return None
        return cache["matrix"][idx]

    def resolve_note(self, note_path: str) -> tuple[str | None, list[str]]:
        """Resolve a user-supplied path to an indexed vault-relative path.

        Returns (resolved, candidates): exact match first, then case-insensitive
        suffix match, then filename-stem match. candidates is filled when the
        match is ambiguous or absent.
        """
        cache = self._get_cache()
        paths = sorted(set(cache["paths"]))
        if note_path in paths:
            return note_path, []
        q = note_path.strip("/").lower()
        if not q.endswith(".md"):
            q += ".md"
        matches = [p for p in paths if p.lower() == q or p.lower().endswith("/" + q)]
        if not matches:
            stem = Path(q).stem
            matches = [p for p in paths if Path(p).stem.lower() == stem]
        if len(matches) == 1:
            return matches[0], []
        return None, matches[:10]

    def stats(self) -> dict:
        with closing(self._connect()) as conn:
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        size = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.db_path) + suffix)
            if p.exists():
                size += p.stat().st_size
        return {"files": files, "chunks": chunks, "db_bytes": size}
