"""SQLite persistence + hybrid (dense + lexical) retrieval.

Two retrieval legs, fused by Reciprocal Rank Fusion:

* **Dense** — embeddings stored as float32 BLOBs, L2-normalized at write time,
  so scoring is a single matrix–vector dot product. At vault scale (~8k chunks,
  ~24 MB matrix) brute force is <10 ms — no vector extension needed.
* **Lexical** — an FTS5 index over chunk text + tags, ranked by SQLite's
  built-in BM25. Catches the rare literal tokens (identifiers, product names,
  hyphenated slugs) that a 768-dim embedding smears into its neighbourhood.

The two are combined with RRF rather than a weighted score sum, because BM25 is
unbounded and corpus-dependent while cosine lives in [-1, 1]: any sum of the two
is just BM25 with rounding noise, and the dense leg stops contributing. RRF only
ever looks at rank position, so the scales never have to be reconciled.

Every operation opens its own connection (WAL mode), so the background
reindex thread and tool calls never share a connection. The chunk matrix is
cached in memory and invalidated on any write that touches chunks. The FTS
index is maintained by triggers, so the indexer's delete/insert delta path
keeps it in lockstep without knowing it exists.
"""

import re
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import numpy as np

# Bump when the FTS table definition (tokenizer, indexed columns) changes —
# a mismatch against meta['fts_version'] forces a rebuild on next open.
FTS_VERSION = "1"

# RRF constant. 60 is the value from Cormack et al. 2009 and the de-facto
# default; it damps the contribution of the head so a single leg's rank-1 hit
# can't unilaterally win a fused ranking.
RRF_K = 60
# Candidates pulled from each leg before fusion.
CANDIDATE_K = 50
# Ceiling on OR-ed terms handed to FTS5, so a pathological query can't build a
# huge match expression.
MAX_FTS_TERMS = 32

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

# External-content FTS5: the index stores only postings, and reads column
# values back from `chunks` via content_rowid=id. No text is duplicated.
#
# No porter stemmer on purpose. The lexical leg exists to nail exact literals;
# morphological and synonym matching is the dense leg's job, and stemming
# collides distinct identifiers (Typst/typ, Kokoro/kokor) for no gain here.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, tags,
    content='chunks',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);
CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, tags)
        VALUES ('delete', old.id, old.text, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, tags)
        VALUES ('delete', old.id, old.text, old.tags);
    INSERT INTO chunks_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
END;
"""

# A token as unicode61 sees it: runs of alphanumerics/underscore, plus any
# non-ASCII (CJK, accented latin) which unicode61 also treats as word chars.
# Every ASCII separator — crucially the double quote — is excluded, so the
# runs matched here can be wrapped in quotes with no escaping needed.
_FTS_TOKEN_RE = re.compile(r"[0-9A-Za-z_-￿]+")


def build_fts_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Raw text cannot be handed to MATCH. FTS5's *query parser* (a separate
    thing from its tokenizer) only accepts barewords of alphanumerics and
    underscore, so `nas-tunnel` raises `fts5: syntax error near "-"` and
    anything containing `:` is read as a column filter. Quoting every term
    sidesteps the parser entirely and doubles as the injection guard.

    Punctuation-joined runs are kept together as a phrase, so `nas-tunnel`
    becomes the phrase "nas tunnel" and still matches the hyphenated original
    (unicode61 splits the document the same way) without also matching every
    note that merely says "tunnel".

    Returns "" when the text has no indexable token, meaning: skip the
    lexical leg entirely.
    """
    terms: list[str] = []
    for word in text.split():
        parts = _FTS_TOKEN_RE.findall(word)
        if not parts:
            continue
        # A phrase for multi-run words ("nas-tunnel" -> "nas tunnel"),
        # a plain quoted term otherwise. Quotes are always legal in FTS5.
        terms.append('"' + " ".join(parts) + '"')
        if len(terms) >= MAX_FTS_TERMS:
            break
    # AND, not OR — this is what makes the lexical leg safe to fuse.
    #
    # Conjunctive matching turns the leg self-gating. A literal lookup
    # ("PostGIS", "Muck Rack") is one or two rare terms that genuinely
    # co-occur, so AND fires and is extremely precise. A discursive query
    # ("what are my tripwires for leaving") has no chunk containing all of its
    # terms, so the leg returns nothing and search degrades to pure dense —
    # exactly the regime where dense was already right.
    #
    # Measured on this vault (14 rare literals / 10 natural-language queries):
    # OR gives literal hit@1 13/14 but wrecks NL, agreeing with the dense-only
    # top hit on just 1/10 — an OR over six common words drags in every chunk
    # that merely says "leaving" or "retainer", and with the two candidate
    # lists barely overlapping, RRF ties hand rank 1 to that noise. AND scores
    # 14/14 on literals and 8/10 NL agreement. An AND-then-OR fallback looks
    # appealing and is the worst of both (14/14, 1/10): the fallback fires on
    # precisely the discursive queries that OR handles badly.
    return " AND ".join(terms)


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict | None = None
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.executescript(FTS_SCHEMA)
        self._migrate_fts()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ---- FTS maintenance -------------------------------------------------

    @staticmethod
    def _fts_indexed_count(conn: sqlite3.Connection) -> int:
        """Number of docs actually present in the FTS index.

        NOT `SELECT count(*) FROM chunks_fts` — for an external-content table a
        full scan is answered from the *content* table, so that count equals
        count(*) FROM chunks even when the index is empty or drifted. It is a
        false green. `chunks_fts_docsize` is the shadow table holding one row
        per genuinely indexed doc, so it is the only honest count.
        """
        return conn.execute("SELECT count(*) FROM chunks_fts_docsize").fetchone()[0]

    def _migrate_fts(self) -> None:
        """Backfill or rebuild the FTS index when it is missing or drifted.

        Covers three cases with one check: a pre-existing DB indexed before FTS
        existed (docsize 0, chunks many), a tokenizer/schema change (version
        mismatch), and any drift that slipped past the triggers. Steady state is
        a single cheap COUNT and no write — so this is safe to run on every open.
        Never touches embeddings, so it can never trigger a re-embed.
        """
        with closing(self._connect()) as conn, conn:
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            indexed = self._fts_indexed_count(conn)
            row = conn.execute("SELECT value FROM meta WHERE key='fts_version'").fetchone()
            version = row[0] if row else None
            if indexed == chunks and version == FTS_VERSION:
                return
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('fts_version', ?)", (FTS_VERSION,)
            )

    def fts_health(self) -> dict:
        """FTS index health, for vault_stats. Runs FTS5's own integrity check."""
        with closing(self._connect()) as conn:
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
            indexed = self._fts_indexed_count(conn)
            try:
                conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
                ok = True
            except sqlite3.DatabaseError:
                ok = False
        return {"indexed": indexed, "in_sync": indexed == chunks, "integrity_ok": ok}

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
                "SELECT id, file_path, heading_path, text, tags, embedding FROM chunks ORDER BY id"
            ).fetchall()
        if not rows:
            return {"ids": [], "pos": {}, "paths": [], "headings": [], "texts": [], "tags": [],
                    "matrix": np.zeros((0, 1), dtype=np.float32)}
        matrix = np.frombuffer(b"".join(r[5] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        return {
            "ids": [r[0] for r in rows],
            # chunk id -> row position in the matrix; the join key for the FTS
            # leg, which ranks by rowid (= chunks.id), not by matrix position.
            "pos": {r[0]: i for i, r in enumerate(rows)},
            "paths": [r[1] for r in rows],
            "headings": [r[2] for r in rows],
            "texts": [r[3] for r in rows],
            "tags": [r[4] for r in rows],
            "matrix": matrix,
        }

    @staticmethod
    def _folder_prefix(folder: str) -> str:
        return folder.strip("/") + "/"

    def _dense_leg(self, qvec: np.ndarray, k: int, folder: str | None) -> tuple[list[int], dict[int, float]]:
        """Top-k matrix positions by cosine, folder applied BEFORE ranking.

        The candidate set is restricted first and only those rows are scored,
        rather than scoring everything and masking to -inf afterwards. With a
        single leg the two were equivalent; with fusion they are not — ranking
        the full corpus and masking later leaves holes that shift every
        surviving item's rank position, which is exactly the input RRF consumes.
        """
        cache = self._get_cache()
        n = cache["matrix"].shape[0]
        if n == 0:
            return [], {}
        if folder:
            prefix = self._folder_prefix(folder)
            cand = np.fromiter(
                (i for i, p in enumerate(cache["paths"]) if p.startswith(prefix)), dtype=np.int64
            )
            if cand.size == 0:
                return [], {}
            scores = cache["matrix"][cand] @ qvec.astype(np.float32)
        else:
            cand = None
            scores = cache["matrix"] @ qvec.astype(np.float32)
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        positions = [int(cand[i]) if cand is not None else int(i) for i in top]
        return positions, {p: float(scores[i]) for p, i in zip(positions, top)}

    def _lexical_leg(self, query_text: str, k: int, folder: str | None) -> list[int]:
        """Top-k matrix positions by BM25 over chunk text + tags.

        The folder constraint is part of the candidate query, not a post-filter,
        for the same rank-integrity reason as the dense leg.

        Column weights (text 1.0, tags 0.5): an exact tag match is real signal,
        but the tags column is a handful of words, and BM25's length
        normalization already boosts short fields hard — left at parity, a
        single matching tag outranks a whole chunk about the subject.
        """
        match = build_fts_query(query_text)
        if not match:
            return []
        cache = self._get_cache()
        sql = (
            "SELECT chunks_fts.rowid FROM chunks_fts "
            "JOIN chunks ON chunks.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?"
        )
        params: list = [match]
        if folder:
            prefix = self._folder_prefix(folder)
            # substr(...) = ?, not LIKE: LIKE is ASCII-case-insensitive and
            # would silently diverge from the dense leg's str.startswith, and
            # it would also need % and _ escaped out of the user's folder.
            sql += " AND substr(chunks.file_path, 1, ?) = ?"
            params += [len(prefix), prefix]
        sql += " ORDER BY bm25(chunks_fts, 1.0, 0.5) LIMIT ?"
        params.append(k)
        try:
            with closing(self._connect()) as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            # A malformed match expression must never take down search; the
            # dense leg alone is a correct, if weaker, answer.
            return []
        # The cache and this query are read at different instants, so a reindex
        # in between can leave `pos` stale. That is safe here only because
        # chunks.id is INTEGER PRIMARY KEY AUTOINCREMENT, which SQLite
        # guarantees never to reuse: a stale map can miss a brand-new row, but
        # can never map an id onto a different chunk than the one BM25 ranked.
        # Drop unknown ids rather than raising.
        pos = cache["pos"]
        return [pos[r[0]] for r in rows if r[0] in pos]

    def search(
        self,
        qvec: np.ndarray,
        k: int = 8,
        folder: str | None = None,
        query_text: str | None = None,
    ) -> list[dict]:
        """Hybrid search: dense + BM25 candidates fused by RRF.

        Falls back to pure dense when `query_text` is omitted or contains no
        indexable token, so the dense-only contract still holds for callers
        that have a vector but no query string.
        """
        cache = self._get_cache()
        if cache["matrix"].shape[0] == 0:
            return []

        dense_pos, cosine = self._dense_leg(qvec, CANDIDATE_K, folder)
        lex_pos = self._lexical_leg(query_text, CANDIDATE_K, folder) if query_text else []

        if not lex_pos:
            ranked = dense_pos[:k]
            legs = {p: "dense" for p in ranked}
            fused = {p: cosine.get(p, 0.0) for p in ranked}
        else:
            fused: dict[int, float] = {}
            for positions in (dense_pos, lex_pos):
                for rank, p in enumerate(positions, start=1):
                    fused[p] = fused.get(p, 0.0) + 1.0 / (RRF_K + rank)
            dense_set, lex_set = set(dense_pos), set(lex_pos)
            # Exact ties are common and consequential here. When the two legs
            # disagree completely — no overlap at all, which is precisely what
            # happens on a rare literal token — each leg's rank-1 scores
            # 1/(RRF_K+1) and RRF has no evidence left to separate them. Left to
            # an arbitrary tie-break the correct lexical hit loses a coin flip
            # about half the time. Preferring the lexical leg *only on an exact
            # tie* is the minimal principled resolution: a dense hit that any
            # other leg corroborates still outranks it, nothing else reorders,
            # and the degenerate case resolves toward the leg that can abstain
            # (dense always returns k neighbours whether or not it knows the
            # term; a BM25 miss returns nothing).
            ranked = sorted(fused, key=lambda p: (-fused[p], p not in lex_set, p))[:k]
            legs = {
                p: "hybrid" if p in dense_set and p in lex_set else ("dense" if p in dense_set else "lexical")
                for p in ranked
            }

        out = []
        for p in ranked:
            hit = {
                "path": cache["paths"][p],
                "heading_path": cache["headings"][p],
                "text": cache["texts"][p],
                "tags": cache["tags"][p],
                "score": fused[p],
                "retrieval": legs[p],
            }
            if p in cosine:
                hit["cosine"] = cosine[p]
            out.append(hit)
        return out

    def related_notes(self, qvec: np.ndarray, exclude_path: str, k: int = 8) -> list[dict]:
        """Nearest notes (aggregated per note, best-chunk score), excluding the source note.

        Deliberately stays pure dense — no lexical leg, no fusion. This is
        note-to-note similarity driven by a mean embedding; there is no query
        string to hand BM25. The available substitute would be to synthesize a
        pseudo-query from the source note's own terms, which mostly retrieves
        notes sharing its boilerplate (same client, same template, same tag
        header) rather than its meaning — the exact failure the dense leg is
        good at avoiding. A "Connections pane" wants conceptual neighbours, so
        the dense-only answer is the right one, not merely the easy one.
        """
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
        if Path(q).suffix not in (".md", ".html", ".htm"):
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
        return {"files": files, "chunks": chunks, "db_bytes": size, "fts": self.fts_health()}
