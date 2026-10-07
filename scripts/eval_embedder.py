#!/usr/bin/env python
"""Embedder bake-off: would a different embedding model find the right note sooner?

Compares the live model (whatever config.toml names; nomic-embed-text until
2026-10-06, EmbeddingGemma 2 since) with a candidate (EmbeddingGemma 2 by default) on known-item search — a query written
for one specific note, scored by where that note lands. Both run through the
real `Store.search`, hybrid and dense-only, on COPIES of the index; the live
index is never written.

    # 1. build the query set once (claude -p haiku, on the subscription; cached)
    uv run --with numpy,pyyaml,httpx,mcp python scripts/eval_embedder.py queries
    # 2. embed the corpus with the candidate and score both
    uv run --with 'sentence-transformers>=6.1' --with numpy,pyyaml,httpx,mcp \\
        python scripts/eval_embedder.py run

Queries come from three places, reported separately because they are not
equally trustworthy:

* **picks** — real launcher picks from launcher.jsonl. Ground truth from
  Chris's own hand, but only a few.
* **terse** — what he'd type into `j`: 2-5 words, may lean on name words.
* **describe** — a sentence about the content that avoids the title words.
  The case a lexical leg can't rescue, so it isolates the embedder.

The synthetic sets are written by a model that saw the note, so they flatter
any embedder a little; that bias is the same for both sides, so the delta is
what to read, not the absolute numbers.
"""

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vault_mcp import DB_PATH, EMBED_MODEL  # noqa: E402
from vault_mcp.embeddings import OllamaEmbedder  # noqa: E402
from vault_mcp.store import Store  # noqa: E402

QUERIES = ROOT / "data" / "eval_embedder_queries.jsonl"
WORK = ROOT / "data" / "eval_embedder"
LAUNCH_LOG = Path("~/.local/state/vault-mcp/launcher.jsonl").expanduser()
SHOWN = 8  # rows the launcher shows
BATCH = 15

PROMPT = """You are building a search test set. For each document below, write two queries a person would use to find THAT document again in their personal notes:
- "terse": 2-5 lowercase words, like typing into a launcher. May use one distinctive name word, but must not be the exact title.
- "describe": one natural sentence (8-16 words) about what the document says or is for. Do NOT use any word from its title.
Return ONLY a JSON array: [{{"id": <id>, "terse": "...", "describe": "..."}}, ...]

{docs}"""


def sample_files(conn: sqlite3.Connection, n: int) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        "SELECT f.path, f.title, group_concat(c.text, '\n') FROM files f "
        "JOIN chunks c ON c.file_path = f.path GROUP BY f.path"
    ).fetchall()
    rows = [r for r in rows if len(r[2]) >= 400]
    # Hashed, not random, so a rerun samples the same notes.
    rows.sort(key=lambda r: hashlib.md5(r[0].encode()).hexdigest())
    return [(p, t or Path(p).stem, body[:1500]) for p, t, body in rows[:n]]


def cmd_queries(args) -> None:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    files = sample_files(conn, args.n)
    out = []
    for i in range(0, len(files), BATCH):
        batch = files[i : i + BATCH]
        docs = "\n\n".join(
            f"### id {j}\ntitle: {t}\n{body}" for j, (_, t, body) in enumerate(batch)
        )
        res = subprocess.run(
            ["claude", "-p", "--model", "haiku", PROMPT.format(docs=docs)],
            capture_output=True, text=True, timeout=300,
        )
        txt = res.stdout
        items = json.loads(txt[txt.index("[") : txt.rindex("]") + 1])
        for it in items:
            path = batch[int(it["id"])][0]
            for kind in ("terse", "describe"):
                out.append({"kind": kind, "query": it[kind].strip(), "target": path})
        print(f"  {min(i + BATCH, len(files))}/{len(files)} notes", flush=True)

    known = {r[0] for r in conn.execute("SELECT path FROM files")}
    picks = 0
    if LAUNCH_LOG.exists():
        for line in LAUNCH_LOG.read_text().splitlines():
            ev = json.loads(line)
            if ev.get("event") != "pick":
                continue
            rel = next((s for s in ev.get("shown", []) if ev["path"].endswith("/" + s.split("/", 1)[-1])), None)
            if rel in known:
                out.append({"kind": "picks", "query": ev["query"], "target": rel})
                picks += 1
    QUERIES.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {len(out)} queries ({picks} real picks) -> {QUERIES}")


def candidate_db(model_id: str, dim: int) -> Path:
    """Copy of the index with every chunk re-embedded by the candidate (cached)."""
    WORK.mkdir(parents=True, exist_ok=True)
    tag = model_id.split("/")[-1]
    vec_path = WORK / f"{tag}-768.npy"
    src = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = src.execute(
        "SELECT c.id, c.heading_path, c.text, coalesce(nullif(f.title,''), c.file_path) "
        "FROM chunks c JOIN files f ON f.path = c.file_path ORDER BY c.id"
    ).fetchall()
    if not vec_path.exists():
        model = load_candidate(model_id)
        # Same content the live indexer embeds, in the candidate's titled format.
        docs = [
            f"title: {t} > {hp} | text: {txt}" if hp else f"title: {t} | text: {txt}"
            for _, hp, txt, t in rows
        ]
        t0 = time.time()
        vecs = model.encode(docs, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        np.save(vec_path, vecs.astype(np.float32))
        print(f"  embedded {len(docs)} chunks in {time.time() - t0:.0f}s")
    vecs = np.load(vec_path)[:, :dim]
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    db = WORK / f"{tag}-{dim}.db"
    if db.exists():
        db.unlink()
    shutil.copy(DB_PATH, db)
    conn = sqlite3.connect(db)
    conn.executemany(
        "UPDATE chunks SET embedding = ? WHERE id = ?",
        [(vecs[i].astype(np.float32).tobytes(), r[0]) for i, r in enumerate(rows)],
    )
    conn.commit()
    conn.close()
    return db


_MODEL = None


def load_candidate(model_id: str):
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _MODEL = SentenceTransformer(
            model_id, device=device,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": torch.float32},
        )
    return _MODEL


def ranks(store: Store, embed, queries: list[dict], hybrid: bool) -> list[int]:
    """File-level rank of each target (1-based; 0 = not in the top 50 files)."""
    out = []
    for q in queries:
        hits = store.search(embed(q["query"]), k=200, candidates=200,
                            query_text=q["query"] if hybrid else None)
        files = list(dict.fromkeys(h["path"] for h in hits))[:50]
        out.append(files.index(q["target"]) + 1 if q["target"] in files else 0)
    return out


def summarize(r: list[int]) -> str:
    a = np.array(r)
    mrr = np.where(a > 0, 1.0 / np.maximum(a, 1), 0).mean()
    return f"MRR {mrr:.3f}  @1 {(a == 1).mean():5.1%}  @{SHOWN} {((a > 0) & (a <= SHOWN)).mean():5.1%}"


def cmd_run(args) -> None:
    queries = [json.loads(l) for l in QUERIES.read_text().splitlines() if l.strip()]
    live = Store(DB_PATH)  # read-only use: search never writes
    # The live index, queried the way vault-mcp queries it (model from config).
    incumbent = OllamaEmbedder()
    model = load_candidate(args.model)

    def gemma(dim):
        return lambda q: model.encode(f"task: search result | query: {q}", truncate_dim=dim,
                                      normalize_embeddings=True)

    systems = [(f"{EMBED_MODEL} (live)", live, incumbent.embed_query)]
    for dim in args.dims:
        systems.append((f"{args.model.split('/')[-1]} @{dim}", Store(candidate_db(args.model, dim)), gemma(dim)))

    kinds = ["picks", "terse", "describe"]
    report = {}
    for mode in ("hybrid", "dense"):
        print(f"\n=== {mode.upper()} {'(what search_vault / j use)' if mode == 'hybrid' else '(embedder alone)'} ===")
        for kind in kinds:
            qs = [q for q in queries if q["kind"] == kind]
            if not qs:
                continue
            print(f"  {kind} (n={len(qs)})")
            for name, store, emb in systems:
                r = ranks(store, emb, qs, mode == "hybrid")
                report[(mode, kind, name)] = r
                print(f"    {name:28s} {summarize(r)}")

    # Which notes moved, so the number can be checked by eye.
    base, cand = systems[0][0], systems[1][0]
    qs = [q for q in queries if q["kind"] == "describe"]
    a, b = report[("hybrid", "describe", base)], report[("hybrid", "describe", cand)]
    moved = sorted(zip(qs, a, b), key=lambda x: (x[2] or 99) - (x[1] or 99))
    print(f"\nBiggest moves, hybrid/describe ({base} -> {cand}; 0 = outside top 50):")
    for q, x, y in moved[:5] + moved[-5:]:
        print(f"  {x:>2} -> {y:>2}  {q['query'][:60]!r}  [{q['target'][:50]}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queries")
    q.add_argument("-n", type=int, default=90, help="notes to sample")
    r = sub.add_parser("run")
    r.add_argument("--model", default="google/embeddinggemma-2")
    r.add_argument("--dims", type=int, nargs="+", default=[768, 256])
    args = ap.parse_args()
    {"queries": cmd_queries, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
