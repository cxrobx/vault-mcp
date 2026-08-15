#!/usr/bin/env python
"""Retrieval eval: does hybrid search actually beat dense-only, and does it
regress natural-language queries?

Run before and after any change to ranking, chunking, or the embedding model.
The lexical leg is easy to add in a way that looks fine on keyword probes while
quietly wrecking the semantic queries that were already working — this measures
both sides so that trade is visible instead of assumed.

    .venv/bin/python scripts/eval_retrieval.py
    .venv/bin/python scripts/eval_retrieval.py --literals PostGIS Kokoro

Two measurements:

* **Literal recall** — for a rare token, is a note that actually contains it
  ranked first? Ground truth is a substring scan of the chunk table, so this
  needs no hand-labelling. Tokens are auto-discovered from the index (rare,
  identifier-shaped) unless you pass --literals, so it works on any vault.
* **NL agreement** — how often hybrid's top hit matches the dense-only top hit.
  Dense is the incumbent and is good at these, so divergence is the risk signal,
  not the goal. Expect high-but-not-perfect; inspect any query that moves.
"""

import argparse
import collections
import hashlib
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vault_mcp import DB_PATH  # noqa: E402
from vault_mcp.embeddings import EmbeddingsUnavailable, OllamaEmbedder  # noqa: E402
from vault_mcp.store import Store  # noqa: E402

# Deliberately generic: these are phrasings, not vault-specific facts, so the
# agreement number means the same thing on someone else's notes.
NL_QUERIES = [
    "pricing strategy for service deals",
    "how we decided on the deployment setup",
    "career direction and values",
    "client onboarding process",
    "what did I learn about hiring",
    "how should I price a retainer",
    "notes about staying balanced and avoiding burnout",
    "how do I run a discovery call",
]

_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_-]{4,}")


def discover_literals(conn: sqlite3.Connection, n: int) -> list[str]:
    """Pick identifier-shaped tokens that appear in only a few notes.

    Rare + capitalized mid-word (PostGIS, FieldPulse) is a cheap proxy for
    "term a human would search for literally", and low document frequency is
    exactly the regime where a dense-only index blurs a term into its
    neighbourhood.
    """
    df: dict[str, set] = collections.defaultdict(set)
    for path, text in conn.execute("SELECT file_path, text FROM chunks"):
        for tok in set(_IDENT.findall(text)):
            df[tok].add(path)
    rare = [
        t for t, paths in df.items()
        if 1 <= len(paths) <= 3
        and t.isalnum()
        # Mixed case, with a capital after the first character: PostGIS,
        # FieldPulse. Requiring a lowercase letter too is what rejects ALL-CAPS
        # prose ("ACCESS", "ACTUAL") — shouting, not identifiers, and often in
        # hundreds of notes despite the rare-looking casing.
        and any(c.isupper() for c in t[1:])
        and any(c.islower() for c in t)
    ]
    # Spread the sample across the corpus instead of taking it alphabetically,
    # which otherwise returns fourteen tokens all starting with "A". Hashed
    # rather than random so successive runs stay comparable.
    return sorted(rare, key=lambda t: hashlib.md5(t.encode()).hexdigest())[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--literals", nargs="*", help="tokens to probe (default: auto-discover)")
    ap.add_argument("--count", type=int, default=14, help="how many to auto-discover")
    ap.add_argument("-k", type=int, default=5, help="cutoff for hit@k")
    args = ap.parse_args()

    store = Store(DB_PATH)
    embedder = OllamaEmbedder()
    conn = sqlite3.connect(str(DB_PATH))

    health = store.fts_health()
    print(f"FTS: {health}")
    if not health["in_sync"] or not health["integrity_ok"]:
        print("  WARNING: FTS index is drifted — lexical results will be incomplete.")

    literals = args.literals or discover_literals(conn, args.count)

    print(f"\n{'='*74}\nLITERAL RECALL  (rare tokens; ground truth = substring scan)\n{'='*74}")
    rows, d1 = [], 0
    h1 = h5 = 0
    for tok in literals:
        truth = {
            r[0] for r in
            conn.execute("SELECT DISTINCT file_path FROM chunks WHERE text LIKE ?", (f"%{tok}%",))
        }
        if not truth:
            print(f"  {tok:22s} not in corpus — skipped")
            continue
        qvec = embedder.embed_query(tok)
        dense = [h["path"] for h in store.search(qvec, k=args.k)]
        hyb = store.search(qvec, k=args.k, query_text=tok)
        paths = [h["path"] for h in hyb]
        d1 += bool(dense) and dense[0] in truth
        a1 = bool(paths) and paths[0] in truth
        h1 += a1
        h5 += any(p in truth for p in paths)
        rows.append(tok)
        leg = hyb[0].get("retrieval", "-") if hyb else "-"
        print(f"  {tok:22s} truth={len(truth):2d}f  hit@1={'Y' if a1 else 'n'}  [{leg:7s}]  {paths[0][:44] if paths else '-'}")
    n = len(rows)
    if n:
        print(f"\n  dense-only  hit@1 {d1}/{n}")
        print(f"  hybrid      hit@1 {h1}/{n}   hit@{args.k} {h5}/{n}")

    print(f"\n{'='*74}\nNL AGREEMENT  (hybrid vs dense-only top hit; divergence = risk)\n{'='*74}")
    agree = 0
    for q in NL_QUERIES:
        qvec = embedder.embed_query(q)
        dense = [h["path"] for h in store.search(qvec, k=args.k)]
        hyb = [h["path"] for h in store.search(qvec, k=args.k, query_text=q)]
        if not dense:
            continue
        same = dense[0] == hyb[0]
        agree += same
        print(f"  overlap@{args.k}={len(set(dense)&set(hyb))}/{args.k}  {'same' if same else 'MOVED'}  {q!r}")
        if not same:
            print(f"      dense : {dense[0]}")
            print(f"      hybrid: {hyb[0]}")
    print(f"\n  top-1 unchanged {agree}/{len(NL_QUERIES)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EmbeddingsUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
