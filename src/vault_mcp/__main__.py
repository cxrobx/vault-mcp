"""Entry points: `python -m vault_mcp` (MCP server over stdio),
`python -m vault_mcp index [--full]` (CLI indexer),
`python -m vault_mcp search "<query>" [--k N] [--folder PREFIX]` (CLI search, JSON out).

The CLI search lets a skill query a second index (VAULT_MCP_CONFIG pointing at
another config with its own vault + db) without registering another MCP server."""

import sys


def _cli_index(full: bool) -> None:
    from . import DB_PATH, VAULT_PATH
    from .embeddings import OllamaEmbedder
    from .indexer import reindex
    from .store import Store

    store = Store(DB_PATH)
    embedder = OllamaEmbedder()
    print(f"Indexing {VAULT_PATH} -> {DB_PATH} ({'full' if full else 'delta'})")

    def progress(n: int, rel_path: str, n_chunks: int) -> None:
        print(f"  [{n}] {rel_path} ({n_chunks} chunks)")

    result = reindex(store, embedder, full=full, progress=progress)
    for key, value in result.items():
        print(f"{key}: {value}")


def _cli_search(argv: list[str]) -> None:
    import argparse
    import json

    from .server import search_vault

    ap = argparse.ArgumentParser(prog="vault_mcp search")
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--folder", default=None)
    a = ap.parse_args(argv)
    print(json.dumps(search_vault(a.query, k=a.k, folder=a.folder), indent=1, ensure_ascii=False))


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "index":
        _cli_index(full="--full" in argv[1:])
    elif argv and argv[0] == "search":
        _cli_search(argv[1:])
    else:
        from .server import run

        run()


if __name__ == "__main__":
    main()
