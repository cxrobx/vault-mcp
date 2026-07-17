"""Entry points: `python -m vault_mcp` (MCP server over stdio),
`python -m vault_mcp index [--full]` (CLI indexer)."""

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


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "index":
        _cli_index(full="--full" in argv[1:])
    else:
        from .server import run

        run()


if __name__ == "__main__":
    main()
