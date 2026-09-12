"""vault-mcp — semantic search MCP server for a local markdown vault."""

import os
from pathlib import Path

VAULT_PATH = Path(os.environ.get("VAULT_MCP_VAULT", "~/Documents/CX")).expanduser()
# Extra folders indexed alongside the vault, each under its own name: "Name=/path"
# pairs joined by os.pathsep. They contribute HTML pages only, listed the way
# Onyx's Artifacts lists them. A folder that doesn't exist is skipped, so the
# author's default is inert elsewhere; set it to "" to index the vault alone.
MOUNTS_SPEC = os.environ.get("VAULT_MCP_MOUNTS", "Artifacts=~/Documents/Artifacts")
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("VAULT_MCP_DB", REPO_ROOT / "data" / "index.db")).expanduser()
OLLAMA_URL = os.environ.get("VAULT_MCP_OLLAMA", "http://localhost:11434")
EMBED_MODEL = os.environ.get("VAULT_MCP_MODEL", "nomic-embed-text")
