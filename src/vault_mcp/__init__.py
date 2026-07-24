"""vault-mcp — semantic search MCP server for a local markdown vault."""

import os
from pathlib import Path

VAULT_PATH = Path(os.environ.get("VAULT_MCP_VAULT", "~/Documents/CX")).expanduser()
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("VAULT_MCP_DB", REPO_ROOT / "data" / "index.db")).expanduser()
OLLAMA_URL = os.environ.get("VAULT_MCP_OLLAMA", "http://localhost:11434")
EMBED_MODEL = os.environ.get("VAULT_MCP_MODEL", "nomic-embed-text")
