"""vault-mcp — semantic search MCP server for a local markdown vault."""

import os
import tomllib
from pathlib import Path

# One config file shared by every entry point (the MCP server each agent
# session starts, the CLI indexer, the launcher). It matters because they all
# write the same index: a delta reindex DELETES whatever its own walk does not
# reach, so two entry points that disagree about the mounts would take turns
# removing and re-embedding each other's files. An env var still wins over the
# file, for one-off runs against a scratch vault.
CONFIG_PATH = Path(os.environ.get("VAULT_MCP_CONFIG", "~/.config/vault-mcp/config.toml")).expanduser()


def _load_config(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


_config = _load_config(CONFIG_PATH)


def _setting(env: str, key: str, default: str) -> str:
    if env in os.environ:
        return os.environ[env]
    value = _config.get(key, default)
    if isinstance(value, list):
        return os.pathsep.join(str(v) for v in value)
    return str(value)


VAULT_PATH = Path(_setting("VAULT_MCP_VAULT", "vault", "~/Documents/CX")).expanduser()
# Extra folders indexed alongside the vault, each under its own name: "Name=/path"
# pairs joined by os.pathsep (or a TOML list of "Name=/path" strings). They
# contribute HTML pages only, listed the way Onyx's Artifacts lists them. A
# folder that doesn't exist is skipped, so the author's default is inert
# elsewhere; set it to "" to index the vault alone.
MOUNTS_SPEC = _setting("VAULT_MCP_MOUNTS", "mounts", "Artifacts=~/Documents/Artifacts")
# Same "Name=/path" shape, but walked the way the vault is: markdown notes and
# HTML notes at any depth. For folders of documents that live outside the vault
# (a repo's docs/, a client folder).
NOTE_MOUNTS_SPEC = _setting("VAULT_MCP_NOTE_MOUNTS", "note_mounts", "")
# Extra plain-text suffixes a note mount indexes beside .md, e.g. ".typ".
TEXT_SUFFIXES = tuple(
    s if s.startswith(".") else f".{s}"
    for s in (
        raw.strip().lower()
        for raw in _setting("VAULT_MCP_TEXT_SUFFIXES", "text_suffixes", "").replace(",", os.pathsep).split(os.pathsep)
    )
    if s
)
# Index paths to leave out, as fnmatch globs against the index-relative path
# ("Proposals/*/src/theme.typ"). NOTE fnmatch's "*" spans "/", so one star is
# enough to cross folders and "Name/*" excludes a whole subtree.
EXCLUDE_GLOBS = tuple(
    g for g in (raw.strip() for raw in _setting("VAULT_MCP_EXCLUDE", "exclude", "").split(os.pathsep)) if g
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(_setting("VAULT_MCP_DB", "db", str(REPO_ROOT / "data" / "index.db"))).expanduser()
OLLAMA_URL = _setting("VAULT_MCP_OLLAMA", "ollama", "http://localhost:11434")
EMBED_MODEL = _setting("VAULT_MCP_MODEL", "model", "nomic-embed-text")
