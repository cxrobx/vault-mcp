#!/usr/bin/env python3
"""Copy a finished index from another machine, so this one never embeds documents.

For a machine with `indexing = false` in its config. The source runs the normal
indexer; this asks it for a consistent snapshot, checks that, and loads it here.
A source that is asleep or off the network is not an error: the run exits and
the next one tries again, so the copy catches up on its own once the source is
back. LaunchAgent: scripts/com.cx.vault-index-pull.plist.

    python scripts/pull_index.py                # from VAULT_INDEX_SOURCE, default "laptop"
    python scripts/pull_index.py --force        # copy even if the chunk count falls by half

Paths in the index are relative to the vault and mounts, so a copy is valid on
any machine whose folders have the same names. The snapshot goes in through
SQLite's backup API, not a file swap: the live database is in WAL mode, and a
replaced file beside an old -wal would be corrupt. Search servers re-check the
index on every query, so they pick the copy up without a restart.
"""

import argparse
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vault_mcp import DB_PATH, EMBED_MODEL  # noqa: E402
from vault_mcp.indexer import _index_flock  # noqa: E402

SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
REMOTE_DB = "Projects/vault-mcp/data/index.db"  # relative to the source's home
SIGNATURE = "SELECT max(id), count(*) FROM chunks"  # what Store uses to spot a change
MIN_KEEP = 0.5


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def remote(host: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH, host, command], capture_output=True, text=True, timeout=600)


def read_one(db: Path, sql: str):
    with closing(sqlite3.connect(db, timeout=30)) as conn:
        return conn.execute(sql).fetchone()


def check(snapshot: Path, local_chunks: int, force: bool) -> str | None:
    """Why this snapshot must not be loaded, or None."""
    if read_one(snapshot, "PRAGMA quick_check")[0] != "ok":
        return "failed quick_check"
    model = read_one(snapshot, "SELECT value FROM meta WHERE key = 'model'")
    if not model or model[0] != EMBED_MODEL:
        return f"built with {model and model[0]}, but queries here use {EMBED_MODEL}"
    chunks = read_one(snapshot, "SELECT count(*) FROM chunks")[0]
    if not force and chunks < local_chunks * MIN_KEEP:
        return f"{chunks} chunks against {local_chunks} here; rerun with --force if that is right"
    return None


def load(snapshot: Path, db: Path) -> None:
    with closing(sqlite3.connect(snapshot)) as src, closing(sqlite3.connect(db, timeout=60)) as dst:
        src.backup(dst)


def pull(host: str, db: Path, force: bool) -> int:
    db_q = shlex.quote(REMOTE_DB)
    probe = remote(host, f"sqlite3 ~/{db_q} {shlex.quote(SIGNATURE)}")
    if probe.returncode != 0:
        log(f"{host} not reachable; will try again next run")
        return 0
    with closing(sqlite3.connect(db, timeout=30)) as conn:
        here = conn.execute(SIGNATURE).fetchone()
    there = probe.stdout.strip()
    if there == f"{here[0]}|{here[1]}":
        log("already current")
        return 0

    t0 = time.monotonic()
    incoming = db.with_suffix(".incoming")
    snap = f"{REMOTE_DB}.snapshot"
    try:
        made = remote(host, f"sqlite3 ~/{db_q} {shlex.quote(f'.backup {snap}')}")
        if made.returncode != 0:
            log(f"snapshot failed on {host}: {made.stderr.strip()[:200]}")
            return 1
        copied = subprocess.run(["rsync", "-a", "--timeout=120", "-e", " ".join(SSH),
                                 f"{host}:{snap}", str(incoming)], capture_output=True, text=True)
        if copied.returncode != 0:
            log(f"rsync failed: {copied.stderr.strip()[:200]}")
            return 1
        reason = check(incoming, here[1], force)
        if reason:
            log(f"refused the snapshot: {reason}")
            return 1
        with _index_flock(db) as acquired:
            if not acquired:
                log("another process holds the index lock; will try again next run")
                return 0
            load(incoming, db)
        log(f"loaded {there} over {here[0]}|{here[1]} in {time.monotonic() - t0:.1f}s")
        return 0
    finally:
        remote(host, f"rm -f ~/{shlex.quote(snap)}")
        for suffix in ("", "-wal", "-shm"):
            Path(f"{incoming}{suffix}").unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None, help='ssh host (default $VAULT_INDEX_SOURCE or "laptop")')
    ap.add_argument("--force", action="store_true", help="accept a much smaller index")
    a = ap.parse_args()
    sys.exit(pull(a.source or os.environ.get("VAULT_INDEX_SOURCE", "laptop"), DB_PATH, a.force))
