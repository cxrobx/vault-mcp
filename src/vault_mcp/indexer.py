"""Vault walker, markdown chunker, and delta-reindex logic.

Walks the vault following symlinks (symlinked folders index like real ones)
with a realpath cycle guard. Chunks split on
H1–H3 headings outside code fences; each chunk is embedded as
"{note title} > {heading path}\\n{body}". Delta reindex: mtime+size stat
sweep, content-hash confirm, per-file transaction — a crash never leaves a
file half-indexed.
"""

import hashlib
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import VAULT_PATH
from .embeddings import EMBED_DIM, OllamaEmbedder
from .store import Store

EXCLUDE_DIR_NAMES = {".obsidian", ".smart-env", ".trash", ".git", ".SynologyWorkingDirectory"}
EXCLUDE_REL_PATHS = {"Other/Templates"}
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 80
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")

_reindex_lock = threading.Lock()


def walk_vault(vault: Path = VAULT_PATH) -> dict[str, Path]:
    """Map vault-relative path -> absolute path for every in-scope .md file."""
    files: dict[str, Path] = {}
    visited: set[Path] = set()

    def _walk(dirpath: Path, rel: str) -> None:
        try:
            real = dirpath.resolve()
        except OSError:
            return
        if real in visited:
            return
        visited.add(real)
        try:
            entries = sorted(dirpath.iterdir(), key=lambda p: p.name)
        except OSError:
            return
        for entry in entries:
            rel_child = f"{rel}/{entry.name}" if rel else entry.name
            if entry.is_dir():
                if entry.name.startswith(".") or entry.name in EXCLUDE_DIR_NAMES or rel_child in EXCLUDE_REL_PATHS:
                    continue
                _walk(entry, rel_child)
            elif entry.is_file() and entry.name.endswith(".md"):
                files[rel_child] = entry

    _walk(vault, "")
    return files


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            try:
                fm = yaml.safe_load("\n".join(lines[1:i])) or {}
            except yaml.YAMLError:
                fm = {}
            if not isinstance(fm, dict):
                fm = {}
            return fm, "\n".join(lines[i + 1 :])
    return {}, text


def extract_tags(frontmatter: dict) -> str:
    raw = frontmatter.get("tags") or frontmatter.get("tag") or []
    if isinstance(raw, str):
        raw = re.split(r"[,\s]+", raw)
    if not isinstance(raw, list):
        return ""
    tags = [str(t).strip().lstrip("#") for t in raw if str(t).strip().lstrip("#")]
    return ", ".join(tags)


def _split_oversized(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
        while len(para) > limit:
            pieces.append(para[:limit])
            para = para[limit:]
        current = para
    if current:
        pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def chunk_note(body: str) -> list[tuple[str, str]]:
    """Split a note body into (heading_path, text) chunks on H1–H3 boundaries."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((" > ".join(t for _, t in stack), text))
        current_lines = []

    for line in body.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current_lines.append(line)
            continue
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            current_lines.append(line)
    flush()

    chunks: list[tuple[str, str]] = []
    for heading_path, text in sections:
        for piece in _split_oversized(text):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append((heading_path, piece))
    return chunks


def reindex(
    store: Store,
    embedder: OllamaEmbedder,
    full: bool = False,
    progress=None,
) -> dict:
    """Delta (or full) reindex of the vault. Serialized by a module lock."""
    with _reindex_lock:
        t0 = time.monotonic()
        if not VAULT_PATH.is_dir():
            raise RuntimeError(f"Vault not found at {VAULT_PATH}")

        # A model swap invalidates every stored vector.
        prev_model = store.get_meta("model")
        if prev_model and prev_model != embedder.model:
            full = True

        current = walk_vault()
        known = store.all_file_stats()
        changed = unchanged = 0
        chunks_embedded = 0

        for rel_path, abs_path in current.items():
            try:
                st = abs_path.stat()
            except OSError:
                continue
            db = known.get(rel_path)
            if db and not full and db[0] == st.st_mtime and db[1] == st.st_size:
                unchanged += 1
                continue
            try:
                text = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            content_hash = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            if db and not full and db[2] == content_hash:
                store.touch_file(rel_path, st.st_mtime, st.st_size)
                unchanged += 1
                continue

            frontmatter, body = split_frontmatter(text)
            tags = extract_tags(frontmatter)
            title = Path(rel_path).stem
            raw_chunks = chunk_note(body)
            embed_texts = [
                f"{title} > {hp}\n{txt}" if hp else f"{title}\n{txt}"
                for hp, txt in raw_chunks
            ]
            vectors = (
                embedder.embed_documents(embed_texts)
                if embed_texts
                else np.zeros((0, EMBED_DIM), dtype=np.float32)
            )
            store.replace_file(
                rel_path,
                st.st_mtime,
                st.st_size,
                content_hash,
                [
                    {"heading_path": hp, "text": txt, "tags": tags, "embedding": vectors[i]}
                    for i, (hp, txt) in enumerate(raw_chunks)
                ],
            )
            changed += 1
            chunks_embedded += len(raw_chunks)
            if progress:
                progress(changed, rel_path, len(raw_chunks))

        removed = sorted(set(known) - set(current))
        if removed:
            store.delete_files(removed)

        store.set_meta("model", embedder.model)
        store.set_meta("last_index_time", datetime.now(timezone.utc).isoformat(timespec="seconds"))

        return {
            "mode": "full" if full else "delta",
            "files_scanned": len(current),
            "files_changed": changed,
            "files_unchanged": unchanged,
            "files_removed": len(removed),
            "chunks_embedded": chunks_embedded,
            "duration_s": round(time.monotonic() - t0, 2),
        }
