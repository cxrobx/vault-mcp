"""Vault walker, markdown and HTML chunkers, and delta-reindex logic.

Walks the vault following symlinks (symlinked folders index like real ones)
with a realpath cycle guard, then each mounted folder (VAULT_MCP_MOUNTS) under
its own name. Markdown chunks split on H1–H3 headings outside code fences;
HTML pages split on <h1>–<h3> over the text a reader sees. Each chunk is
embedded as "{title} > {heading path}\\n{body}". Delta reindex: mtime+size
stat sweep, content-hash confirm, per-file transaction — a crash never leaves
a file half-indexed.
"""

import hashlib
import html
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import MOUNTS_SPEC, VAULT_PATH
from .embeddings import EMBED_DIM, OllamaEmbedder
from .store import Store

EXCLUDE_DIR_NAMES = {
    ".obsidian", ".smart-env", ".trash", ".git", ".SynologyWorkingDirectory", "node_modules", "__pycache__",
}
EXCLUDE_REL_PATHS = {"Other/Templates"}
MAX_CHUNK_CHARS = 2000
MIN_CHUNK_CHARS = 80
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")

HTML_SUFFIXES = {".html", ".htm"}
INDEX_NAMES = {"index.html", "index.htm"}
# Onyx's own cap on reading a page's text. A bigger page is kept with no
# chunks, so the delta sweep's stat check passes over it until it changes.
MAX_HTML_BYTES = 4_000_000

_reindex_lock = threading.Lock()


def parse_mounts(spec: str) -> list[tuple[str, Path]]:
    """ "Name=/path" pairs joined by os.pathsep -> [(name, path)]; malformed pairs are dropped."""
    mounts: list[tuple[str, Path]] = []
    for item in spec.split(os.pathsep):
        name, sep, raw = item.partition("=")
        name, raw = name.strip(), raw.strip()
        if sep and name and raw and "/" not in name and not name.startswith("."):
            mounts.append((name, Path(raw).expanduser()))
    return mounts


MOUNTS = parse_mounts(MOUNTS_SPEC)


def mount_status(vault: Path = VAULT_PATH, mounts: list[tuple[str, Path]] | None = None) -> list[dict]:
    """Each mount and whether it is indexed: "ok", "missing", or "shadowed".

    A mount is shadowed when the vault has a top-level entry of the same name:
    its paths would be indistinguishable from the vault's own, so it stays out.
    """
    out = []
    for name, path in MOUNTS if mounts is None else mounts:
        if os.path.lexists(vault / name):
            status = "shadowed"
        elif not path.is_dir():
            status = "missing"
        else:
            status = "ok"
        out.append({"name": name, "path": str(path), "status": status})
    return out


def walk_vault(vault: Path = VAULT_PATH, mounts: list[tuple[str, Path]] | None = None) -> dict[str, Path]:
    """Map index path -> absolute path for every in-scope file.

    The vault gives its markdown notes and its HTML notes — except an .html
    beside a same-name .md, which is that note's rendering, so the .md stands
    for both. Each mount then gives its HTML pages under "<name>/", listed the
    way Onyx's Artifacts sidebar lists them: below the project level, a folder
    holding index.html is one page, and the rest of that folder (assets,
    inlined copies) stays out.

    Mounts never repeat what an earlier root already gave: a folder or file
    reachable from the vault is skipped when a mount reaches it again, which
    files a page that is in both under the vault — where Onyx files it too.
    Inside the vault nothing changes: only the folder cycle guard applies, so a
    note symlinked to another note keeps both addresses, as it always has.
    """
    files: dict[str, Path] = {}
    visited: set[Path] = set()
    seen: set[Path] = set()

    def _add(rel: str, entry: Path, mounted: bool) -> None:
        try:
            real = entry.resolve()
        except OSError:
            return
        if mounted and real in seen:
            return
        seen.add(real)
        files[rel] = entry

    def _walk(dirpath: Path, rel: str, depth: int, mounted: bool) -> None:
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
        if mounted and depth >= 2:
            index = next((e for e in entries if e.name.lower() in INDEX_NAMES and e.is_file()), None)
            if index is not None:
                _add(f"{rel}/{index.name}", index, mounted)
                return
        md_stems = set() if mounted else {e.stem for e in entries if e.name.endswith(".md")}
        for entry in entries:
            rel_child = f"{rel}/{entry.name}" if rel else entry.name
            if entry.is_dir():
                if entry.name.startswith(".") or entry.name in EXCLUDE_DIR_NAMES or rel_child in EXCLUDE_REL_PATHS:
                    continue
                _walk(entry, rel_child, depth + 1, mounted)
            elif not entry.is_file():
                continue
            elif not mounted and entry.name.endswith(".md"):
                _add(rel_child, entry, mounted)
            elif (
                entry.suffix.lower() in HTML_SUFFIXES
                and not entry.name.startswith(".")
                and entry.stem not in md_stems
            ):
                _add(rel_child, entry, mounted)

    _walk(vault, "", 0, False)
    for mount in mount_status(vault, mounts):
        if mount["status"] == "ok":
            _walk(Path(mount["path"]), mount["name"], 0, True)
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


def _pieces(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Split oversized sections and drop the too-short, keeping each heading path."""
    chunks: list[tuple[str, str]] = []
    for heading_path, text in sections:
        for piece in _split_oversized(text):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append((heading_path, piece))
    return chunks


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

    return _pieces(sections)


# What a reader of an HTML page never sees: code and metadata elements,
# comments, and the page's <nav> — a table of contents repeating every
# heading, which would otherwise rank for a query about any one section.
_HTML_HIDDEN_RE = re.compile(
    r"<(script|style|template|noscript|head|nav)\b.*?</\1\s*>|<!--.*?-->", re.IGNORECASE | re.DOTALL
)
_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL)
_HTML_HEADING_RE = re.compile(r"<h([1-3])\b[^>]*>(.*?)</h\1\s*>", re.IGNORECASE | re.DOTALL)
# Tags that end a block of text. Each becomes a paragraph break, so an
# oversized section still splits between blocks rather than mid-sentence.
_HTML_BLOCK_RE = re.compile(
    r"</?(?:p|div|section|article|main|header|footer|aside|figure|figcaption|blockquote|pre"
    r"|ul|ol|li|dl|dt|dd|table|thead|tbody|tr|details|summary|h[1-6]|br|hr)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _html_text(fragment: str) -> str:
    """An HTML fragment as plain text, one paragraph per block."""
    text = html.unescape(_HTML_TAG_RE.sub(" ", _HTML_BLOCK_RE.sub("\n\n", fragment)))
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paragraphs if p)


def html_title(raw: str) -> str:
    """The page's <title>, which is what Onyx names it by; "" when it has none."""
    match = _HTML_TITLE_RE.search(raw)
    return " ".join(html.unescape(_HTML_TAG_RE.sub("", match.group(1))).split()) if match else ""


def _page_name(rel_path: str) -> str:
    """Onyx's name for a page with no <title>: its folder for an index page, else its filename."""
    path = Path(rel_path)
    return path.parent.name if path.name.lower() in INDEX_NAMES and path.parent.name else path.stem


def chunk_html(raw: str) -> list[tuple[str, str]]:
    """Split an HTML page into (heading_path, text) chunks on <h1>–<h3> boundaries."""
    body = _HTML_HIDDEN_RE.sub(" ", raw)
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    start = 0

    def flush(fragment: str) -> None:
        text = _html_text(fragment)
        if text:
            sections.append((" > ".join(t for _, t in stack), text))

    for match in _HTML_HEADING_RE.finditer(body):
        flush(body[start : match.start()])
        level = int(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        heading = " ".join(html.unescape(_HTML_TAG_RE.sub(" ", match.group(2))).split())
        if heading:
            stack.append((level, heading))
        start = match.end()
    flush(body[start:])

    return _pieces(sections)


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
            is_html = abs_path.suffix.lower() in HTML_SUFFIXES
            try:
                if is_html and st.st_size > MAX_HTML_BYTES:
                    text = ""
                else:
                    text = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            content_hash = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
            if db and not full and db[2] == content_hash:
                store.touch_file(rel_path, st.st_mtime, st.st_size)
                unchanged += 1
                continue

            if is_html:
                tags = ""
                title = html_title(text) or _page_name(rel_path)
                raw_chunks = chunk_html(text)
            else:
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
