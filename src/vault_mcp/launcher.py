"""A typed-phrase front end over the index, for a launcher such as Alfred.

Not an MCP tool and not meant to become one. A person types "acme last seo
report" or "vault note on relationships"; an agent never needs to, because it
can pass `folder=` to search_vault directly. So the phrase parsing lives here,
outside the server, and the store only ever sees structured arguments.

A phrase carries up to three things besides its topic, and embedding them with
the topic hurts: "vault note on relationships" retrieves notes about vaults.

* a scope      — "acme", "vault", "in onyx": which part of the index
* a recency    — "last", "most recent": order by date rather than by match
* a file kind  — "md", "page"

They are read only from the leading and trailing runs of the phrase, never
from its middle, so "last mile delivery" keeps its "last".

Every run and every pick is appended to a JSONL log. That log is the point of
version 1: it shows how often the wanted file is outside the top three, and
what those phrases have in common, before anything heavier is built.

Run:  python -m vault_mcp.launcher query "<phrase>"   -> Alfred Script Filter JSON
      python -m vault_mcp.launcher pick "<abs path>"  -> log the pick, open the file

No daemon: a cold call — interpreter, index load, query embedding, search —
measured 0.28 s on an 1,100-file index, under what a launcher needs.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from . import DB_PATH, VAULT_PATH
from .indexer import HTML_SUFFIXES, INDEX_NAMES, html_title, mount_status

LOG_PATH = Path(os.environ.get("VAULT_MCP_LAUNCHER_LOG", "~/.local/state/vault-mcp/launcher.jsonl")).expanduser()
SHOWN = 8
# A recency phrase with no file named for its topic falls back to this many of
# the best topic matches, newest first.
RECENCY_FALLBACK = 10
# A folder name only scopes a phrase when the folder is big enough that
# narrowing to it means something. "jev" names a two-file folder and is also
# simply the topic; "acme" names a client with two hundred files.
MIN_SCOPE_FILES = 8
RRF_K = 60

RECENCY_WORDS = {"last", "latest", "newest", "recent"}
FILLER_WORDS = {
    "the", "my", "a", "an", "open", "show", "find", "note", "notes", "doc", "docs", "file",
    "on", "about", "in", "from", "for", "of", "most", "that",
}
KIND_WORDS = {"md": ".md", "markdown": ".md", "html": ".html", "page": ".html"}
# Words that name a part of the index rather than a folder in it.
VAULT_WORDS = {"vault", "obsidian"}
PAGES_WORDS = {"onyx", "artifacts"}

_TOKEN_RE = re.compile(r"[0-9a-z]+")
_DATE_ISO_RE = re.compile(r"(?<!\d)(20\d\d)-(\d\d)-(\d\d)(?!\d)")
_DATE_US_RE = re.compile(r"(?<!\d)(\d\d)\.(\d\d)\.(\d\d)(?!\d)")
# Files written FOR an agent, never opened by a person. They stay in the index
# (an agent may well search for one) but are kept off a typed list unless the
# phrase asks for them by name — they are long, they touch every topic their
# project touches, and so they match almost anything.
AGENT_FILES = {"claude.md", "claude.local.md", "agents.md", "gemini.md", "cursor.md"}
# Stems that say nothing without the folder they sit in.
_GENERIC_FOLDERS = {"src", "docs", "doc", "plans", "specs", "build", "guides"}
_GENERIC_STEMS = {"index", "readme", "claude", "agents", "status", "changelog", "proposal", "sections", "plan", "notes"}


@dataclass
class Parsed:
    topic: str
    scope_words: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    recent: bool = False
    suffix: str = ""
    # "vault" names the vault itself, which has no prefix of its own: it is
    # whatever is not under a mount, so it is a path test rather than a folder.
    vault_only: bool = False


def scope_vocabulary(paths: list[str]) -> dict[str, list[str]]:
    """Folder name (lowercased, single word) -> the folders of that name worth scoping to."""
    counts: dict[str, int] = {}
    for path in paths:
        parts = path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            prefix = "/".join(parts[:depth])
            counts[prefix] = counts.get(prefix, 0) + 1
    # Once one folder of a name is big enough to be a scope, every folder of
    # that name is in it: the client's three-file proposal folder belongs to
    # "globex" as much as its two-hundred-file engagement folder does.
    by_name: dict[str, list[str]] = {}
    for prefix in counts:
        name = prefix.rsplit("/", 1)[-1].lower()
        if " " not in name:
            by_name.setdefault(name, []).append(prefix)
    return {
        name: sorted(prefixes)
        for name, prefixes in by_name.items()
        if max(counts[p] for p in prefixes) >= MIN_SCOPE_FILES
    }


def parse(phrase: str, vocab: dict[str, list[str]], page_mounts: list[str]) -> Parsed:
    """Split a typed phrase into its topic and its routing words.

    Routing words are taken from the leading run and the trailing run only. A
    folder name stays in the topic as well as scoping the search: inside the
    scope it costs nothing, and if the scope comes back empty the unscoped
    retry still knows what was asked for.
    """
    tokens = phrase.split()
    parsed = Parsed(topic="")
    keep = [True] * len(tokens)

    def route(i: int) -> bool:
        word = tokens[i].lower().strip(",.;:")
        if word in RECENCY_WORDS:
            parsed.recent = True
        elif word in KIND_WORDS:
            parsed.suffix = KIND_WORDS[word]
        elif word in VAULT_WORDS:
            parsed.scope_words.append(word)
            parsed.vault_only = True
        elif word in PAGES_WORDS:
            parsed.scope_words.append(word)
            parsed.folders.extend(page_mounts)
        elif word in vocab:
            parsed.scope_words.append(word)
            parsed.folders.extend(vocab[word])
            return True  # routed, but kept in the topic
        elif word not in FILLER_WORDS:
            return False
        keep[i] = False
        return True

    i = 0
    while i < len(tokens) and route(i):
        i += 1
    j = len(tokens) - 1
    while j >= i and route(j):
        j -= 1

    parsed.topic = " ".join(t for t, k in zip(tokens, keep) if k)
    return parsed


def index_tokens(rel_path: str, title: str) -> tuple[frozenset[str], frozenset[str]]:
    """The words a person could call this file by: (its own, its folders').

    Its own are the title and the filename — a document named "Engagement
    Dossier" is what someone typing "dossier" means. Its folders are weaker: a
    whole client folder shares them, so they place a file rather than name it.
    """
    own = f"{title} {Path(rel_path).stem}"
    folders = " ".join(Path(rel_path).parts[:-1])
    return frozenset(_TOKEN_RE.findall(own.lower())), frozenset(_TOKEN_RE.findall(folders.lower()))


def name_match(terms: list[str], own: frozenset[str], folders: frozenset[str]) -> int:
    """0 = not named by these words, 1 = its folders carry them, 2 = its own name does.

    Every term must match, and a term matches a token it is a PREFIX of — the
    phrase is typed a character at a time, so "doss" has to reach "Dossier"
    before the last two letters arrive. Rank 2 needs at least one term in the
    file's own name, or every file in a client folder would be "named" by the
    client's word alone.
    """
    if not terms:
        return 0
    strong = False
    for term in terms:
        if any(t.startswith(term) for t in own):
            strong = True
        elif not any(t.startswith(term) for t in folders):
            return 0
    return 2 if strong else 1


def name_score(terms: list[str], own: frozenset[str], folders: frozenset[str]) -> int:
    """How much of these words the file carries: 2 per term in its own name, 1 per term in its folders.

    The graded companion to `name_match`. Naming a file demands every word;
    gathering the files a phrase is ABOUT does not — "seo report" has to reach
    both the SEO baseline and the engagement report, neither of which carries
    both words.
    """
    total = 0
    for term in terms:
        if any(t.startswith(term) for t in own):
            total += 2
        elif any(t.startswith(term) for t in folders):
            total += 1
    return total


def doc_date(rel_path: str, mtime: float) -> date:
    """The date a document is about: one in its filename, else when it last changed."""
    name = rel_path.rsplit("/", 1)[-1]
    for regex, order in ((_DATE_ISO_RE, (0, 1, 2)), (_DATE_US_RE, (2, 0, 1))):
        match = regex.search(name)
        if match:
            y, m, d = (int(match.group(i + 1)) for i in order)
            try:
                return date(y if y > 99 else 2000 + y, m, d)
            except ValueError:
                continue
    return datetime.fromtimestamp(mtime).date()


class Launcher:
    def __init__(self, store=None, embedder=None):
        from .embeddings import OllamaEmbedder
        from .store import Store

        self.store = store or Store(DB_PATH)
        self.embedder = embedder or OllamaEmbedder()
        mounts = [m for m in mount_status() if m["status"] == "ok"]
        self.mount_paths = {m["name"]: Path(m["path"]) for m in mounts}
        self.page_mounts = [m["name"] for m in mounts if m["kind"] == "pages"]

    def _files(self) -> dict[str, float]:
        return {path: mtime for path, (mtime, _) in self.store.file_index().items()}

    def _tokens(self) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
        return {p: index_tokens(p, title) for p, (_, title) in self.store.file_index().items()}

    def abs_path(self, rel_path: str) -> Path:
        head, _, rest = rel_path.partition("/")
        if head in self.mount_paths and rest:
            return self.mount_paths[head] / rest
        return VAULT_PATH / rel_path

    def _in_vault(self, rel_path: str) -> bool:
        return rel_path.partition("/")[0] not in self.mount_paths

    def find(self, phrase: str, k: int = SHOWN) -> tuple[Parsed, list[dict]]:
        files = self._files()
        tokens = self._tokens()
        parsed = parse(phrase, scope_vocabulary(list(files)), self.page_mounts)

        topic_words = {w.lower().strip(",.;:") for w in parsed.topic.split()}
        asked_for_agent_file = bool(topic_words & {Path(n).stem.lower() for n in AGENT_FILES})

        def allowed(path: str) -> bool:
            if parsed.suffix and not path.lower().endswith(parsed.suffix):
                return False
            if not asked_for_agent_file and path.rsplit("/", 1)[-1].lower() in AGENT_FILES:
                return False
            return self._in_vault(path) if parsed.vault_only else True

        ranked = self._topic_ranked(parsed, allowed) if parsed.topic else []
        ranked = self._named_first(parsed, ranked, files, tokens, allowed)
        if not ranked and parsed.folders and parsed.topic:
            # The scope matched nothing on this topic. Asking again without it
            # beats an empty list: the scope word is still in the topic.
            ranked = self._topic_ranked(Parsed(topic=parsed.topic), allowed)

        if parsed.recent:
            ranked = self._by_date(parsed, ranked, files, tokens, allowed)

        rows = []
        for row in ranked[:k]:
            path = row["path"]
            rows.append(
                {
                    "path": path,
                    "abs": str(self.abs_path(path)),
                    "heading": row.get("heading", ""),
                    "date": (row.get("date") or doc_date(path, files.get(path, 0.0))).isoformat(),
                }
            )
        return parsed, rows

    def _terms(self, parsed: Parsed) -> list[str]:
        """The topic's own words: what is left once the scope words are taken out."""
        scope = set(parsed.scope_words)
        return [w for w in (t.lower().strip(",.;:") for t in parsed.topic.split()) if w not in scope and len(w) > 1]

    def _named_first(
        self, parsed: Parsed, ranked: list[dict], files: dict[str, float], tokens: dict, allowed
    ) -> list[dict]:
        """Put the files these words NAME above the files that merely discuss them.

        Content search cannot do this. A long status page mentioning the
        dossier outranks the dossier itself, because the dossier's own name is
        one line of it and the status page is full of the subject. And a
        half-typed word reaches the content legs as nothing at all: BM25 has no
        prefix matching here, so "doss" matches zero rows and the search
        silently degrades to dense-only, which is what let the status page win.
        """
        terms = self._terms(parsed)
        if not terms:
            return ranked
        prefixes = tuple(f.strip("/") + "/" for f in parsed.folders)
        rank = {r["path"]: r["rank"] for r in ranked}
        named: list[tuple[int, int, float, dict]] = []
        for path, (own, folders) in tokens.items():
            if not allowed(path) or (prefixes and not path.startswith(prefixes)):
                continue
            score = name_match(terms, own, folders)
            if score:
                row = {"path": path, "heading": "", "rank": rank.get(path, len(tokens))}
                named.append((-score, row["rank"], -files.get(path, 0.0), row))
        if not named:
            return ranked
        named.sort(key=lambda x: x[:3])
        head = [row for *_, row in named]
        seen = {r["path"] for r in head}
        return head + [r for r in ranked if r["path"] not in seen]

    def _by_date(
        self, parsed: Parsed, ranked: list[dict], files: dict[str, float], tokens: dict, allowed
    ) -> list[dict]:
        """Order a recency phrase: the newest document OF THAT KIND, not the newest good match.

        Match strength cannot pick the pool. Inside one client's folder every
        document embeds close to every other — measured on this index, the
        meeting notes, the status page and the actual proposal all sat within
        0.05 cosine of each other for "globex proposal" — so "the 40 best matches"
        is the whole folder and the newest meeting note wins. What says a file
        IS a proposal, a report or a meeting is its name and the folders it
        sits in, so the pool is the files whose path carries a topic word, and
        date orders within it. With no such file, the few best matches by date.
        """
        terms = [t for t in self._terms(parsed) if len(t) > 2]
        prefixes = tuple(f.strip("/") + "/" for f in parsed.folders)
        rank = {r["path"]: r for r in ranked}
        in_scope = [p for p in files if allowed(p) and (not prefixes or p.startswith(prefixes))]
        if not in_scope:
            in_scope = [p for p in files if allowed(p)]

        def row(path: str, matched: int) -> dict:
            base = rank.get(path, {"path": path, "heading": "", "rank": len(files)})
            return {**base, "date": doc_date(path, files[path]), "matched": matched}

        pool = [row(p, n) for p in in_scope if (n := name_score(terms, *tokens[p]))] if terms else []
        if not pool:
            pool = [row(r["path"], 0) for r in ranked[:RECENCY_FALLBACK]] if terms else [row(p, 0) for p in in_scope]
        # Date first, and only then how strongly the name matched. "Last" is a
        # question about time: among the files these words fit, the newest is
        # the answer, even when an older one wears the word in its title. A
        # 2026-04 file called "Acme Meeting Notes" is not the last meeting.
        pool.sort(key=lambda r: (-r["date"].toordinal(), -r["matched"], r["rank"]))
        # A short pool is padded with the plain topic order, so a file the
        # name test missed is still on the list rather than absent.
        named = {r["path"] for r in pool}
        return pool + [r for r in ranked if r["path"] not in named]

    def _topic_ranked(self, parsed: Parsed, allowed) -> list[dict]:
        qvec = self.embedder.embed_query(parsed.topic)
        hits = self.store.search(
            qvec, k=400, query_text=parsed.topic, folders=parsed.folders or None, candidates=400
        )
        seen: dict[str, dict] = {}
        for hit in hits:
            path = hit["path"]
            if path not in seen and allowed(path):
                seen[path] = {"path": path, "heading": hit["heading_path"], "rank": len(seen) + 1}
        return list(seen.values())


def display_title(abs_path: Path, rel_path: str) -> str:
    path = Path(rel_path)
    title = ""
    if path.suffix.lower() in HTML_SUFFIXES:
        try:
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                title = html_title(fh.read(16_384))
        except OSError:
            title = ""
    if not title:
        title = path.stem
        if path.name.lower() in INDEX_NAMES or path.stem.lower() in _GENERIC_STEMS:
            # Name it by the folders above it, skipping the ones that are as
            # generic as the file ("globex / proposal.typ", not "src / proposal.typ").
            parents = [p for p in path.parts[:-1] if p.lower() not in _GENERIC_FOLDERS][-2:]
            title = " / ".join([*parents[-1:], path.name]) if parents else path.name
    return title


def alfred_items(phrase: str, parsed: Parsed, rows: list[dict]) -> dict:
    shown = [r["path"] for r in rows]
    items = []
    for rank, row in enumerate(rows, start=1):
        folder = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else ""
        subtitle = " · ".join(x for x in (row["date"], folder, row["heading"]) if x)
        items.append(
            {
                "title": display_title(Path(row["abs"]), row["path"]),
                "subtitle": subtitle,
                "arg": row["abs"],
                "type": "file:skipcheck",
                "quicklookurl": row["abs"],
                "text": {"copy": row["abs"]},
                "variables": {"pick_rank": str(rank), "pick_query": phrase, "pick_shown": json.dumps(shown)},
            }
        )
    if not items:
        scope = f" in {', '.join(parsed.scope_words)}" if parsed.scope_words else ""
        items.append({"title": "Nothing found", "subtitle": f'"{parsed.topic}"{scope}', "valid": False})
    return {"items": items}


def log_event(event: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **event}) + "\n")
    except OSError:
        pass  # a full disk must not break the launcher


def open_file(abs_path: str) -> None:
    """Vault markdown in Obsidian, HTML in Onyx, anything else in its default app."""
    path = Path(abs_path)
    suffix = path.suffix.lower()
    try:
        in_vault = path.resolve().is_relative_to(VAULT_PATH.resolve())
    except OSError:
        in_vault = False
    if suffix == ".md" and in_vault:
        target = ["open", "obsidian://open?path=" + urllib.parse.quote(str(path), safe="")]
    elif suffix in HTML_SUFFIXES:
        target = ["open", "-b", "com.cx.onyx", str(path)]
    else:
        target = ["open", str(path)]
    if subprocess.run(target, check=False).returncode != 0 and target[1] == "-b":
        subprocess.run(["open", str(path)], check=False)


def run_query(launcher: Launcher, phrase: str) -> dict:
    t0 = time.monotonic()
    phrase = phrase.strip()
    if phrase in ("", "...", "…"):
        return {"items": [{"title": "Find a document", "subtitle": "acme last seo report · vault note on relationships", "valid": False}]}
    parsed, rows = launcher.find(phrase)
    log_event(
        {
            "event": "query",
            "query": phrase,
            "parsed": {k: v for k, v in asdict(parsed).items() if v},
            "shown": [r["path"] for r in rows],
            "ms": round((time.monotonic() - t0) * 1000),
        }
    )
    return alfred_items(phrase, parsed, rows)


def main(argv: list[str]) -> None:
    command = argv[1] if len(argv) > 1 else ""
    if command == "query":
        print(json.dumps(run_query(Launcher(), " ".join(argv[2:]))))
    elif command == "pick":
        path = argv[2]
        log_event(
            {
                "event": "pick",
                "query": os.environ.get("pick_query", ""),
                "rank": int(os.environ.get("pick_rank") or 0),
                "path": path,
                "shown": json.loads(os.environ.get("pick_shown") or "[]"),
            }
        )
        open_file(path)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
