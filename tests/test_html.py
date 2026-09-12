"""HTML indexing: page text and chunks, the vault + mount walk, and note lookup.

Run: .venv/bin/python -m unittest discover -s tests
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vault_mcp.embeddings import EMBED_DIM
from vault_mcp.indexer import chunk_html, html_title, mount_status, parse_mounts, walk_vault
from vault_mcp.store import Store

PARA = (
    "Late payments are chased on day seven with a firm but friendly reminder email, "
    "and escalate to a phone call from the account lead on day fourteen."
)


def page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html><head><title>{title}</title><style>.x{{color:red}}</style></head>"
        f"<body>{body}</body></html>"
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class ChunkHtmlTest(unittest.TestCase):
    def test_title_is_the_title_tag_as_text(self):
        self.assertEqual(html_title(page("GSFA &amp; <b>Late</b>   Payment", "")), "GSFA & Late Payment")
        self.assertEqual(html_title("<p>no title here</p>"), "")

    def test_splits_on_h1_to_h3_with_heading_paths(self):
        raw = page(
            "T",
            f"<h1>Playbook</h1><p>{PARA}</p><h2>Day <em>seven</em></h2><p>{PARA}</p>"
            f"<h3>Email</h3><p>{PARA}</p><h4>Not a split</h4><p>{PARA}</p><h2>Day 30</h2><p>{PARA}</p>",
        )
        self.assertEqual(
            [hp for hp, _ in chunk_html(raw)],
            ["Playbook", "Playbook > Day seven", "Playbook > Day seven > Email", "Playbook > Day 30"],
        )

    def test_drops_what_a_reader_never_sees(self):
        raw = page(
            "T",
            "<nav><a href='#a'>Every heading again</a></nav><script>var secret = 1;</script>"
            f"<!-- note to self --><template><p>hidden</p></template><header><p>{PARA}</p></header>",
        )
        text = " ".join(t for _, t in chunk_html(raw))
        for gone in ("Every heading again", "secret", "note to self", "hidden", "color:red"):
            self.assertNotIn(gone, text)
        self.assertIn("day seven", text)  # <header> is content; only <head> is metadata

    def test_entities_are_text_and_blocks_are_paragraphs(self):
        raw = page("T", f"<p>{PARA} &lt;div&gt; is a tag</p><ul><li>{PARA}</li><li>{PARA}</li></ul>")
        [(heading, text)] = chunk_html(raw)
        self.assertEqual(heading, "")
        self.assertIn("<div> is a tag", text)
        self.assertEqual(text.count("\n\n"), 2)

    def test_short_sections_drop_and_long_ones_split_between_blocks(self):
        raw = page("T", "<h2>Tiny</h2><p>too short</p><h2>Long</h2>" + f"<p>{PARA}</p>" * 60)
        chunks = chunk_html(raw)
        self.assertEqual({hp for hp, _ in chunks}, {"Long"})
        self.assertGreater(len(chunks), 1)
        for _, text in chunks:
            self.assertLessEqual(len(text), 2000)
            self.assertTrue(text.startswith("Late payments") and text.endswith("fourteen."))


class WalkTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.vault, self.artifacts = self.base / "vault", self.base / "artifacts"
        v, a = self.vault, self.artifacts
        write(v / "Meetings" / "kickoff.md", "# Kickoff")
        write(v / "Meetings" / "kickoff.html", page("Kickoff", ""))  # the .md's rendering
        (v / "Meetings" / "alias.md").symlink_to(v / "Meetings" / "kickoff.md")  # in-vault twin: kept
        write(v / "Resources" / "Playbook" / "playbook.html", page("Playbook", ""))
        write(v / "Other" / "Templates" / "template.md", "# t")
        write(a / "Learnings" / "index.html", page("Learnings", ""))  # project level: a page, not a folder
        write(a / "Learnings" / "one-pager.html", page("One pager", ""))
        write(a / "Learnings" / "Topic" / "notes.md", "# markdown is not an artifact")
        write(a / "Learnings" / "Topic" / "guides" / "g" / "index.html", page("Guide", ""))
        write(a / "Learnings" / "Topic" / "guides" / "g" / "index.inline.html", page("Guide", ""))
        write(a / "Learnings" / "Topic" / "node_modules" / "pkg.html", page("pkg", ""))
        write(a / "Learnings" / ".hidden.html", page("hidden", ""))
        (a / "Resources").mkdir()
        (a / "Resources" / "Playbook").symlink_to(v / "Resources" / "Playbook", target_is_directory=True)
        (a / "Learnings" / "copy.html").symlink_to(v / "Resources" / "Playbook" / "playbook.html")

    def test_vault_then_mounts_each_page_once(self):
        files = walk_vault(self.vault, [("Artifacts", self.artifacts)])
        self.assertEqual(
            set(files),
            {
                "Meetings/kickoff.md",
                "Meetings/alias.md",
                "Resources/Playbook/playbook.html",
                "Artifacts/Learnings/index.html",
                "Artifacts/Learnings/one-pager.html",
                "Artifacts/Learnings/Topic/guides/g/index.html",
            },
        )

    def test_a_mount_the_vault_shadows_is_left_out(self):
        (self.vault / "Artifacts").mkdir()
        mounts = [("Artifacts", self.artifacts), ("Gone", self.base / "nope")]
        self.assertEqual([m["status"] for m in mount_status(self.vault, mounts)], ["shadowed", "missing"])
        self.assertFalse([p for p in walk_vault(self.vault, mounts) if p.startswith("Artifacts/")])

    def test_parse_mounts(self):
        spec = os.pathsep.join(["Artifacts=~/x", "no-equals", ".hidden=/y", "A/B=/z", " Site = /s ", ""])
        self.assertEqual(parse_mounts(spec), [("Artifacts", Path.home() / "x"), ("Site", Path("/s"))])
        self.assertEqual(parse_mounts(""), [])


class ResolveNoteTest(unittest.TestCase):
    def test_html_paths_resolve_like_markdown(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = Store(Path(tmp.name) / "index.db")
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        guide = "Artifacts/Learnings/Topic/guides/g/index.html"
        for path in ("Meetings/kickoff.md", guide):
            store.replace_file(path, 0.0, 0, "h", [{"heading_path": "", "text": "x", "tags": "", "embedding": vec}])
        self.assertEqual(store.resolve_note(guide)[0], guide)
        self.assertEqual(store.resolve_note("guides/g/index.html")[0], guide)
        self.assertEqual(store.resolve_note("kickoff")[0], "Meetings/kickoff.md")


if __name__ == "__main__":
    unittest.main()
