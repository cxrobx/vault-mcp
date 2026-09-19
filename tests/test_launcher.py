"""The launcher's phrase parsing, document dates, recency ordering, and note mounts.

Run: .venv/bin/python -m unittest discover -s tests
"""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np

from vault_mcp import indexer
from vault_mcp.embeddings import EMBED_DIM
from vault_mcp.indexer import mount_status, walk_vault
from vault_mcp.launcher import Launcher, Parsed, doc_date, parse, scope_vocabulary
from vault_mcp.store import Store

VOCAB = {"acme": ["Clients/Acme"], "globex": ["Clients/Globex"], "jev": []}


class ParseTest(unittest.TestCase):
    def parsed(self, phrase: str) -> Parsed:
        return parse(phrase, {k: v for k, v in VOCAB.items() if v}, ["Artifacts"])

    def test_leading_scope_and_recency_leave_the_topic(self):
        p = self.parsed("acme last seo report")
        self.assertEqual((p.topic, p.folders, p.recent), ("acme seo report", ["Clients/Acme"], True))

    def test_most_recent_is_two_routing_words(self):
        p = self.parsed("most recent globex proposal")
        self.assertEqual((p.topic, p.folders, p.recent), ("globex proposal", ["Clients/Globex"], True))

    def test_vault_note_on_is_all_routing(self):
        p = self.parsed("vault note on relationships")
        self.assertEqual((p.topic, p.vault_only, p.folders), ("relationships", True, []))

    def test_trailing_in_onyx_scopes_to_the_page_mounts(self):
        p = self.parsed("open agent debugging guide in onyx")
        self.assertEqual((p.topic, p.folders), ("agent debugging guide", ["Artifacts"]))

    def test_md_is_a_file_kind_not_a_topic(self):
        p = self.parsed("md on jev usecases")
        self.assertEqual((p.topic, p.suffix, p.folders), ("jev usecases", ".md", []))

    def test_a_routing_word_in_the_middle_stays_in_the_topic(self):
        p = self.parsed("pricing for last mile delivery")
        self.assertEqual((p.topic, p.recent), ("pricing for last mile delivery", False))

    def test_a_small_folder_is_not_a_scope(self):
        paths = [f"Big/Acme/n{i}.md" for i in range(9)] + ["Small/Jev/a.md", "Small/Jev/b.md"]
        vocab = scope_vocabulary(paths)
        self.assertEqual(vocab.get("acme"), ["Big/Acme"])
        self.assertNotIn("jev", vocab)

    def test_a_small_folder_joins_a_scope_its_name_already_earned(self):
        paths = [f"Big/Acme/n{i}.md" for i in range(9)] + ["Proposals/acme/proposal.typ"]
        self.assertEqual(scope_vocabulary(paths)["acme"], ["Big/Acme", "Proposals/acme"])


class DocDateTest(unittest.TestCase):
    def test_filename_date_beats_mtime(self):
        self.assertEqual(doc_date("a/site-audit-2026-09-01.md", 0.0), date(2026, 9, 1))
        self.assertEqual(doc_date("Meetings/Acme Sync 08.28.26.md", 0.0), date(2026, 8, 28))

    def test_an_impossible_filename_date_falls_back_to_mtime(self):
        self.assertEqual(doc_date("Meetings/05.xx.26 agenda.md", 86400.0 * 365), doc_date("x.md", 86400.0 * 365))
        self.assertEqual(doc_date("v13.45.26.md", 86400.0 * 365), doc_date("x.md", 86400.0 * 365))


class FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        vec[0] = 1.0
        return vec


class RecencyTest(unittest.TestCase):
    def test_newest_file_named_for_the_topic_wins_over_a_newer_unrelated_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "index.db")
            vec = np.zeros(EMBED_DIM, dtype=np.float32)
            vec[0] = 1.0
            names = [f"Clients/Globex/filler-{i}.md" for i in range(5)] + [
                "Clients/Globex/Meetings/Globex Sync 09.02.26.md",
                "Clients/Globex/globex-proposal-2026-04-23.md",
                "Clients/Globex/globex-proposal-2026-06-01.md",
            ]
            for name in names:
                store.replace_file(name, 0.0, 1, "h", [{"heading_path": "", "text": "globex proposal terms", "tags": "", "embedding": vec}])
            with mock.patch("vault_mcp.launcher.mount_status", return_value=[]):
                launcher = Launcher(store=store, embedder=FakeEmbedder())
            _, rows = launcher.find("most recent globex proposal")
            self.assertEqual(
                [r["path"] for r in rows[:2]],
                ["Clients/Globex/globex-proposal-2026-06-01.md", "Clients/Globex/globex-proposal-2026-04-23.md"],
            )
            self.assertIn("Clients/Globex/Meetings/Globex Sync 09.02.26.md", [r["path"] for r in rows])


class NoteMountTest(unittest.TestCase):
    def test_a_note_mount_gives_markdown_and_text_suffixes_a_page_mount_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vault").mkdir()
            for rel in ("docs/plan.md", "docs/deep/a/b/index.html", "docs/deep/a/b/extra.md", "docs/offer.typ", "docs/node_modules/x.md"):
                (root / "ext" / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / "ext" / rel).write_text("x" * 200)
            with mock.patch.object(indexer, "TEXT_SUFFIXES", (".typ",)):
                notes = walk_vault(root / "vault", mounts=[], note_mounts=[("Repos", root / "ext")])
                pages = walk_vault(root / "vault", mounts=[("Repos", root / "ext")], note_mounts=[])
            self.assertEqual(
                sorted(notes),
                ["Repos/docs/deep/a/b/extra.md", "Repos/docs/deep/a/b/index.html", "Repos/docs/offer.typ", "Repos/docs/plan.md"],
            )
            self.assertEqual(sorted(pages), ["Repos/docs/deep/a/b/index.html"])

    def test_a_second_mount_of_the_same_name_is_shadowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ("vault", "a", "b"):
                (root / d).mkdir()
            status = mount_status(root / "vault", mounts=[("Docs", root / "a")], note_mounts=[("Docs", root / "b")])
            self.assertEqual([(m["kind"], m["status"]) for m in status], [("pages", "ok"), ("notes", "shadowed")])


if __name__ == "__main__":
    unittest.main()


class AgentFileTest(unittest.TestCase):
    def launcher(self, store):
        with mock.patch("vault_mcp.launcher.mount_status", return_value=[]):
            return Launcher(store=store, embedder=FakeEmbedder())

    def index(self, tmp: str, names: list[str]) -> Store:
        store = Store(Path(tmp) / "index.db")
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        vec[0] = 1.0
        for name in names:
            store.replace_file(name, 0.0, 1, "h", [{"heading_path": "", "text": "acme engagement posture", "tags": "", "embedding": vec}])
        return store

    def test_agent_files_are_kept_off_a_typed_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.index(tmp, ["Clients/Acme/CLAUDE.md", "Clients/Acme/AGENTS.md", "Clients/Acme/engagement-report.html"])
            _, rows = self.launcher(store).find("acme dossier")
            self.assertEqual([r["path"] for r in rows], ["Clients/Acme/engagement-report.html"])

    def test_naming_one_brings_it_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.index(tmp, ["Clients/Acme/CLAUDE.md", "Clients/Acme/engagement-report.html"])
            _, rows = self.launcher(store).find("acme claude")
            self.assertIn("Clients/Acme/CLAUDE.md", [r["path"] for r in rows])
