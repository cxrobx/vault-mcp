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
from vault_mcp.launcher import (
    Launcher, Parsed, doc_date, index_tokens, name_match, name_score, parse, rendered_twin, scope_vocabulary,
)
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


class NameMatchTest(unittest.TestCase):
    def test_a_prefix_reaches_the_whole_word(self):
        own, folders = index_tokens("Clients/Acme/engagement-report.html", "The Acme Engagement Dossier")
        self.assertEqual(name_match(["doss"], own, folders), 2)
        self.assertEqual(name_match(["dossier"], own, folders), 2)
        self.assertEqual(name_match(["dossiers"], own, folders), 0)

    def test_folders_place_a_file_but_do_not_name_it(self):
        own, folders = index_tokens("Clients/Acme/Meetings/2026-09-02-sync.md", "2026-09-02-sync")
        self.assertEqual(name_match(["meeting"], own, folders), 1)
        self.assertEqual(name_match(["sync"], own, folders), 2)
        self.assertEqual(name_match(["meeting", "sync"], own, folders), 2)
        self.assertEqual(name_match(["meeting", "budget"], own, folders), 0)


class NamedFirstTest(unittest.TestCase):
    def rows(self, names: dict[str, str], phrase: str, mtimes: dict[str, float] | None = None):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "index.db")
            vec = np.zeros(EMBED_DIM, dtype=np.float32)
            vec[0] = 1.0
            for path, title in names.items():
                store.replace_file(
                    path, (mtimes or {}).get(path, 0.0), 1, "h",
                    [{"heading_path": "", "text": "the dossier is discussed at length here", "tags": "", "embedding": vec}],
                    title=title,
                )
            with mock.patch("vault_mcp.launcher.mount_status", return_value=[]):
                launcher = Launcher(store=store, embedder=FakeEmbedder())
            return [r["path"] for r in launcher.find(phrase)[1]]

    def test_the_file_named_by_the_words_beats_the_file_that_discusses_them(self):
        rows = self.rows(
            {"Clients/Acme/STATUS.md": "STATUS", "Clients/Acme/engagement-report.html": "Acme Engagement Dossier"},
            "acme doss",
        )
        self.assertEqual(rows[0], "Clients/Acme/engagement-report.html")

    def test_last_is_a_question_about_time_not_about_titles(self):
        # "acme" has to name a folder big enough to be a scope word, or it
        # stays in the topic and "last" is never at an end to be read.
        names = {f"Clients/Acme/filler-{i}.md": f"filler-{i}" for i in range(8)}
        names["Clients/Acme/Meetings/old Acme Meeting Notes.md"] = "old Acme Meeting Notes"
        names["Clients/Acme/Meetings/2026-09-02-preread.md"] = "2026-09-02-preread"
        rows = self.rows(names, "acme last meeting")
        self.assertEqual(rows[0], "Clients/Acme/Meetings/2026-09-02-preread.md")

    def test_a_recency_pool_gathers_on_any_word_not_all_of_them(self):
        names = {f"Clients/Acme/filler-{i}.md": f"filler-{i}" for i in range(8)}
        names["Clients/Acme/STATUS.md"] = "STATUS"                                  # neither word
        names["Clients/Acme/Audit/technical-seo-baseline-2026-05-05.md"] = "technical-seo-baseline-2026-05-05"
        names["Clients/Acme/engagement-report-2026-07-23.md"] = "engagement-report-2026-07-23"
        rows = self.rows(names, "acme last seo report", mtimes={"Clients/Acme/STATUS.md": 4e9})
        self.assertEqual(rows[0], "Clients/Acme/engagement-report-2026-07-23.md")
        self.assertNotIn("Clients/Acme/STATUS.md", rows[:2])


class OpenTest(unittest.TestCase):
    def test_a_typst_source_opens_its_built_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "globex"
            (root / "src").mkdir(parents=True)
            (root / "build").mkdir()
            src = root / "src" / "proposal.typ"
            src.write_text("x")
            self.assertIsNone(rendered_twin(src))          # nothing built yet
            (root / "build" / "proposal.pdf").write_text("%PDF")
            self.assertEqual(rendered_twin(src), root / "build" / "proposal.pdf")

    def test_other_files_have_no_twin(self):
        self.assertIsNone(rendered_twin(Path("/a/b/notes.md")))
        self.assertIsNone(rendered_twin(Path("/a/loose.typ")))


class ExcludeTest(unittest.TestCase):
    def test_globs_drop_files_and_whole_subtrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vault").mkdir()
            for rel in ("globex/src/proposal.typ", "globex/src/theme.typ", "_template/src/proposal.typ"):
                (root / "ext" / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / "ext" / rel).write_text("x" * 200)
            with mock.patch.object(indexer, "TEXT_SUFFIXES", (".typ",)):
                kept = indexer.walk_vault(
                    root / "vault",
                    mounts=[],
                    note_mounts=[("Proposals", root / "ext")],
                    exclude=("Proposals/*/src/theme.typ", "Proposals/_template/*"),
                )
            self.assertEqual(sorted(kept), ["Proposals/globex/src/proposal.typ"])
