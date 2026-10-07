"""Media files in the walk and the indexer (decoding itself is the embed server's job).

Run: .venv/bin/python -m unittest discover -s tests
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from vault_mcp import indexer, media
from vault_mcp.embeddings import EMBED_DIM
from vault_mcp.indexer import walk_vault
from vault_mcp.store import Store

ALL = frozenset({"pdf", "image", "video"})


def _vecs(n: int) -> np.ndarray:
    v = np.zeros((n, EMBED_DIM), dtype=np.float32)
    v[:, 0] = 1.0
    return v


class FakeEmbedder:
    model = "fake"
    multimodal = True

    def format_document(self, title, heading, text):
        return f"{title}|{heading}|{text}"

    def embed_documents(self, texts):
        return _vecs(len(texts))

    def embed_items(self, items):
        self.items = items
        return _vecs(len(items))


class WalkTest(unittest.TestCase):
    def test_media_is_walked_only_when_enabled_and_exports_stay_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            for name in ["note.md", "note.pdf", "deck.pdf", "clip.mp4", ".hidden.png", "song.m4a", "icon.png"]:
                (vault / name).write_bytes(b"x")
            (vault / "shot.png").write_bytes(b"x" * media.MIN_IMAGE_BYTES)
            off = walk_vault(vault, mounts=[], note_mounts=[], exclude=(), media_kinds=frozenset())
            self.assertEqual(set(off), {"note.md"})
            on = walk_vault(vault, mounts=[], note_mounts=[], exclude=(), media_kinds=ALL)
            # note.pdf is note.md's export; .m4a isn't a kind; hidden files never are; icon.png is too small.
            self.assertEqual(set(on), {"note.md", "deck.pdf", "shot.png", "clip.mp4"})

    def test_pages_mounts_never_take_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            vault, art = Path(tmp) / "v", Path(tmp) / "a"
            vault.mkdir(), art.mkdir()
            (art / "pic.png").write_bytes(b"x")
            files = walk_vault(vault, mounts=[("Artifacts", art)], note_mounts=[], exclude=(), media_kinds=ALL)
            self.assertEqual(files, {})


class IndexMediaTest(unittest.TestCase):
    def _index(self, chunks=None, error=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        f = Path(tmp.name) / "deck.pdf"
        f.write_bytes(b"%PDF")
        store = Store(Path(tmp.name) / "index.db")
        emb = FakeEmbedder()
        side = {"side_effect": error} if error else {"return_value": chunks}
        with mock.patch.object(media, "chunks_for", **side):
            out = indexer._index_media(store, emb, "deck.pdf", f, f.stat(), "pdf", None, False)
        return store, emb, out

    def test_text_and_visual_pages_keep_page_order_and_kind(self):
        chunks = [
            {"heading": "p. 1", "text": "cover", "media": {"image": {"pdf": "/x.pdf", "page": 1}}},
            {"heading": "p. 2", "text": "a long page of words", "media": None},
            {"heading": "p. 3", "text": "", "media": {"image": {"pdf": "/x.pdf", "page": 3}}},
        ]
        store, emb, out = self._index(chunks)
        self.assertEqual(out, 3)
        hits = store.search(_vecs(1)[0], k=10)
        self.assertEqual(
            sorted((h["heading_path"], h["kind"]) for h in hits),
            [("p. 1", "pdf"), ("p. 2", "text"), ("p. 3", "pdf")],
        )
        cache = store._get_cache()
        self.assertEqual(cache["headings"], ["p. 1", "p. 2", "p. 3"])
        # Placeholders go into the text the model sees; the stored text stays clean.
        self.assertEqual(emb.items[0]["text"], "deck|p. 1|cover <|image|>")
        self.assertEqual(emb.items[0]["image"], {"pdf": "/x.pdf", "page": 1})

    def test_an_unreadable_file_is_reported_not_stored_empty(self):
        store, _, out = self._index(error=media.MediaError("pdftotext failed"))
        self.assertEqual(out, "pdftotext failed")
        self.assertEqual(store.all_file_stats(), {})


class VideoWindowTest(unittest.TestCase):
    def test_windows_carry_their_slice_of_the_transcript(self):
        words = [(1.0, 4.0, "power it on"), (29.0, 33.0, "hold the trigger"), (61.0, 62.0, "done")]
        with mock.patch.object(media, "_duration", return_value=65.0), \
             mock.patch.object(media, "_has_audio", return_value=True), \
             mock.patch.object(media, "transcript", return_value=words):
            chunks = media.video_chunks(Path("/v.mp4"))
        windows = [c for c in chunks if c["media"]]
        self.assertEqual([c["heading"] for c in windows], ["@ 0:00", "@ 0:30", "@ 1:00"])
        self.assertEqual([c["text"] for c in windows], ["power it on hold the trigger", "hold the trigger", "done"])
        self.assertEqual(windows[2]["media"]["video"], {"path": "/v.mp4", "start": 60.0, "duration": 5.0})
        self.assertIn("audio", windows[0]["media"])

    def test_long_enough_narration_is_also_a_text_chunk(self):
        said = "tap tools and choose select template; your supervisor decides which one you should use"
        with mock.patch.object(media, "_duration", return_value=20.0), \
             mock.patch.object(media, "_has_audio", return_value=True), \
             mock.patch.object(media, "transcript", return_value=[(0.0, 9.0, said)]):
            chunks = media.video_chunks(Path("/v.mp4"))
        self.assertEqual([(c["heading"], c["media"] is None) for c in chunks], [("@ 0:00", False), ("@ 0:00", True)])


class SecondsAtTest(unittest.TestCase):
    def test_video_headings_parse_and_others_do_not(self):
        self.assertEqual(media.seconds_at("@ 1:30"), 90)
        self.assertEqual(media.seconds_at("@ 1:02:03"), 3723)
        self.assertEqual(media.seconds_at("@ " + media.clock(4321)), 4321)
        self.assertIsNone(media.seconds_at("p. 3"))
        self.assertIsNone(media.seconds_at("Intro > @ 1:30"))


if __name__ == "__main__":
    unittest.main()
