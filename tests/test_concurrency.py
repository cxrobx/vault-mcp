"""Several processes share one index: the indexer lock and the search cache.

Run: .venv/bin/python -m unittest discover -s tests
"""

import multiprocessing
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vault_mcp.embeddings import EMBED_DIM
from vault_mcp.indexer import _index_flock
from vault_mcp.store import Store


def _vec() -> np.ndarray:
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    v[0] = 1.0
    return v


def _hold(db: str, ready, release) -> None:
    with _index_flock(Path(db)) as acquired:
        assert acquired
        ready.set()
        release.wait(10)


class IndexLockTest(unittest.TestCase):
    def test_a_second_process_is_told_to_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "index.db")
            ready, release = multiprocessing.Event(), multiprocessing.Event()
            holder = multiprocessing.Process(target=_hold, args=(db, ready, release))
            holder.start()
            try:
                self.assertTrue(ready.wait(10))
                with _index_flock(Path(db)) as acquired:
                    self.assertFalse(acquired)
            finally:
                release.set()
                holder.join(10)
            with _index_flock(Path(db)) as acquired:
                self.assertTrue(acquired)

    def test_a_dead_holder_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "index.db")
            ready, release = multiprocessing.Event(), multiprocessing.Event()
            holder = multiprocessing.Process(target=_hold, args=(db, ready, release))
            holder.start()
            self.assertTrue(ready.wait(10))
            holder.kill()
            holder.join(10)
            with _index_flock(Path(db)) as acquired:
                self.assertTrue(acquired)


class CacheTest(unittest.TestCase):
    def test_search_sees_another_writers_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "index.db"
            reader, writer = Store(db), Store(db)  # two processes' worth of stores
            writer.replace_file("a.md", 0.0, 1, "h", [{"heading_path": "", "text": "alpha", "tags": "", "embedding": _vec()}])
            self.assertEqual([h["path"] for h in reader.search(_vec())], ["a.md"])
            writer.replace_file("b.md", 0.0, 1, "h", [{"heading_path": "", "text": "beta", "tags": "", "embedding": _vec()}])
            self.assertEqual({h["path"] for h in reader.search(_vec())}, {"a.md", "b.md"})
            writer.delete_files(["a.md"])
            self.assertEqual([h["path"] for h in reader.search(_vec())], ["b.md"])


if __name__ == "__main__":
    unittest.main()
