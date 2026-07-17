"""Ollama embeddings client.

nomic-embed-text is a task-prefixed model: documents must be embedded with
`search_document: ` and queries with `search_query: ` or retrieval quality
degrades measurably. Vectors are L2-normalized here so the store can score
with a plain dot product.
"""

import httpx
import numpy as np

from . import EMBED_MODEL, OLLAMA_URL

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
BATCH_SIZE = 32
EMBED_DIM = 768


class EmbeddingsUnavailable(RuntimeError):
    """Ollama is down or the embedding model is missing."""


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


class OllamaEmbedder:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = EMBED_MODEL):
        self.base_url = base_url
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=httpx.Timeout(120.0, connect=5.0))
        self._checked = False

    def check(self) -> None:
        """Verify Ollama is reachable and the model is pulled; raise a clear error if not."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingsUnavailable(
                f"Ollama is not reachable at {self.base_url} — start the Ollama app (or run `ollama serve`)."
            ) from exc
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            raise EmbeddingsUnavailable(
                f"Embedding model '{self.model}' is not pulled — run `ollama pull {self.model}`."
            )
        self._checked = True

    def _embed(self, inputs: list[str]) -> np.ndarray:
        try:
            resp = self._client.post("/api/embed", json={"model": self.model, "input": inputs})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self._checked = False
            self.check()  # raises the precise cause if Ollama/model is the problem
            raise EmbeddingsUnavailable(f"Ollama embed call failed: {exc}") from exc
        vecs = resp.json().get("embeddings", [])
        if len(vecs) != len(inputs):
            raise EmbeddingsUnavailable(
                f"Ollama returned {len(vecs)} embeddings for {len(inputs)} inputs."
            )
        return _normalize(np.asarray(vecs, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not self._checked:
            self.check()
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        batches = [
            self._embed([DOC_PREFIX + t for t in texts[i : i + BATCH_SIZE]])
            for i in range(0, len(texts), BATCH_SIZE)
        ]
        return np.vstack(batches)

    def embed_query(self, text: str) -> np.ndarray:
        if not self._checked:
            self.check()
        return self._embed([QUERY_PREFIX + text])[0]
