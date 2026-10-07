"""Embeddings client for Ollama's API (Ollama itself, or embed_server/).

Both supported models are task-prefixed, each in its own format, and a wrong
or missing prefix degrades retrieval silently rather than failing. So the
prefixes live in one table keyed by model name, and the document text is built
here (`format_document`) rather than by the indexer:

* nomic-embed-text — `search_document: ` / `search_query: `.
* embeddinggemma-2 — `title: {title} | text: {body}` / `task: search result |
  query: `, per its model card. Served by embed_server/ (not on Ollama).

Vectors are L2-normalized here so the store can score with a plain dot product.
"""

import httpx
import numpy as np

from . import EMBED_MODEL, OLLAMA_URL

# model -> (document template, query prefix, multimodal). Templates get
# title/heading/text. Multimodal models also embed media items (embed_items).
PROFILES = {
    "nomic-embed-text": ("search_document: {title}{sep}{heading}\n{text}", "search_query: ", False),
    "embeddinggemma-2": ("title: {title}{sep}{heading} | text: {text}", "task: search result | query: ", True),
}
MEDIA_BATCH = 4
BATCH_SIZE = 32
EMBED_DIM = 768


class EmbeddingsUnavailable(RuntimeError):
    """Ollama is down or the embedding model is missing."""


class MediaRejected(RuntimeError):
    """The server refused one media item (unreadable, outside the roots). Not an outage."""


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
        try:
            self._doc_template, self._query_prefix, self.multimodal = PROFILES[model.split(":")[0]]
        except KeyError:
            raise EmbeddingsUnavailable(
                f"No prompt profile for embedding model '{model}' — add one to PROFILES in embeddings.py."
            ) from None

    def format_document(self, title: str, heading: str, text: str) -> str:
        """The exact string a chunk is embedded as, prefix included."""
        return self._doc_template.format(
            title=title, sep=" > " if heading else "", heading=heading, text=text
        )

    def check(self) -> None:
        """Verify Ollama is reachable and the model is pulled; raise a clear error if not."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingsUnavailable(
                f"Embedding server not reachable at {self.base_url} — start the Ollama app, or for "
                f"{self.model} the com.cx.vault-embed LaunchAgent (embed_server/)."
            ) from exc
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            raise EmbeddingsUnavailable(
                f"Embedding model '{self.model}' is not pulled — run `ollama pull {self.model}`."
            )
        self._checked = True

    def _embed(self, inputs: list) -> np.ndarray:
        try:
            resp = self._client.post("/api/embed", json={"model": self.model, "input": inputs})
            if resp.status_code == 400:
                raise MediaRejected(resp.json().get("error", resp.text))
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
        """Embed texts already built by format_document()."""
        if not self._checked:
            self.check()
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        batches = [
            self._embed(texts[i : i + BATCH_SIZE])
            for i in range(0, len(texts), BATCH_SIZE)
        ]
        return np.vstack(batches)

    def embed_items(self, items: list[dict]) -> np.ndarray:
        """Embed media chunks: {"text": format_document(...) with placeholders, + media specs}."""
        if not self.multimodal:
            raise EmbeddingsUnavailable(f"{self.model} can't embed media; set `media = []` or use a multimodal model.")
        if not self._checked:
            self.check()
        return np.vstack([self._embed(items[i : i + MEDIA_BATCH]) for i in range(0, len(items), MEDIA_BATCH)])

    def embed_query(self, text: str) -> np.ndarray:
        if not self._checked:
            self.check()
        return self._embed([self._query_prefix + text])[0]
