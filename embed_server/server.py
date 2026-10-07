"""Local embedding service: a sentence-transformers model behind Ollama's API.

vault-mcp talks to Ollama's `/api/tags` + `/api/embed`. Models Ollama doesn't
carry (EmbeddingGemma 2, at the time of writing) run here instead, behind the
same two endpoints, so the client only changes its URL and model name.

It exists as a daemon, not an in-process import, because the launcher (`j`)
starts cold on every keystroke-search: loading torch and the model there would
add seconds to a ~130 ms call. Here the model loads once and stays warm.

Input text arrives already prefixed — task prefixes are the client's job
(vault_mcp.embeddings), so this never applies a prompt of its own. Vectors are
returned L2-normalized, at the model's full dimension.

    embed_server/.venv/bin/python embed_server/server.py   # LaunchAgent com.cx.vault-embed
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("VAULT_EMBED_MODEL_ID", "google/embeddinggemma-2")
# The name clients ask for — what vault-mcp's config `model` must say.
MODEL_NAME = os.environ.get("VAULT_EMBED_MODEL_NAME", MODEL_ID.split("/")[-1])
HOST = os.environ.get("VAULT_EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("VAULT_EMBED_PORT", "11436"))

# Text only: vision and audio encoders are separate modules (270M vs 740M) and
# nothing indexes images or audio yet.
model = SentenceTransformer(
    MODEL_ID,
    device="mps" if torch.backends.mps.is_available() else "cpu",
    config_kwargs={"vision_config": None, "audio_config": None},
    # float32, not float16: the model card warns fp16 overflows to NaN.
    model_kwargs={"torch_dtype": torch.float32},
)
# MPS is not safe to drive from several threads at once.
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": MODEL_NAME, "model": MODEL_ID}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/embed":
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if req.get("model") != MODEL_NAME:
                return self._send(404, {"error": f"model '{req.get('model')}' not found; this server has '{MODEL_NAME}'"})
            inputs = req.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            with _lock:
                vecs = model.encode(inputs, batch_size=16, normalize_embeddings=True)
            self._send(200, {"model": MODEL_NAME, "embeddings": vecs.tolist()})
        except Exception as exc:  # a bad request must not kill the daemon
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt, *args):  # one line per request is noise at this rate
        pass


if __name__ == "__main__":
    print(f"vault-embed: {MODEL_ID} as '{MODEL_NAME}' on {HOST}:{PORT}", file=sys.stderr, flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
