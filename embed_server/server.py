"""Local embedding service: a sentence-transformers model behind Ollama's API.

vault-mcp talks to Ollama's `/api/tags` + `/api/embed`. Models Ollama doesn't
carry (EmbeddingGemma 2, at the time of writing) run here instead, behind the
same two endpoints, so the client only changes its URL and model name.

It exists as a daemon, not an in-process import, because the launcher (`j`)
starts cold on every keystroke-search: loading torch and the model there would
add seconds to a ~130 ms call. Here the model loads on the first request and
stays warm while it's in use; after VAULT_EMBED_IDLE_S (default 600 s) without a
request the process exits and launchd starts a fresh, empty one. Held warm, float32 plus the MPS allocator's cache
sat at ~9.6 GB around the clock (measured 2026-10-14) for a search used a few
times a week; dropped, the process is a few hundred MB, and the first search
after a quiet spell pays a reload of a few seconds.

Text input arrives already prefixed — task prefixes are the client's job
(vault_mcp.embeddings), so this never applies a prompt of its own. Vectors are
returned L2-normalized, at the model's full dimension.

**Media.** An `input` item may be an object instead of a string:

    {"text": "title: x > p. 3 | text: ... <|image|>", "image": {"pdf": "/abs.pdf", "page": 3}}
    {"text": "... <|image|>", "image": {"path": "/abs.png"}}
    {"text": "... <|video|> <|audio|>",
     "video": {"path": "/abs.mp4", "start": 30, "duration": 30},
     "audio": {"path": "/abs.mp4", "start": 30, "duration": 30}}

The placeholders in `text` mark where each medium goes, as the model card
describes. Media is decoded HERE (pdftoppm, ffmpeg) so the client never ships
pixels over HTTP, and only files under the roots vault-mcp's config indexes
are readable — this is a localhost service, but it still shouldn't be a way to
read arbitrary files. Video is passed as 1 fps frames rather than a path, which
keeps the clip window exact and avoids the torchcodec dependency.

**Priority.** The GPU takes one request at a time. A search (text only) waiting
for it goes ahead of the next media item, so a background reindex of a long
video holds a search up by one item (~4 s for a 30 s window) at most.

    embed_server/.venv/bin/python embed_server/server.py   # LaunchAgent com.cx.vault-embed
"""

import io
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("VAULT_EMBED_MODEL_ID", "google/embeddinggemma-2")
# The name clients ask for — what vault-mcp's config `model` must say.
MODEL_NAME = os.environ.get("VAULT_EMBED_MODEL_NAME", MODEL_ID.split("/")[-1])
HOST = os.environ.get("VAULT_EMBED_HOST", "127.0.0.1")
PORT = int(os.environ.get("VAULT_EMBED_PORT", "11436"))
CONFIGS = [Path(p).expanduser() for p in os.environ.get(
    "VAULT_EMBED_CONFIGS", "~/.config/vault-mcp/config.toml").split(os.pathsep)]
# Frame size for video. 448 wide keeps on-screen text legible to the vision
# encoder without decoding full-HD frames we'd only shrink again.
FRAME_W, FRAME_H = 448, 252
PDF_DPI = 110
MAX_MEDIA_SECONDS = 60  # one window; the client segments longer media
IDLE_S = float(os.environ.get("VAULT_EMBED_IDLE_S", "600"))


def _roots() -> list[Path]:
    """Folders media may be read from: every vault and mount the configs index."""
    roots = []
    for cfg in CONFIGS:
        try:
            data = tomllib.loads(cfg.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        roots.append(data.get("vault", "~/Documents/CX"))
        for spec in data.get("mounts", []) + data.get("note_mounts", []):
            roots.append(spec.partition("=")[2])
    return [q for r in roots if r for q in {Path(os.path.normpath(Path(r).expanduser())), Path(r).expanduser().resolve()}]


ROOTS = _roots()

_model: SentenceTransformer | None = None
_last_used = 0.0


def _get_model() -> SentenceTransformer:
    """The model, loaded on first use. Only called inside the GPU gate."""
    global _model
    if _model is None:
        t = time.monotonic()
        _model = SentenceTransformer(
            MODEL_ID,
            device="mps" if torch.backends.mps.is_available() else "cpu",
            # float32, not float16: the model card warns fp16 overflows to NaN.
            model_kwargs={"torch_dtype": torch.float32},
        )
        print(f"vault-embed: model loaded in {time.monotonic() - t:.1f}s", file=sys.stderr, flush=True)
    return _model


_gate = threading.Condition()
_busy = False
_queries_waiting = 0


def _exit_when_idle() -> None:
    """After IDLE_S without a request, exit; launchd (KeepAlive) starts a fresh one.

    Exiting rather than dropping the model in place: freeing it and emptying the
    MPS cache still left the process at ~1.1 GB, against ~330 MB for a fresh
    start with nothing loaded (measured 2026-10-14).
    """
    while True:
        time.sleep(min(60.0, IDLE_S / 4))
        with _gate:
            if _model is None or _busy or _queries_waiting or time.monotonic() - _last_used < IDLE_S:
                continue
            print("vault-embed: idle, exiting to free the model", file=sys.stderr, flush=True)
            os._exit(0)


def _run(fn, is_query: bool):
    """Serialize GPU work; waiting queries go before waiting media."""
    global _busy, _queries_waiting, _last_used
    with _gate:
        if is_query:
            _queries_waiting += 1
        try:
            _gate.wait_for(lambda: not _busy and (is_query or _queries_waiting == 0))
        finally:
            if is_query:
                _queries_waiting -= 1
        _busy = True
    try:
        return fn(_get_model())
    finally:
        with _gate:
            _last_used = time.monotonic()
            _busy = False
            _gate.notify_all()


def _readable(path: str) -> str:
    # The path as named OR as resolved may sit under a root: the indexer follows
    # symlinks inside the vault on purpose (the XRF folder links into the
    # cxventures repo). normpath, not resolve, for the named form so ".." can't
    # walk out of a root.
    named = Path(os.path.normpath(Path(path).expanduser()))
    real = named.resolve()
    if not any(p.is_relative_to(r) for r in ROOTS for p in (named, real)):
        raise PermissionError(f"{path} is outside the indexed folders")
    if not real.is_file():
        raise FileNotFoundError(path)
    return str(real)


def _window(spec: dict) -> tuple[str, float, float]:
    dur = float(spec.get("duration", MAX_MEDIA_SECONDS))
    if not 0 < dur <= MAX_MEDIA_SECONDS:
        raise ValueError(f"duration must be in (0, {MAX_MEDIA_SECONDS}]")
    return _readable(spec["path"]), max(0.0, float(spec.get("start", 0))), dur


def _ffmpeg(args: list[str]) -> bytes:
    return subprocess.run(["ffmpeg", "-v", "error", *args, "-"], capture_output=True, check=True).stdout


def _image(spec: dict) -> Image.Image:
    if "pdf" in spec:
        page = int(spec["page"])
        png = subprocess.run(
            ["pdftoppm", "-png", "-r", str(PDF_DPI), "-f", str(page), "-l", str(page), "-singlefile",
             _readable(spec["pdf"])],
            capture_output=True, check=True,
        ).stdout
        return Image.open(io.BytesIO(png)).convert("RGB")
    return Image.open(_readable(spec["path"])).convert("RGB")


def _video(spec: dict) -> np.ndarray:
    path, start, dur = _window(spec)
    raw = _ffmpeg(["-ss", str(start), "-t", str(dur), "-i", path,
                   "-vf", f"fps=1,scale={FRAME_W}:{FRAME_H}", "-f", "rawvideo", "-pix_fmt", "rgb24"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, FRAME_H, FRAME_W, 3)
    if not len(frames):
        raise ValueError("no frames in that window")
    return frames


def _audio(spec: dict) -> dict:
    path, start, dur = _window(spec)
    raw = _ffmpeg(["-ss", str(start), "-t", str(dur), "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le"])
    return {"array": np.frombuffer(raw, np.float32).copy(), "sampling_rate": 16000}


def _decode(item: dict) -> dict:
    out = {"text": item["text"]}
    if "image" in item:
        specs = item["image"] if isinstance(item["image"], list) else [item["image"]]
        out["image"] = [_image(s) for s in specs]
    if "video" in item:
        out["video"] = _video(item["video"])
    if "audio" in item:
        out["audio"] = _audio(item["audio"])
    return out


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
            self._send(200, {"models": [{"name": MODEL_NAME, "model": MODEL_ID, "multimodal": True}]})
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
            if all(isinstance(x, str) for x in inputs):
                vecs = _run(lambda m: m.encode(inputs, batch_size=16, normalize_embeddings=True),
                            is_query=len(inputs) == 1)
            else:
                # Media is decoded outside the GPU gate (ffmpeg/pdftoppm are CPU),
                # then embedded one item per gate pass so a query can cut in.
                rows = []
                for x in inputs:
                    item = x if isinstance(x, str) else _decode(x)
                    rows.append(_run(lambda m: m.encode(item, normalize_embeddings=True), is_query=False))
                vecs = np.vstack(rows)
            self._send(200, {"model": MODEL_NAME, "embeddings": np.asarray(vecs).tolist()})
        except (PermissionError, FileNotFoundError, ValueError, KeyError) as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
        except subprocess.CalledProcessError as exc:
            self._send(400, {"error": f"decode failed: {exc.stderr.decode(errors='replace')[:300]}"})
        except Exception as exc:  # a bad request must not kill the daemon
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):  # one line per request is noise at this rate
        pass


if __name__ == "__main__":
    print(f"vault-embed: {MODEL_ID} as '{MODEL_NAME}' on {HOST}:{PORT}; media roots {[str(r) for r in ROOTS]}",
          file=sys.stderr, flush=True)
    threading.Thread(target=_exit_when_idle, daemon=True).start()
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
