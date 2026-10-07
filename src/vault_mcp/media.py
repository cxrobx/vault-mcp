"""PDFs, images and videos as index chunks.

Each medium becomes the same thing a note does — chunks with a heading and a
text body — plus, where pixels or sound carry meaning the text doesn't, a
`media` spec the embed server decodes (embed_server/server.py). The text half
of every chunk is what FTS indexes and what a search result shows, so a media
chunk is never textless when there is any text to be had:

* **PDF** — `pdftotext` per page. A page with real text is indexed as text,
  exactly like a note section ("p. 3"). A page with little text (a slide, a
  scan, a chart) is embedded from its rendered image too, with whatever text it
  has alongside.
* **Image** — the picture itself, plus `tesseract` OCR. Most images here are
  screenshots, where the words on screen are the best handle a search has.
* **Video** — 30-second windows ("@ 1:30"), each embedded from its frames
  (1 fps), its audio, and the whisper.cpp transcript of that window — and the
  transcript again as a plain text chunk, so narration is findable as text. 30 s, not
  the model's ~58 s frame budget, so a search waiting behind a window being
  embedded waits ~4 s at most.

Everything runs locally (poppler, tesseract, whisper.cpp, ffmpeg); nothing
leaves the machine.
"""

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from . import WHISPER_MODEL

SUFFIXES = {
    "pdf": {".pdf"},
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif"},
    "video": {".mp4", ".mov", ".m4v", ".webm"},
}
# A page with fewer characters than this is "visual" and also gets its image.
PDF_TEXT_MIN = 300
MAX_PDF_PAGES = 300
# Icons, spacers and tracking pixels aren't worth a vision pass.
MIN_IMAGE_BYTES = 8_000
MAX_IMAGE_BYTES = 30_000_000
OCR_CHARS = 2000
WINDOW_S = 30
MAX_VIDEO_S = 3 * 3600
TRANSCRIPT_CHARS = 2000
MIN_SPOKEN_CHARS = 80


class MediaError(RuntimeError):
    """This one file couldn't be read; the reindex goes on without it."""


def kind_of(path: Path) -> str | None:
    suffix = path.suffix.lower()
    return next((k for k, s in SUFFIXES.items() if suffix in s), None)


def worth_indexing(path: Path) -> bool:
    """Size gate, applied in the walk so a skipped file never gets an index row.

    Icons and spacers have nothing to find, and a row with no chunks still
    shows up in the launcher's by-name matching.
    """
    if kind_of(path) != "image":
        return True
    try:
        return MIN_IMAGE_BYTES <= path.stat().st_size <= MAX_IMAGE_BYTES
    except OSError:
        return False


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _run(args: list[str], timeout: int = 300) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout,
                              errors="replace").stdout
    except FileNotFoundError as exc:
        raise MediaError(f"{args[0]} is not installed") from exc
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"{args[0]} failed: {exc.stderr.strip()[:200]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{args[0]} timed out") from exc


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def seconds_at(heading: str) -> int | None:
    """The offset a video heading names ("@ 1:30" -> 90), else None."""
    m = re.fullmatch(r"@ (?:(\d+):)?(\d+):(\d{2})", heading.strip())
    if not m:
        return None
    h, mins, secs = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mins * 60 + secs


# Each chunk: {"heading": str, "text": str, "media": dict | None}. "media" holds
# the image/video/audio specs the embed server takes; None = embed as text.


def pdf_chunks(path: Path, split) -> list[dict]:
    pages = _run(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"]).split("\f")
    chunks = []
    for n, raw in enumerate(pages[:MAX_PDF_PAGES], start=1):
        text = _tidy(raw)
        heading = f"p. {n}"
        if len(text) >= PDF_TEXT_MIN:
            chunks += [{"heading": heading, "text": piece, "media": None} for piece in split(text)]
        elif n < len(pages) or text:  # pdftotext ends on a trailing \f: skip that empty tail
            chunks.append({"heading": heading, "text": text[:OCR_CHARS],
                           "media": {"image": {"pdf": str(path), "page": n}}})
    return chunks


def image_chunks(path: Path) -> list[dict]:
    try:
        ocr = _tidy(_run(["tesseract", str(path), "-", "--psm", "3"], timeout=60))
    except MediaError:
        ocr = ""  # an unreadable-to-OCR image is still worth its pixels
    return [{"heading": "", "text": ocr[:OCR_CHARS], "media": {"image": {"path": str(path)}}}]


def _duration(path: Path) -> float:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)])
    try:
        return float(out.strip())
    except ValueError as exc:
        raise MediaError("no duration (not a playable video?)") from exc


def _has_audio(path: Path) -> bool:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                "-of", "csv=p=0", str(path)])
    return bool(out.strip())


def transcript(path: Path) -> list[tuple[float, float, str]]:
    """[(start_s, end_s, text)] from whisper.cpp; [] when there's no model or no speech."""
    if not WHISPER_MODEL.is_file():
        return []
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        _run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", str(wav)], timeout=600)
        _run(["whisper-cli", "-m", str(WHISPER_MODEL), "-f", str(wav), "-oj", "-of", str(Path(tmp) / "t"), "-np"],
             timeout=3600)
        data = json.loads((Path(tmp) / "t.json").read_text())
    out = []
    for seg in data.get("transcription", []):
        off = seg.get("offsets", {})
        text = seg.get("text", "").strip()
        if text and not text.startswith("["):  # [BLANK_AUDIO], [Music]
            out.append((off.get("from", 0) / 1000, off.get("to", 0) / 1000, text))
    return out


def video_chunks(path: Path) -> list[dict]:
    duration = min(_duration(path), MAX_VIDEO_S)
    audio = _has_audio(path)
    words = transcript(path) if audio else []
    chunks = []
    start = 0.0
    while start < duration - 0.5:
        dur = min(WINDOW_S, duration - start)
        said = " ".join(t for s, e, t in words if s < start + dur and e > start)
        window = {"path": str(path), "start": start, "duration": dur}
        media = {"video": window, **({"audio": window} if audio else {})}
        heading = f"@ {clock(start)}"
        chunks.append({"heading": heading, "text": said[:TRANSCRIPT_CHARS], "media": media})
        # What was said, again on its own as text. In the window's own vector the
        # frames dominate, so a query about the narration ("which template do I
        # pick") ranked the right window ~40th; as plain text it competes like a
        # note section does. The media chunk still carries what's on screen.
        if len(said) >= MIN_SPOKEN_CHARS:
            chunks.append({"heading": heading, "text": said[:TRANSCRIPT_CHARS], "media": None})
        start += WINDOW_S
    return chunks


def chunks_for(path: Path, kind: str, split) -> list[dict]:
    if kind == "pdf":
        return pdf_chunks(path, split)
    if kind == "image":
        return image_chunks(path)
    if kind == "video":
        return video_chunks(path)
    raise ValueError(kind)


def placeholders(media: dict | None) -> str:
    """The model's markers for where each medium sits in the text."""
    if not media:
        return ""
    marks = []
    if "image" in media:
        marks.append("<|image|>")
    if "video" in media:
        marks.append("<|video|>")
    if "audio" in media:
        marks.append("<|audio|>")
    return " ".join(marks)
