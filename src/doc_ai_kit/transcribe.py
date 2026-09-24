"""Reading pages that have no usable text layer, by having Claude transcribe an image of them.

A scanned page has no text for PyMuPDF to read, so the extracted text of that page is empty
and every question whose answer is on it scores as a miss that no prompt can fix. Each such
page is rendered to an image and Claude transcribes it into the text cache, in place. The
rest of the pipeline never knows: it reads cached text either way.

The decision is made page by page. Deciding per document, on the average, lets a report
whose typed narrative carries five scanned pages of accounts through with those pages blank
-- and unflagged, because the average looked healthy.

Transcriptions are cached per page, keyed on the PDF's contents, the model and the prompt,
so re-extracting never pays twice for the same page.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from doc_ai_kit.config import TranscriptionConfig

logger = logging.getLogger(__name__)

# bumped whenever the prompt changes, so cached transcriptions made under the old one are redone
PROMPT_VERSION = 1

TRANSCRIPTION_PROMPT = """\
This is one page of a document. Transcribe all of its text exactly as written, in reading \
order, in its original language.

- Do not correct spelling, summarize, translate, or add anything that is not on the page.
- Tables: one row per line, cells separated by " | ", including the header row. Keep every \
number in the row it belongs to.
- Handwriting: transcribe it if you can read it.
- A word you cannot read: write [illegible].
- Checkboxes: [x] if ticked, [ ] if not.
- Leave out purely decorative elements such as logos and page borders.

Reply with the transcription only: no preamble and no commentary. If the page has no text, \
reply with nothing."""

PageStatus = Literal["ok", "truncated", "failed"]


class TranscriptionError(RuntimeError):
    """Raised when transcription is needed but cannot run at all."""


@dataclass(frozen=True)
class PageTranscription:
    """One page as Claude read it."""

    page: int
    text: str
    status: PageStatus
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    detail: str = ""
    cached: bool = False


def pdf_fingerprint(pdf_path: Path) -> str:
    """Return a hash of the PDF's bytes, so a replaced file is never matched to an old transcription."""
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_page(pdf_path: Path, page_index: int, max_image_px: int) -> bytes:
    """Render one page to PNG, with its longer edge at max_image_px.

    :param pdf_path: The PDF
    :param page_index: Zero-based page number
    :param max_image_px: The longer edge of the image, in pixels
    :returns: The PNG bytes
    """
    import pymupdf

    with pymupdf.open(pdf_path) as document:  # type: ignore[no-untyped-call]
        page = document[page_index]
        longer = max(page.rect.width, page.rect.height) or 1.0
        zoom = max_image_px / longer
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)  # type: ignore[no-untyped-call]
        return bytes(pixmap.tobytes("png"))


class Transcriber:
    """Transcribes pages with Claude, caching each page on disk."""

    def __init__(self, config: TranscriptionConfig, cache_dir: Path, client: Any = None) -> None:
        """Set up a transcriber.

        :param config: The project's transcription settings
        :param cache_dir: Where per-page transcriptions are kept
        :param client: An Anthropic client; created on first use if not given, so a corpus
            that never needs transcription never needs credentials
        """
        self.config = config
        self.cache_dir = cache_dir
        self._client = client

    @property
    def model_name(self) -> str:
        """Return the model id as the Anthropic API expects it, without the provider prefix."""
        if self.config.model is None:
            raise TranscriptionError("extraction.transcription.model is not set")
        return self.config.model.removeprefix("anthropic/")

    @property
    def client(self) -> Any:
        """Return the Anthropic client, creating it on first use."""
        if self._client is None:
            import anthropic

            # the SDK retries rate limits and server errors itself, with backoff
            self._client = anthropic.Anthropic(max_retries=self.config.max_retries)
        return self._client

    def transcribe(self, pdf_path: Path, doc_id: str, pages: Sequence[int]) -> dict[int, PageTranscription]:
        """Transcribe the given pages of one PDF, reusing cached pages.

        :param pdf_path: The PDF
        :param doc_id: The document id, naming the cache directory
        :param pages: Zero-based page numbers to transcribe
        :returns: Page number to its transcription, for every page asked for
        """
        if not pages:
            return {}
        fingerprint = pdf_fingerprint(pdf_path)
        results: dict[int, PageTranscription] = {}
        pending: list[int] = []
        for page in pages:
            cached = self._load(doc_id, page, fingerprint)
            if cached is not None:
                results[page] = cached
            else:
                pending.append(page)
        if pending:
            logger.info("%s: transcribing %d page(s) with %s", doc_id, len(pending), self.model_name)
            with ThreadPoolExecutor(max_workers=self.config.num_threads) as pool:
                futures = {
                    page: pool.submit(contextvars.copy_context().run, self._transcribe_page, pdf_path, page)
                    for page in pending
                }
                for page, future in futures.items():
                    result = future.result()
                    results[page] = result
                    # a failure is not cached, so the next run tries the page again
                    if result.status != "failed":
                        self._store(doc_id, result, fingerprint)
        return dict(sorted(results.items()))

    def _transcribe_page(self, pdf_path: Path, page: int) -> PageTranscription:
        """Send one page image to Claude and read back its transcription."""
        model = self.model_name
        try:
            image = base64.standard_b64encode(render_page(pdf_path, page, self.config.max_image_px)).decode("ascii")
            response = self.client.messages.create(
                model=model,
                max_tokens=self.config.max_tokens,
                # copying text out of an image needs no reasoning, and thinking is billed as output
                thinking={"type": "disabled"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}},
                            {"type": "text", "text": TRANSCRIPTION_PROMPT},
                        ],
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a failed page, never swallowed
            logger.warning("page %d of %s could not be transcribed: %s", page + 1, pdf_path.name, exc)
            return PageTranscription(
                page=page, text="", status="failed", model=model, detail=f"{type(exc).__name__}: {exc}"
            )

        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        stop = getattr(response, "stop_reason", None)
        status: PageStatus = "ok"
        detail = ""
        if stop == "max_tokens":
            # the page ran past the output limit: what came back is real but incomplete
            status, detail = "truncated", "hit max_tokens"
        elif stop not in ("end_turn", "stop_sequence", None):
            status, detail, text = "failed", f"stop_reason {stop}", ""
        return PageTranscription(
            page=page,
            text=text,
            status=status,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            detail=detail,
        )

    def _cache_path(self, doc_id: str, page: int) -> Path:
        return self.cache_dir / doc_id / f"page-{page + 1:04d}.json"

    def _load(self, doc_id: str, page: int, fingerprint: str) -> PageTranscription | None:
        path = self._cache_path(doc_id, page)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload.get("pdf_sha256"), payload.get("prompt_version"), payload.get("transcription", {}).get("model"))
        if key != (fingerprint, PROMPT_VERSION, self.model_name):
            return None
        stored = payload["transcription"]
        return PageTranscription(**{**stored, "cached": True, "input_tokens": 0, "output_tokens": 0})

    def _store(self, doc_id: str, result: PageTranscription, fingerprint: str) -> None:
        path = self._cache_path(doc_id, result.page)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pdf_sha256": fingerprint,
            "prompt_version": PROMPT_VERSION,
            "transcription": {key: value for key, value in asdict(result).items() if key != "cached"},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
