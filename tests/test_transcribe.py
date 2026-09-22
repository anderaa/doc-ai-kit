"""Transcribing pages that have no text layer, with a fake Claude client.

The PDFs here are built with a real scanned page: text rendered to an image and placed on a
page with no text layer, which is what a scanner produces.
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from doc_harness.config import ExtractionConfig, TranscriptionConfig
from doc_harness.extract import extract_corpus, extract_document
from doc_harness.transcribe import Transcriber

TYPED = "\n".join(f"Trustees' report, paragraph {i}: the charity continued its work this year." for i in range(12))
SCANNED = "Total income | 1,204,500\nTotal expenditure | 998,210"


def _scanned_page(document: Any, text: str) -> None:
    """Add a page that carries text only as an image, as a scanner would produce it."""
    import pymupdf

    source = pymupdf.open()  # type: ignore[no-untyped-call]
    page = source.new_page()
    page.insert_text((72, 100), text, fontsize=14)
    pixmap = page.get_pixmap(dpi=100)
    scanned = document.new_page()
    scanned.insert_image(scanned.rect, pixmap=pixmap)


def make_mixed_pdf(path: Path, layout: str) -> Path:
    """Write a PDF whose pages are typed (t) or scanned (s), in the given order."""
    import pymupdf

    document = pymupdf.open()  # type: ignore[no-untyped-call]
    for kind in layout:
        if kind == "t":
            document.new_page().insert_text((72, 72), TYPED, fontsize=9)
        else:
            _scanned_page(document, SCANNED)
    document.save(path)
    return path


class FakeClient:
    """Stands in for anthropic.Anthropic: records each request and answers from a script."""

    def __init__(self, replies: list[tuple[str, str]] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        with self._lock:
            self.calls.append(kwargs)
            text, stop = self.replies.pop(0) if self.replies else (SCANNED, "end_turn")
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=stop,
            usage=SimpleNamespace(input_tokens=1500, output_tokens=40),
        )


def _config(**overrides: Any) -> ExtractionConfig:
    settings = {"model": "anthropic/claude-sonnet-5", "num_threads": 1, **overrides}
    return ExtractionConfig(transcription=TranscriptionConfig(**settings))


def _transcriber(tmp_path: Path, config: ExtractionConfig, client: FakeClient) -> Transcriber:
    return Transcriber(config.transcription, tmp_path / "transcripts", client=client)


def test_only_the_scanned_page_is_sent_and_it_lands_in_place(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "report.pdf", "tst")
    config = _config()
    client = FakeClient()
    document = extract_document(pdf, config, transcriber=_transcriber(tmp_path, config, client))

    assert len(client.calls) == 1, "typed pages must not be sent"
    text = document.text
    # between the end of the first typed page and the start of the second
    assert text.index("paragraph 11") < text.index(SCANNED) < text.rindex("paragraph 0")
    assert (document.thin_pages, document.transcribed_pages, document.unread_pages) == (1, 1, 0)
    assert document.transcription_model == "claude-sonnet-5"
    assert document.transcription_tokens == (1500, 40)


def test_the_request_is_an_image_with_thinking_off(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "scan.pdf", "s")
    config = _config(max_image_px=1000)
    client = FakeClient()
    extract_document(pdf, config, transcriber=_transcriber(tmp_path, config, client))

    request = client.calls[0]
    assert request["model"] == "claude-sonnet-5"
    assert request["thinking"] == {"type": "disabled"}
    assert "temperature" not in request
    image = request["messages"][0]["content"][0]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png"

    import pymupdf

    png = pymupdf.Pixmap(base64.b64decode(image["source"]["data"]))  # type: ignore[no-untyped-call]
    assert max(png.width, png.height) == 1000


def test_a_document_that_averages_fine_still_gets_its_scanned_page_read(tmp_path: Path) -> None:
    """The case the old whole-document average missed: plenty of text overall, one blank page."""
    pdf = make_mixed_pdf(tmp_path / "mostly_typed.pdf", "tttts")
    config = _config()
    client = FakeClient()
    document = extract_document(pdf, config, transcriber=_transcriber(tmp_path, config, client))
    assert document.chars_per_page > 100
    assert document.transcribed_pages == 1 and SCANNED in document.text


def test_transcriptions_are_cached_per_page(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "scan.pdf", "ss")
    config = _config()
    client = FakeClient()
    transcriber = _transcriber(tmp_path, config, client)
    extract_document(pdf, config, transcriber=transcriber)
    assert len(client.calls) == 2

    again = extract_document(pdf, config, transcriber=transcriber)
    assert len(client.calls) == 2, "a cached page was paid for twice"
    assert again.transcription_tokens == (0, 0)
    assert again.text.count(SCANNED) == 2

    # a different model is a different transcription
    other = _config(model="anthropic/claude-opus-5")
    extract_document(pdf, other, transcriber=_transcriber(tmp_path, other, client))
    assert len(client.calls) == 4


def test_a_replaced_pdf_is_transcribed_afresh(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "scan.pdf", "s")
    config = _config()
    client = FakeClient()
    transcriber = _transcriber(tmp_path, config, client)
    extract_document(pdf, config, transcriber=transcriber)
    make_mixed_pdf(pdf, "ts")
    extract_document(pdf, config, transcriber=transcriber)
    assert len(client.calls) == 2


def test_failed_and_cut_off_pages_are_counted_as_unread(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "scan.pdf", "sss")
    config = _config()
    client = FakeClient([(SCANNED, "end_turn"), ("Total inc", "max_tokens"), ("", "refusal")])
    document = extract_document(pdf, config, transcriber=_transcriber(tmp_path, config, client))
    assert (document.transcribed_pages, document.unread_pages) == (2, 2)
    assert "Total inc" in document.text


def test_a_failed_page_is_not_cached_so_the_next_run_retries_it(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "scan.pdf", "s")
    config = _config()
    client = FakeClient([("", "refusal")])
    transcriber = _transcriber(tmp_path, config, client)
    assert extract_document(pdf, config, transcriber=transcriber).unread_pages == 1
    assert extract_document(pdf, config, transcriber=transcriber).unread_pages == 0
    assert len(client.calls) == 2


def test_an_api_error_is_a_failed_page_not_a_crash(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "scan.pdf", "ts")
    config = _config()
    client = FakeClient()

    def boom(**kwargs: Any) -> Any:
        raise ConnectionError("network down")

    client.messages = SimpleNamespace(create=boom)
    document = extract_document(pdf, config, transcriber=_transcriber(tmp_path, config, client))
    assert document.unread_pages == 1 and "Trustees' report" in document.text


def test_all_pages_mode_reads_typed_pages_too(tmp_path: Path) -> None:
    """For PDFs whose text layer exists but is garbage."""
    pdf = make_mixed_pdf(tmp_path / "report.pdf", "ts")
    config = _config(mode="all_pages")
    client = FakeClient([("Clean page one", "end_turn"), (SCANNED, "end_turn")])
    document = extract_document(pdf, config, transcriber=_transcriber(tmp_path, config, client))
    assert len(client.calls) == 2
    assert document.text == f"Clean page one\n\n{SCANNED}"


def test_off_mode_sends_nothing(tmp_path: Path) -> None:
    pdf = make_mixed_pdf(tmp_path / "scan.pdf", "s")
    config = _config(mode="off")
    client = FakeClient()
    document = extract_document(pdf, config, transcriber=_transcriber(tmp_path, config, client))
    assert not client.calls and document.unread_pages == 1


def test_setting_a_model_later_redoes_only_the_flagged_documents(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    make_mixed_pdf(pdf_dir / "typed.pdf", "tt")
    make_mixed_pdf(pdf_dir / "scanned.pdf", "ts")
    text_dir = tmp_path / "text"
    manifest = tmp_path / "manifest.csv"

    first = {d.doc_id: d for d in extract_corpus(pdf_dir, text_dir, manifest, ExtractionConfig())}
    assert first["scanned"].unread_pages == 1 and first["typed"].unread_pages == 0

    marker = "CACHED TYPED TEXT"
    (text_dir / "typed.md").write_text(marker, encoding="utf-8")
    client = FakeClient()
    second = {d.doc_id: d for d in extract_corpus(pdf_dir, text_dir, manifest, _config(), client=client)}
    assert len(client.calls) == 1
    assert second["scanned"].unread_pages == 0 and SCANNED in (text_dir / "scanned.md").read_text()
    assert second["typed"].text == marker, "a document with nothing unread was re-extracted"
    assert (tmp_path / "transcripts" / "scanned" / "page-0002.json").exists()


def test_no_client_is_built_when_nothing_needs_transcribing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corpus of clean PDFs must not need credentials."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    make_mixed_pdf(pdf_dir / "typed.pdf", "tt")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    documents = extract_corpus(pdf_dir, tmp_path / "text", tmp_path / "m.csv", _config())
    assert documents[0].transcribed_pages == 0
