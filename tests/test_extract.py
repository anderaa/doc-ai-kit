"""Extraction tests over PDFs generated on the fly."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from doc_harness.config import ExtractionConfig, TruncationConfig
from doc_harness.extract import extract_corpus, extract_document, load_texts, truncate


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a small text-layer PDF with one block of text per page."""
    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    for page_text in pages:
        text_object = pdf.beginText(72, 720)
        for line in page_text.splitlines() or [""]:
            text_object.textLine(line)
        pdf.drawText(text_object)
        pdf.showPage()
    pdf.save()
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    body = "\n".join(f"This agreement is governed by the laws of California. Line {i}." for i in range(30))
    make_pdf(pdf_dir / "doc_a.pdf", [body, body])
    make_pdf(pdf_dir / "doc_b.pdf", [body])
    # a scanned-looking document: a page with almost no text layer
    make_pdf(pdf_dir / "doc_scan.pdf", ["x"])
    return pdf_dir


def test_extracts_text_and_pages(corpus: Path) -> None:
    config = ExtractionConfig(ocr_fallback=False)
    document = extract_document(corpus / "doc_a.pdf", config)
    assert document.doc_id == "doc_a"
    assert document.page_count == 2
    assert "California" in document.text
    assert document.extractor == "pymupdf"
    assert document.ocr is False
    assert document.ocr_needed is False
    assert document.char_count == len(document.text)


def test_thin_text_layer_is_flagged(corpus: Path) -> None:
    """A scanned document must be flagged, never quietly passed through as near-empty."""
    config = ExtractionConfig(ocr_fallback=False, ocr_chars_per_page=100)
    document = extract_document(corpus / "doc_scan.pdf", config)
    assert document.ocr_needed is True
    assert document.ocr is False
    assert document.chars_per_page < 100


def test_fallback_extractor_is_recorded(corpus: Path) -> None:
    config = ExtractionConfig(extractor="pdfplumber", fallback_extractor=None, ocr_fallback=False)
    document = extract_document(corpus / "doc_a.pdf", config)
    assert document.extractor == "pdfplumber"
    assert "California" in document.text


@pytest.mark.parametrize(
    "strategy,expected_head,expected_tail",
    [("head", True, False), ("head_tail", True, True)],
)
def test_truncation_policy(strategy: str, expected_head: bool, expected_tail: bool) -> None:
    text = "HEAD" + "x" * 500 + "TAIL"
    policy = TruncationConfig(max_chars=100, strategy=strategy)  # type: ignore[arg-type]
    cut, truncated = truncate(text, policy)
    assert truncated is True
    assert len(cut) <= 100 + len("\n\n[... truncated ...]\n\n")
    assert cut.startswith("HEAD") is expected_head
    assert cut.endswith("TAIL") is expected_tail


def test_truncation_is_a_no_op_when_it_fits() -> None:
    text = "short"
    cut, truncated = truncate(text, TruncationConfig(max_chars=100))
    assert (cut, truncated) == ("short", False)
    cut, truncated = truncate(text, TruncationConfig())
    assert (cut, truncated) == ("short", False)


def test_corpus_writes_cache_and_manifest(tmp_path: Path, corpus: Path) -> None:
    text_dir = tmp_path / "text"
    manifest = tmp_path / "extraction_manifest.csv"
    config = ExtractionConfig(ocr_fallback=False)
    documents = extract_corpus(corpus, text_dir, manifest, config)

    assert [d.doc_id for d in documents] == ["doc_a", "doc_b", "doc_scan"]
    assert sorted(p.name for p in text_dir.glob("*.md")) == ["doc_a.md", "doc_b.md", "doc_scan.md"]

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = {row["doc_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"doc_a", "doc_b", "doc_scan"}
    assert rows["doc_a"]["page_count"] == "2"
    assert rows["doc_scan"]["ocr_needed"] == "True"
    assert rows["doc_a"]["stratum"] == "random"
    assert rows["doc_a"]["extractor"] == "pymupdf"


def test_corpus_reuses_the_cache(tmp_path: Path, corpus: Path) -> None:
    """Re-running extraction must reuse cached text, so downstream metrics stay comparable."""
    text_dir = tmp_path / "text"
    manifest = tmp_path / "extraction_manifest.csv"
    config = ExtractionConfig(ocr_fallback=False)
    extract_corpus(corpus, text_dir, manifest, config)
    marker = "CACHED SENTINEL"
    (text_dir / "doc_a.md").write_text(marker, encoding="utf-8")

    again = extract_corpus(corpus, text_dir, manifest, config)
    assert next(d for d in again if d.doc_id == "doc_a").text == marker

    forced = extract_corpus(corpus, text_dir, manifest, config, force=True)
    assert "California" in next(d for d in forced if d.doc_id == "doc_a").text


def test_strata_are_recorded(tmp_path: Path, corpus: Path) -> None:
    text_dir = tmp_path / "text"
    manifest = tmp_path / "extraction_manifest.csv"
    documents = extract_corpus(
        corpus, text_dir, manifest, ExtractionConfig(ocr_fallback=False), strata={"doc_b": "keyword"}
    )
    assert next(d for d in documents if d.doc_id == "doc_b").stratum == "keyword"


def test_empty_directory_fails_loudly(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no PDFs found"):
        extract_corpus(empty, tmp_path / "text", tmp_path / "m.csv", ExtractionConfig())


def test_missing_cached_text_fails_loudly(tmp_path: Path, corpus: Path) -> None:
    text_dir = tmp_path / "text"
    extract_corpus(corpus, text_dir, tmp_path / "m.csv", ExtractionConfig(ocr_fallback=False))
    with pytest.raises(FileNotFoundError, match="no cached text for 1 document"):
        load_texts(text_dir, ["doc_a", "does_not_exist"])


def test_load_texts_returns_every_requested_document(tmp_path: Path, corpus: Path) -> None:
    text_dir = tmp_path / "text"
    extract_corpus(corpus, text_dir, tmp_path / "m.csv", ExtractionConfig(ocr_fallback=False))
    texts = load_texts(text_dir, ["doc_a", "doc_b"])
    assert set(texts) == {"doc_a", "doc_b"}
    assert "California" in texts["doc_a"]
