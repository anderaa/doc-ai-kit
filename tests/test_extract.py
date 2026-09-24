"""Extraction tests over PDFs generated on the fly."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from doc_ai_kit.config import ExtractionConfig, TruncationConfig
from doc_ai_kit.extract import extract_corpus, extract_document, load_texts, truncate


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


def make_scanned_pdf(path: Path) -> Path:
    """Write a one-page PDF whose only content is an image of text, as a scanner produces."""
    import pymupdf

    source = pymupdf.open()  # type: ignore[no-untyped-call]
    source.new_page().insert_text((72, 100), "Scanned text", fontsize=14)
    pixmap = source[0].get_pixmap(dpi=72)
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    page = document.new_page()
    page.insert_image(page.rect, pixmap=pixmap)
    document.save(path)
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    body = "\n".join(f"This agreement is governed by the laws of California. Line {i}." for i in range(30))
    make_pdf(pdf_dir / "doc_a.pdf", [body, body])
    make_pdf(pdf_dir / "doc_b.pdf", [body])
    # a scanned document: the page is an image, with no text layer
    make_scanned_pdf(pdf_dir / "doc_scan.pdf")
    return pdf_dir


def test_extracts_text_and_pages(corpus: Path) -> None:
    config = ExtractionConfig()
    document = extract_document(corpus / "doc_a.pdf", config)
    assert document.doc_id == "doc_a"
    assert document.page_count == 2
    assert "California" in document.text
    assert document.extractor == "pymupdf"
    assert document.thin_pages == document.unread_pages == document.transcribed_pages == 0
    assert document.char_count == len(document.text)


def test_thin_text_layer_is_flagged(corpus: Path) -> None:
    """A scanned document must be flagged, never quietly passed through as near-empty."""
    document = extract_document(corpus / "doc_scan.pdf", ExtractionConfig())
    assert document.thin_pages == document.unread_pages == 1
    assert document.transcribed_pages == 0
    assert document.chars_per_page < 100


def test_fallback_extractor_is_recorded(corpus: Path) -> None:
    config = ExtractionConfig(extractor="pdfplumber", fallback_extractor=None)
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
    config = ExtractionConfig()
    documents = extract_corpus(corpus, text_dir, manifest, config)

    assert [d.doc_id for d in documents] == ["doc_a", "doc_b", "doc_scan"]
    assert sorted(p.name for p in text_dir.glob("*.md")) == ["doc_a.md", "doc_b.md", "doc_scan.md"]

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = {row["doc_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"doc_a", "doc_b", "doc_scan"}
    assert rows["doc_a"]["page_count"] == "2"
    assert rows["doc_scan"]["unread_pages"] == "1"
    assert rows["doc_a"]["unread_pages"] == "0"
    assert rows["doc_a"]["stratum"] == "random"
    assert rows["doc_a"]["extractor"] == "pymupdf"


def test_corpus_reuses_the_cache(tmp_path: Path, corpus: Path) -> None:
    """Re-running extraction must reuse cached text, so downstream metrics stay comparable."""
    text_dir = tmp_path / "text"
    manifest = tmp_path / "extraction_manifest.csv"
    config = ExtractionConfig()
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
    documents = extract_corpus(corpus, text_dir, manifest, ExtractionConfig(), strata={"doc_b": "keyword"})
    assert next(d for d in documents if d.doc_id == "doc_b").stratum == "keyword"


def test_empty_directory_fails_loudly(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no PDFs found"):
        extract_corpus(empty, tmp_path / "text", tmp_path / "m.csv", ExtractionConfig())


def test_missing_cached_text_fails_loudly(tmp_path: Path, corpus: Path) -> None:
    text_dir = tmp_path / "text"
    extract_corpus(corpus, text_dir, tmp_path / "m.csv", ExtractionConfig())
    with pytest.raises(FileNotFoundError, match="no cached text for 1 document"):
        load_texts(text_dir, ["doc_a", "does_not_exist"])


def test_load_texts_returns_every_requested_document(tmp_path: Path, corpus: Path) -> None:
    text_dir = tmp_path / "text"
    extract_corpus(corpus, text_dir, tmp_path / "m.csv", ExtractionConfig())
    texts = load_texts(text_dir, ["doc_a", "doc_b"])
    assert set(texts) == {"doc_a", "doc_b"}
    assert "California" in texts["doc_a"]


def test_removed_ocr_settings_say_what_replaced_them() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="ocr_chars_per_page -> transcription.min_chars_per_page"):
        ExtractionConfig.model_validate({"ocr_fallback": True, "ocr_chars_per_page": 100})


def test_an_unquoted_off_means_off() -> None:
    """YAML reads `mode: off` as false."""
    assert ExtractionConfig.model_validate({"transcription": {"mode": False}}).transcription.mode == "off"


def test_transcription_model_must_be_claude() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must be an anthropic/ model"):
        ExtractionConfig.model_validate({"transcription": {"model": "openai/gpt-x"}})


def test_manifests_from_before_transcription_still_load(tmp_path: Path, corpus: Path) -> None:
    """A project upgraded from 0.1.6 keeps its cache; a flagged document is retried once a model is set."""
    text_dir = tmp_path / "text"
    manifest = tmp_path / "m.csv"
    extract_corpus(corpus, text_dir, manifest, ExtractionConfig())
    old_columns = (
        "doc_id,source_path,page_count,char_count,chars_per_page,extractor,"
        "ocr,ocr_needed,truncated,original_char_count,stratum\n"
    )
    rows = [
        f"doc_a,{corpus}/doc_a.pdf,2,10,5.0,pymupdf,False,False,False,10,random",
        f"doc_b,{corpus}/doc_b.pdf,1,10,10.0,ocr,True,True,False,10,random",
        f"doc_scan,{corpus}/doc_scan.pdf,1,1,1.0,pymupdf,False,True,False,1,random",
    ]
    manifest.write_text(old_columns + "\n".join(rows) + "\n", encoding="utf-8")
    documents = {d.doc_id: d for d in extract_corpus(corpus, text_dir, manifest, ExtractionConfig())}
    assert documents["doc_b"].transcription_model == "tesseract" and documents["doc_b"].unread_pages == 0
    assert documents["doc_scan"].unread_pages == 1


def test_uppercase_pdf_extensions_are_found(tmp_path: Path) -> None:
    """Seen in CUAD: 311 of 510 files end in .PDF, and a *.pdf glob skipped every one of them."""
    from doc_ai_kit.extract import list_pdfs

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    body = "The agreement is governed by the laws of Delaware. " * 5
    make_pdf(pdf_dir / "lower.pdf", [body])
    make_pdf(pdf_dir / "Upper Case Agreement .PDF", [body])
    (pdf_dir / "notes.docx").write_text("not a pdf", encoding="utf-8")
    (pdf_dir / ".DS_Store").write_text("", encoding="utf-8")
    assert [path.name for path in list_pdfs(pdf_dir)] == ["Upper Case Agreement .PDF", "lower.pdf"]

    documents = extract_corpus(pdf_dir, tmp_path / "text", tmp_path / "m.csv", ExtractionConfig())
    assert sorted(d.doc_id for d in documents) == ["Upper Case Agreement", "lower"]
    assert (tmp_path / "text" / "Upper Case Agreement.md").exists()


def test_two_files_with_the_same_document_id_are_refused(tmp_path: Path) -> None:
    from doc_ai_kit.extract import list_pdfs

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    make_pdf(pdf_dir / "Contract.pdf", ["x"])
    make_pdf(pdf_dir / "Contract .PDF", ["x"])
    with pytest.raises(ValueError, match="would both be document 'Contract'"):
        list_pdfs(pdf_dir)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.pdf", "a"),
        ("A.PDF", "A"),
        ("Deal .PDF", "Deal"),
        ("LECLANCHE S.A. - AGREEMENT", "LECLANCHE S.A. - AGREEMENT"),
    ],
)
def test_doc_id_from_name(name: str, expected: str) -> None:
    from doc_ai_kit.extract import doc_id_from_name

    assert doc_id_from_name(name) == expected


def test_short_and_blank_pages_are_not_scans(tmp_path: Path) -> None:
    """Seen in CUAD: 230 pages under 100 characters, of which 2 were scans. The rest are just short."""
    pdf = make_pdf(tmp_path / "short.pdf", ["EXHIBIT A\n11", "", "Signature page follows."])
    document = extract_document(pdf, ExtractionConfig())
    assert document.thin_pages == document.unread_pages == 0
