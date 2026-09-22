"""PDF text extraction and its manifest.

Text is extracted once and cached on disk. Validation and production read the same cached
text, because metrics computed on one rendering of a document do not transfer to another.

Pages whose text layer is effectively empty -- scans -- are transcribed by Claude from an
image of the page (see :mod:`doc_harness.transcribe`), page by page, and spliced back in
place. A page that still has no usable text is counted in the manifest rather than quietly
passed through: that count has to reach error analysis and production QA triage, since a
task scoring zero on an unread page is an extraction problem and no amount of prompt
optimization will fix it.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from doc_harness.config import ExtractionConfig, TruncationConfig
from doc_harness.hooks import get_extractor, register_extractor
from doc_harness.transcribe import Transcriber

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = (
    "doc_id",
    "source_path",
    "page_count",
    "char_count",
    "chars_per_page",
    "extractor",
    "thin_pages",
    "transcribed_pages",
    "unread_pages",
    "transcription_model",
    "truncated",
    "original_char_count",
    "stratum",
)


@dataclass(frozen=True)
class ExtractedText:
    """The raw result of running one extractor over one PDF."""

    text: str
    page_count: int
    extractor: str
    # each page's text, when the extractor can tell pages apart; transcription is decided per page
    pages: tuple[str, ...] | None = None


@dataclass
class ExtractedDocument:
    """One document's cached text and everything the manifest records about it."""

    doc_id: str
    source_path: str
    text: str
    page_count: int
    char_count: int
    chars_per_page: float
    extractor: str
    truncated: bool
    original_char_count: int
    stratum: str = "random"
    # pages whose text layer was below extraction.transcription.min_chars_per_page
    thin_pages: int = 0
    # pages whose text came from Claude's transcription
    transcribed_pages: int = 0
    # pages still without complete text: not transcribed, failed, or cut off at max_tokens
    unread_pages: int = 0
    transcription_model: str = ""
    # tokens spent on transcription in this run; cached pages cost nothing. Not in the manifest
    transcription_tokens: tuple[int, int] = field(default=(0, 0))

    @property
    def needs_review(self) -> bool:
        """Return whether any page's text is missing, incomplete, or read from an image."""
        return self.unread_pages > 0 or self.transcribed_pages > 0

    def manifest_row(self) -> dict[str, object]:
        """Return this document's manifest row, without the text itself."""
        row = asdict(self)
        row.pop("text")
        row.pop("transcription_tokens")
        return row


@register_extractor("pymupdf")
def extract_with_pymupdf(pdf_path: Path, config: ExtractionConfig) -> ExtractedText:
    """Extract a PDF's text layer with PyMuPDF."""
    import pymupdf

    # pymupdf ships stubs but leaves open() untyped
    with pymupdf.open(pdf_path) as document:  # type: ignore[no-untyped-call]
        pages = [page.get_text("text") for page in document]
    return ExtractedText(text="\n\n".join(pages), page_count=len(pages), extractor="pymupdf", pages=tuple(pages))


@register_extractor("pdfplumber")
def extract_with_pdfplumber(pdf_path: Path, config: ExtractionConfig) -> ExtractedText:
    """Extract a PDF's text layer with pdfplumber."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as document:
        pages = [page.extract_text() or "" for page in document.pages]
    return ExtractedText(text="\n\n".join(pages), page_count=len(pages), extractor="pdfplumber", pages=tuple(pages))


def truncate(text: str, policy: TruncationConfig) -> tuple[str, bool]:
    """Apply the project's truncation policy.

    :param text: The full extracted text
    :param policy: The policy declared once in config.yaml
    :returns: The text to cache, and whether truncation actually happened
    """
    if policy.max_chars is None or len(text) <= policy.max_chars:
        return text, False
    if policy.strategy == "head":
        return text[: policy.max_chars], True
    head_chars = int(policy.max_chars * policy.head_share)
    tail_chars = policy.max_chars - head_chars
    marker = "\n\n[... truncated ...]\n\n"
    return text[:head_chars] + marker + text[-tail_chars:], True


def extract_document(
    pdf_path: Path,
    config: ExtractionConfig,
    doc_id: str | None = None,
    stratum: str = "random",
    transcriber: Transcriber | None = None,
) -> ExtractedDocument:
    """Extract one PDF to text, transcribing pages that have no usable text layer.

    :param pdf_path: The PDF to read
    :param config: The project's extraction settings
    :param doc_id: The document id, defaulting to the file stem
    :param stratum: Which sampling stratum this document came from
    :param transcriber: Reads page images with Claude; None when no transcription model is
        set, in which case thin pages are counted as unread
    :returns: The extracted document, ready to cache
    """
    doc_id = doc_id or pdf_path.stem
    extracted = get_extractor(config.extractor)(pdf_path, config)
    if not extracted.text.strip() and config.fallback_extractor:
        logger.info("%s: %s produced no text, trying %s", doc_id, config.extractor, config.fallback_extractor)
        extracted = get_extractor(config.fallback_extractor)(pdf_path, config)

    settings = config.transcription
    page_count = max(extracted.page_count, 1)
    if extracted.pages is not None:
        pages = list(extracted.pages)
        thin = [index for index, page in enumerate(pages) if len(page.strip()) < settings.min_chars_per_page]
    else:
        # an extractor that cannot tell pages apart: judged on the whole document instead
        pages = [""] * page_count
        pages[0] = extracted.text
        thin_document = len(extracted.text) / page_count < settings.min_chars_per_page
        thin = list(range(page_count)) if thin_document else []

    wanted: list[int] = []
    if settings.mode == "all_pages":
        wanted = list(range(len(pages)))
    elif settings.mode == "thin_pages":
        wanted = list(thin)

    transcribed: set[int] = set()
    incomplete: set[int] = set()
    tokens = (0, 0)
    model = ""
    if wanted and transcriber is not None:
        results = transcriber.transcribe(pdf_path, doc_id, wanted)
        model = transcriber.model_name
        tokens = (sum(r.input_tokens for r in results.values()), sum(r.output_tokens for r in results.values()))
        for index, result in results.items():
            if result.status == "failed":
                continue
            # a thin page keeps its own few characters if the transcription somehow came back shorter
            if settings.mode == "thin_pages" and len(result.text) < len(pages[index].strip()):
                continue
            pages[index] = result.text
            transcribed.add(index)
            if result.status == "truncated":
                incomplete.add(index)
    elif wanted:
        logger.warning(
            "%s: %d page(s) have no usable text layer and extraction.transcription.model is not set; "
            "they are left as they are and flagged",
            doc_id,
            len(thin),
        )

    unread = (set(thin) - transcribed) | incomplete
    full_text = "\n\n".join(pages)
    text, truncated = truncate(full_text, config.truncation)
    return ExtractedDocument(
        doc_id=doc_id,
        source_path=str(pdf_path),
        text=text,
        page_count=extracted.page_count,
        char_count=len(text),
        chars_per_page=round(len(full_text) / page_count, 1),
        extractor=extracted.extractor,
        truncated=truncated,
        original_char_count=len(full_text),
        stratum=stratum,
        thin_pages=len(thin),
        transcribed_pages=len(transcribed),
        unread_pages=len(unread),
        transcription_model=model,
        transcription_tokens=tokens,
    )


def text_path(text_dir: Path, doc_id: str) -> Path:
    """Return where one document's cached text lives."""
    return text_dir / f"{doc_id}.md"


def extract_corpus(
    pdf_dir: Path,
    text_dir: Path,
    manifest_path: Path,
    config: ExtractionConfig,
    strata: dict[str, str] | None = None,
    force: bool = False,
    transcript_dir: Path | None = None,
    client: Any = None,
) -> list[ExtractedDocument]:
    """Extract every PDF in a directory, caching text and writing the manifest.

    Extraction is skipped for documents already cached, unless ``force`` is set, so the run
    is cheap to repeat and the same text is reused everywhere.

    :param pdf_dir: Directory of source PDFs
    :param text_dir: Where cached text is written
    :param manifest_path: Where extraction_manifest.csv is written
    :param config: The project's extraction settings
    :param strata: Optional per-document stratum labels
    :param force: Re-extract even when cached text exists
    :param transcript_dir: Where page transcriptions are cached; data/transcripts beside the text
    :param client: An Anthropic client for transcription; created on first use if not given
    :returns: One record per document, in doc_id order
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found in {pdf_dir}")
    text_dir.mkdir(parents=True, exist_ok=True)
    strata = strata or {}
    cached = _load_manifest(manifest_path)
    settings = config.transcription
    transcriber = (
        Transcriber(settings, transcript_dir or text_dir.parent / "transcripts", client=client)
        if settings.mode != "off" and settings.model is not None
        else None
    )

    documents: list[ExtractedDocument] = []
    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        target = text_path(text_dir, doc_id)
        previous = cached.get(doc_id)
        # a document with unread pages is retried once a transcription model is set, without --force
        retry = transcriber is not None and previous is not None and int(previous.get("unread_pages") or 0) > 0
        if not force and not retry and target.exists() and previous is not None:
            documents.append(_rehydrate(previous, target.read_text(encoding="utf-8")))
            continue
        document = extract_document(
            pdf_path, config, doc_id=doc_id, stratum=strata.get(doc_id, "random"), transcriber=transcriber
        )
        target.write_text(document.text, encoding="utf-8")
        documents.append(document)

    if len(documents) != len(pdfs):
        raise RuntimeError(f"extracted {len(documents)} documents from {len(pdfs)} PDFs: documents were dropped")
    write_manifest(manifest_path, documents)
    flagged = [document.doc_id for document in documents if document.unread_pages]
    if flagged:
        logger.warning("%d document(s) have pages with no usable text: %s", len(flagged), ", ".join(flagged))
    return documents


def _rehydrate(row: dict[str, str], text: str) -> ExtractedDocument:
    """Rebuild a record from a manifest row and cached text."""
    return ExtractedDocument(
        doc_id=row["doc_id"],
        source_path=row["source_path"],
        text=text,
        page_count=int(row["page_count"]),
        char_count=int(row["char_count"]),
        chars_per_page=float(row["chars_per_page"]),
        extractor=row["extractor"],
        truncated=row["truncated"] == "True",
        original_char_count=int(row["original_char_count"]),
        stratum=row.get("stratum", "random"),
        **_transcription_columns(row),
    )


def _transcription_columns(row: dict[str, str]) -> dict[str, Any]:
    """Read the per-page counts, including from a manifest written before 0.1.7.

    Those manifests carried whole-document OCR flags. A document Tesseract read keeps its
    text, marked as transcribed by it; one flagged but never read counts every page unread,
    so it is retried as soon as a transcription model is set.
    """
    if "unread_pages" in row:
        return {
            "thin_pages": int(row["thin_pages"] or 0),
            "transcribed_pages": int(row["transcribed_pages"] or 0),
            "unread_pages": int(row["unread_pages"] or 0),
            "transcription_model": row.get("transcription_model", ""),
        }
    pages = int(row["page_count"])
    if row.get("ocr") == "True":
        return {"thin_pages": pages, "transcribed_pages": pages, "unread_pages": 0, "transcription_model": "tesseract"}
    if row.get("ocr_needed") == "True":
        return {"thin_pages": pages, "transcribed_pages": 0, "unread_pages": pages, "transcription_model": ""}
    return {}


def _load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Read an existing manifest, so cached documents keep their recorded provenance."""
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        return {row["doc_id"]: row for row in csv.DictReader(handle)}


def write_manifest(manifest_path: Path, documents: Iterable[ExtractedDocument]) -> None:
    """Write extraction_manifest.csv."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for document in documents:
            writer.writerow(document.manifest_row())
    logger.info("wrote %s", manifest_path)


def load_texts(text_dir: Path, doc_ids: Sequence[str]) -> dict[str, str]:
    """Load cached text for a set of documents, failing loudly on a missing one.

    :param text_dir: Where cached text lives
    :param doc_ids: The documents to load
    :returns: Document id to text
    """
    texts: dict[str, str] = {}
    missing: list[str] = []
    for doc_id in doc_ids:
        path = text_path(text_dir, doc_id)
        if not path.exists():
            missing.append(doc_id)
            continue
        texts[doc_id] = path.read_text(encoding="utf-8")
    if missing:
        raise FileNotFoundError(f"no cached text for {len(missing)} document(s): {', '.join(sorted(missing))}")
    return texts
