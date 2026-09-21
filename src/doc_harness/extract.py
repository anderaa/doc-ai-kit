"""PDF text extraction and its manifest.

Text is extracted once and cached on disk. Validation and production read the same cached
text, because metrics computed on one rendering of a document do not transfer to another.

Documents whose text layer is effectively empty are flagged rather than quietly passed
through: that flag has to reach error analysis and production QA triage, since a task
scoring zero on a scanned document is an extraction problem and no amount of prompt
optimization will fix it.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from doc_harness.config import ExtractionConfig, TruncationConfig
from doc_harness.hooks import HookError, get_extractor, register_extractor

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = (
    "doc_id",
    "source_path",
    "page_count",
    "char_count",
    "chars_per_page",
    "extractor",
    "ocr",
    "ocr_needed",
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
    ocr: bool
    ocr_needed: bool
    truncated: bool
    original_char_count: int
    stratum: str = "random"

    def manifest_row(self) -> dict[str, object]:
        """Return this document's manifest row, without the text itself."""
        row = asdict(self)
        row.pop("text")
        return row


@register_extractor("pymupdf")
def extract_with_pymupdf(pdf_path: Path, config: ExtractionConfig) -> ExtractedText:
    """Extract a PDF's text layer with PyMuPDF."""
    import pymupdf

    # pymupdf ships stubs but leaves open() untyped
    with pymupdf.open(pdf_path) as document:  # type: ignore[no-untyped-call]
        pages = [page.get_text("text") for page in document]
    return ExtractedText(text="\n\n".join(pages), page_count=len(pages), extractor="pymupdf")


@register_extractor("pdfplumber")
def extract_with_pdfplumber(pdf_path: Path, config: ExtractionConfig) -> ExtractedText:
    """Extract a PDF's text layer with pdfplumber."""
    import pdfplumber

    with pdfplumber.open(pdf_path) as document:
        pages = [page.extract_text() or "" for page in document.pages]
    return ExtractedText(text="\n\n".join(pages), page_count=len(pages), extractor="pdfplumber")


@register_extractor("ocr")
def extract_with_ocr(pdf_path: Path, config: ExtractionConfig) -> ExtractedText:
    """Rasterise a PDF and read it with Tesseract.

    Requires the optional ``ocr`` extra; the import error is raised rather than swallowed,
    because falling back to empty text would look like a document with nothing in it.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise HookError(
            "OCR fallback needs the optional dependencies: pip install 'doc-harness[ocr]' "
            "and a working tesseract binary"
        ) from exc

    images = convert_from_path(str(pdf_path))
    pages = [pytesseract.image_to_string(image, lang=config.ocr_language) for image in images]
    return ExtractedText(text="\n\n".join(pages), page_count=len(pages), extractor="ocr")


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
) -> ExtractedDocument:
    """Extract one PDF to text, applying the fallback and OCR policy.

    :param pdf_path: The PDF to read
    :param config: The project's extraction settings
    :param doc_id: The document id, defaulting to the file stem
    :param stratum: Which sampling stratum this document came from
    :returns: The extracted document, ready to cache
    """
    doc_id = doc_id or pdf_path.stem
    extracted = get_extractor(config.extractor)(pdf_path, config)
    if not extracted.text.strip() and config.fallback_extractor:
        logger.info("%s: %s produced no text, trying %s", doc_id, config.extractor, config.fallback_extractor)
        extracted = get_extractor(config.fallback_extractor)(pdf_path, config)

    pages = max(extracted.page_count, 1)
    chars_per_page = len(extracted.text) / pages
    ocr_needed = chars_per_page < config.ocr_chars_per_page
    used_ocr = False
    if ocr_needed and config.ocr_fallback:
        logger.info("%s: %.0f chars/page is below %d, running OCR", doc_id, chars_per_page, config.ocr_chars_per_page)
        extracted = get_extractor(config.ocr_extractor)(pdf_path, config)
        used_ocr = True
        pages = max(extracted.page_count, 1)
        chars_per_page = len(extracted.text) / pages
    elif ocr_needed:
        logger.warning(
            "%s: %.0f chars/page is below %d and OCR is disabled; this document is flagged",
            doc_id,
            chars_per_page,
            config.ocr_chars_per_page,
        )

    original_chars = len(extracted.text)
    text, truncated = truncate(extracted.text, config.truncation)
    return ExtractedDocument(
        doc_id=doc_id,
        source_path=str(pdf_path),
        text=text,
        page_count=extracted.page_count,
        char_count=len(text),
        chars_per_page=round(chars_per_page, 1),
        extractor=extracted.extractor,
        ocr=used_ocr,
        ocr_needed=ocr_needed,
        truncated=truncated,
        original_char_count=original_chars,
        stratum=stratum,
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
    :returns: One record per document, in doc_id order
    """
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDFs found in {pdf_dir}")
    text_dir.mkdir(parents=True, exist_ok=True)
    strata = strata or {}
    cached = _load_manifest(manifest_path)

    documents: list[ExtractedDocument] = []
    for pdf_path in pdfs:
        doc_id = pdf_path.stem
        target = text_path(text_dir, doc_id)
        previous = cached.get(doc_id)
        if not force and target.exists() and previous is not None:
            documents.append(_rehydrate(previous, target.read_text(encoding="utf-8")))
            continue
        document = extract_document(pdf_path, config, doc_id=doc_id, stratum=strata.get(doc_id, "random"))
        target.write_text(document.text, encoding="utf-8")
        documents.append(document)

    if len(documents) != len(pdfs):
        raise RuntimeError(f"extracted {len(documents)} documents from {len(pdfs)} PDFs: documents were dropped")
    write_manifest(manifest_path, documents)
    flagged = [document.doc_id for document in documents if document.ocr_needed]
    if flagged:
        logger.warning("%d document(s) had a thin or missing text layer: %s", len(flagged), ", ".join(flagged))
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
        ocr=row["ocr"] == "True",
        ocr_needed=row["ocr_needed"] == "True",
        truncated=row["truncated"] == "True",
        original_char_count=int(row["original_char_count"]),
        stratum=row.get("stratum", "random"),
    )


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
