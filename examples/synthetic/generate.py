"""Generate the synthetic corpus used as doc-ai-kit's acceptance test.

Twenty short agreements covering every task type. The span task's gold values are computed
*after* extraction, against the cached text, because a character offset only means anything
relative to the exact text the program is shown.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from doc_ai_kit.config import ExtractionConfig, TranscriptionConfig
from doc_ai_kit.dataset import LabelRecord, load_labels, write_labels
from doc_ai_kit.extract import extract_corpus

logger = logging.getLogger(__name__)

PROVIDER = "Northwind Traders, Inc."

STATES = {
    "CA": "California",
    "NY": "New York",
    "TX": "Texas",
    "DE": "Delaware",
}

COUNTERPARTIES = [
    "Contoso Manufacturing LLC",
    "Fabrikam Industries, Inc.",
    "Tailspin Toys Limited",
    "Adventure Works Corporation",
    "Wide World Importers, Inc.",
    "Proseware Holdings LLC",
    "Litware Analytics, Inc.",
    "Lucerne Publishing Ltd.",
    "Graphic Design Institute",
    "Woodgrove Logistics LLC",
]

PEOPLE = [
    "Jane Doe",
    "John Roe",
    "Ana Ruiz",
    "Kofi Mensah",
    "Mei Tanaka",
    "Priya Nair",
    "Omar Haddad",
    "Lena Vogel",
    "Tomas Novak",
    "Sara Lindqvist",
]

PRODUCTS = ["hardware", "software", "services", "support"]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip


@dataclass
class DocumentSpec:
    """The ground truth a synthetic document is generated from."""

    doc_id: str
    state: str
    arbitration: bool
    products: list[str]
    contract_number: str
    counterparty: str
    signatories: list[str]
    value: int
    year: int
    month: int
    day: int
    governing_law_sentence: str = field(default="")

    @property
    def effective_date(self) -> str:
        """Return the ISO 8601 effective date."""
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    @property
    def date_text(self) -> str:
        """Return the effective date as it is written in the document."""
        return f"{MONTHS[self.month - 1]} {self.day}, {self.year}"

    def labels(self) -> dict[str, object]:
        """Return the gold labels that do not depend on the extracted text."""
        return {
            "has_arbitration_clause": self.arbitration,
            "filing_state": self.state,
            "covered_products": sorted(self.products),
            "contract_number": self.contract_number,
            "counterparty": self.counterparty,
            "signatories": list(self.signatories),
            "contract_value": {"value": float(self.value), "unit": "USD"},
            "effective_date": self.effective_date,
        }


def build_specs(count: int = 20) -> list[DocumentSpec]:
    """Build the corpus specification, keeping every class above this run's support floor.

    Deterministic rather than random: with twenty documents, a random draw leaves classes
    with one or two examples and the acceptance test stops testing the thing it is for.

    :param count: How many documents to generate
    :returns: One spec per document
    """
    state_codes = list(STATES)
    specs: list[DocumentSpec] = []
    for index in range(count):
        # states and the arbitration flag are assigned round-robin so no class goes scarce
        state = state_codes[index % len(state_codes)]
        arbitration = index % 2 == 0
        products = sorted({PRODUCTS[index % 4], PRODUCTS[(index + 1 + index // 4) % 4]})
        signatories = [PEOPLE[index % len(PEOPLE)]]
        if index % 3 == 0:
            signatories.append(PEOPLE[(index + 3) % len(PEOPLE)])
        specs.append(
            DocumentSpec(
                doc_id=f"doc_{index:02d}",
                state=state,
                arbitration=arbitration,
                products=products,
                contract_number=f"MSA-{2024 + index % 2}-{1000 + index * 7}",
                counterparty=COUNTERPARTIES[index % len(COUNTERPARTIES)],
                signatories=signatories,
                value=(index + 1) * 125_000,
                year=2024 + index % 2,
                month=1 + (index * 5) % 12,
                day=1 + (index * 7) % 28,
            )
        )
    return specs


def _sentences(spec: DocumentSpec) -> list[tuple[str, str]]:
    """Return the document's paragraphs as (style, text) pairs."""
    dispute = (
        "Any dispute arising under or relating to this Agreement shall be finally resolved by "
        "binding arbitration administered in the jurisdiction named below, and the parties waive "
        "any right to a trial by jury."
        if spec.arbitration
        else "Any dispute arising under or relating to this Agreement shall be resolved exclusively "
        "in the state and federal courts of competent jurisdiction, and the parties expressly "
        "decline arbitration."
    )
    governing = (
        f"This Agreement shall be governed by and construed in accordance with the laws of the "
        f"State of {STATES[spec.state]}, without regard to its conflict of laws principles."
    )
    spec.governing_law_sentence = governing
    products = ", ".join(spec.products)
    signature_lines = "<br/>".join(f"{name}, authorised signatory" for name in spec.signatories)
    return [
        ("Title", "MASTER SERVICES AGREEMENT"),
        ("Body", f"Contract Number: {spec.contract_number}"),
        (
            "Body",
            f"This Master Services Agreement is entered into as of {spec.date_text}, by and between "
            f'{PROVIDER}, a corporation with offices in Wilmington, Delaware ("Provider"), and '
            f'{spec.counterparty} ("Customer").',
        ),
        ("Heading", "1. Scope"),
        ("Body", f"The Provider shall supply the following product lines to the Customer: {products}."),
        ("Heading", "2. Consideration"),
        (
            "Body",
            f"The total contract value is ${spec.value:,}, payable in equal quarterly instalments " "over the term.",
        ),
        ("Heading", "3. Dispute Resolution"),
        ("Body", dispute),
        ("Heading", "4. Governing Law"),
        ("Body", governing),
        ("Heading", "5. Signatures"),
        ("Body", f"Executed by the Customer on {spec.date_text}:<br/>{signature_lines}"),
    ]


def write_pdf(spec: DocumentSpec, pdf_dir: Path) -> Path:
    """Render one synthetic agreement to a PDF."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Title"], fontSize=14, spaceAfter=12)
    heading_style = ParagraphStyle("DocHeading", parent=styles["Heading2"], fontSize=11, spaceBefore=10)
    body_style = ParagraphStyle("DocBody", parent=styles["BodyText"], fontSize=10, leading=14)
    lookup = {"Title": title_style, "Heading": heading_style, "Body": body_style}

    path = pdf_dir / f"{spec.doc_id}.pdf"
    document = SimpleDocTemplate(str(path), pagesize=LETTER, title=spec.doc_id)
    flowables: list[object] = []
    for style_name, text in _sentences(spec):
        flowables.append(Paragraph(text, lookup[style_name]))
        flowables.append(Spacer(1, 4))
    document.build(flowables)
    return path


def span_for(sentence: str, text: str) -> dict[str, int] | None:
    """Locate the governing-law sentence in the extracted text.

    Offsets are computed against the cached text rather than the source, because that is the
    text the program is shown, and an offset into anything else describes nothing.
    """
    needle = " ".join(sentence.split())
    haystack = " ".join(text.split())
    position = haystack.find(needle)
    if position < 0:
        return None
    # map back onto the original text by matching the prefix word by word
    words = haystack[:position].split()
    offset = 0
    for word in words:
        offset = text.find(word, offset) + len(word)
    start = text.find(needle.split()[0], offset)
    if start < 0:
        return None
    last_word = needle.split()[-1]
    end = text.find(last_word, start)
    if end < 0:
        return None
    return {"start": start, "end": end + len(last_word)}


def generate(project_dir: Path, count: int = 20) -> list[LabelRecord]:
    """Generate the corpus, extract it, and write gold labels including the span offsets.

    :param project_dir: The synthetic project's root
    :param count: How many documents to generate
    :returns: The gold label records
    """
    data = project_dir / "data"
    pdf_dir = data / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs(count)
    for spec in specs:
        write_pdf(spec, pdf_dir)
    logger.info("wrote %d synthetic PDFs to %s", len(specs), pdf_dir)

    documents = extract_corpus(
        pdf_dir,
        data / "text",
        data / "extraction_manifest.csv",
        ExtractionConfig(transcription=TranscriptionConfig(mode="off")),
        force=True,
    )
    texts = {document.doc_id: document.text for document in documents}

    records: list[LabelRecord] = []
    missing_spans = []
    for spec in specs:
        labels = spec.labels()
        span = span_for(spec.governing_law_sentence, texts[spec.doc_id])
        if span is None:
            missing_spans.append(spec.doc_id)
        labels["governing_law_span"] = span
        records.append(
            LabelRecord(
                doc_id=spec.doc_id,
                labels=labels,
                stratum="random",
                inclusion_probability=1.0,
                labeling_mode="corrected",
            )
        )
    if missing_spans:
        raise RuntimeError(f"could not locate the governing-law sentence in: {', '.join(missing_spans)}")
    write_labels(data / "labels.jsonl", records)
    return records


def mark_holdout_blind(project_dir: Path) -> int:
    """Mark the holdout documents as blind-labeled, once the splits exist.

    In a real project this is a labeling pass, not a flag: the holdout is labeled from
    scratch, without seeing any prediction, after the splits are fixed. The synthetic corpus
    has no human in it, so this records what that pass would have produced. The ordering is
    the part that matters and is preserved -- splits first, blind labels second.

    :param project_dir: The synthetic project's root
    :returns: How many records were marked
    """
    from doc_ai_kit.dataset import load_splits

    data = project_dir / "data"
    splits = load_splits(data / "splits.json")
    holdout = set(splits.holdout)
    records = load_labels(data / "labels.jsonl")
    updated = [
        LabelRecord(
            doc_id=record.doc_id,
            labels=record.labels,
            stratum=record.stratum,
            inclusion_probability=record.inclusion_probability,
            labeling_mode="blind" if record.doc_id in holdout else record.labeling_mode,
            notes=record.notes,
        )
        for record in records
    ]
    write_labels(data / "labels.jsonl", updated)
    return len(holdout)


def main() -> None:
    """Generate the synthetic corpus into a project directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="The synthetic project directory")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--mark-holdout-blind",
        action="store_true",
        help="Record the holdout as blind-labeled; run this after make-splits, never before.",
    )
    args = parser.parse_args()
    if args.mark_holdout_blind:
        marked = mark_holdout_blind(args.project)
        print(f"Marked {marked} holdout document(s) as blind-labeled.")
        return
    records = generate(args.project, args.count)
    print(f"Generated {len(records)} documents into {args.project / 'data'}.")


if __name__ == "__main__":
    main()
