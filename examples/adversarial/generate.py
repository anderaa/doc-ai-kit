"""Generate the adversarial corpus: documents where the right answer sits next to a wrong one.

The synthetic corpus states every answer plainly, so its run never made a threshold decide
anything -- fuzzy and strict F1 came out identical. Here each document plants traps: a
"formerly known as" name, a second dollar figure, a notary in the signature block, a venue
clause naming the wrong state. The task definitions are unchanged from the synthetic project,
so any difference in the results comes from the documents.

Every trap carries two lists alongside its gold value:

* ``accept`` -- answer forms a correct model could return, which the matcher must accept;
* ``reject`` -- the distractor, which the matcher must refuse.

Those lists are an offline oracle, checked in tests without any API calls. They are still
guesses about model output; only the live run shows what the model actually writes.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from doc_ai_kit.config import ExtractionConfig, TranscriptionConfig
from doc_ai_kit.dataset import LabelRecord, write_labels
from doc_ai_kit.extract import extract_corpus

logger = logging.getLogger(__name__)

PROVIDER = "Northwind Traders, Inc."
STATES = {"CA": "California", "NY": "New York", "TX": "Texas", "DE": "Delaware"}
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]  # fmt: skip
PEOPLE = [
    ("Jane", "Doe"), ("Kofi", "Mensah"), ("Ana", "Ruiz"), ("Mei", "Tanaka"), ("Priya", "Nair"),
    ("Omar", "Haddad"), ("Lena", "Vogel"), ("Tomas", "Novak"), ("Sara", "Lindqvist"), ("John", "Roe"),
]  # fmt: skip

# state per document: TX is held to three examples so it falls under the support floor of 4
STATE_PLAN = [
    "CA", "NY", "DE", "CA", "NY", "TX", "CA", "DE", "NY", "CA", "DE", "NY",
    "CA", "TX", "DE", "NY", "CA", "DE", "NY", "CA", "TX", "DE", "NY", "CA",
]  # fmt: skip
FAA_DOCS = {1, 8, 15}
VENUE_DOCS = {5, 20}
SEVERABILITY_DOCS = {3, 10, 12, 17, 22}

ARBITRATION_CYCLE = ["binding", "mediation", "optional", "nonbinding", "silent"]
COUNTERPARTY_CYCLE = ["plain", "formerly", "dba", "subsidiary", "dotted", "the", "ampersand"]
VALUE_CYCLE = ["plain", "two_figures", "words", "magnitude", "per_year", "dollars_word"]
DATE_CYCLE = ["plain", "signed_vs_effective", "relative", "month_only", "european"]
SIGNATORY_CYCLE = ["plain", "notary", "titles", "middle_initial", "both_parties"]
PRODUCT_CYCLE = ["plain", "exclusion", "synonyms", "future_option"]
NUMBER_CYCLE = ["plain", "po_and_superseded", "en_dash"]


@dataclass
class Trap:
    """One task's answer in one document, with the forms a matcher must accept or refuse."""

    gold: Any
    kind: str
    accept: list[Any] = field(default_factory=list)
    reject: list[Any] = field(default_factory=list)


@dataclass
class Document:
    """One adversarial agreement and the ground truth it was written from."""

    doc_id: str
    paragraphs: list[tuple[str, str]]
    traps: dict[str, Trap]
    governing_sentence: str

    def labels(self) -> dict[str, Any]:
        """Return the gold labels, as a labeler would have written them."""
        return {task_id: trap.gold for task_id, trap in self.traps.items()}


def _date_text(year: int, month: int, day: int) -> str:
    return f"{MONTHS[month - 1]} {day}, {year}"


def _iso(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def _counterparty(kind: str) -> tuple[str, Trap]:
    """Return the name as the document writes it, and the trap around it."""
    if kind == "formerly":
        return (
            "Contoso Manufacturing LLC (formerly known as Contoso Industries LLC)",
            Trap(
                gold="Contoso Manufacturing LLC",
                kind=kind,
                accept=["Contoso Manufacturing LLC", "Contoso Manufacturing, LLC", "Contoso Manufacturing"],
                reject=["Contoso Industries LLC"],
            ),
        )
    if kind == "dba":
        return (
            "Tailspin Holdings Ltd., doing business as Tailspin Toys",
            Trap(
                gold="Tailspin Holdings Ltd.",
                kind=kind,
                accept=["Tailspin Holdings Ltd.", "Tailspin Holdings Limited", "Tailspin Holdings"],
                reject=["Tailspin Toys"],
            ),
        )
    if kind == "subsidiary":
        return (
            "Fabrikam Industries, Inc., a wholly owned subsidiary of Fabrikam Group plc",
            Trap(
                gold="Fabrikam Industries, Inc.",
                kind=kind,
                accept=["Fabrikam Industries, Inc.", "Fabrikam Industries Inc", "Fabrikam Industries"],
                reject=["Fabrikam Group plc", "Fabrikam Group"],
            ),
        )
    if kind == "dotted":
        # the labeler copied the document's capitals and dotted suffix verbatim
        return (
            "WOODGROVE LOGISTICS, L.L.C.",
            Trap(
                gold="WOODGROVE LOGISTICS, L.L.C.",
                kind=kind,
                accept=["Woodgrove Logistics LLC", "Woodgrove Logistics, L.L.C.", "WOODGROVE LOGISTICS, L.L.C."],
                reject=["Woodgrove Freight LLC"],
            ),
        )
    if kind == "the":
        return (
            "The Adventure Works Corporation",
            Trap(
                gold="Adventure Works Corporation",
                kind=kind,
                accept=["The Adventure Works Corporation", "Adventure Works Corp.", "Adventure Works"],
                reject=["Adventure Travel Corporation"],
            ),
        )
    if kind == "ampersand":
        # written out by the labeler, abbreviated and symbolised by the document
        return (
            "Smith & Hart Ltd.",
            Trap(
                gold="Smith and Hart Limited",
                kind=kind,
                accept=["Smith & Hart Ltd.", "Smith and Hart Ltd", "Smith & Hart"],
                reject=["Smith & Hartley Ltd."],
            ),
        )
    return (
        "Litware Analytics, Inc.",
        Trap(
            gold="Litware Analytics, Inc.",
            kind="plain",
            accept=["Litware Analytics, Inc.", "Litware Analytics Inc", "Litware Analytics"],
            reject=["Litware Holdings, Inc."],
        ),
    )


def _value(kind: str, index: int) -> tuple[str, Trap]:
    """Return the consideration clause and the trap in it."""
    if kind == "two_figures":
        main, fee = (index + 3) * 200_000, 50_000 * (index % 3 + 1)
        return (
            f"The total contract value is ${main:,}, exclusive of an annual support fee of ${fee:,}.",
            Trap(
                gold={"value": float(main), "unit": "USD"},
                kind=kind,
                accept=[{"value": float(main), "unit": "USD"}, f"${main:,}"],
                reject=[{"value": float(fee), "unit": "USD"}, {"value": float(main + fee), "unit": "USD"}],
            ),
        )
    if kind == "words":
        return (
            "The total contract value is one million two hundred fifty thousand US dollars (US$1,250,000).",
            Trap(
                gold={"value": 1_250_000.0, "unit": "USD"},
                kind=kind,
                accept=[{"value": 1_250_000.0, "unit": "USD"}, "$1,250,000", "$1.25 million"],
                reject=[{"value": 1_250.0, "unit": "USD"}],
            ),
        )
    if kind == "magnitude":
        amount = (index + 4) * 250_000
        return (
            f"The total contract value is USD {amount / 1_000_000:g} million.",
            Trap(
                gold={"value": float(amount), "unit": "USD"},
                kind=kind,
                # the model answered in this last form at medium effort; it is right
                accept=[
                    {"value": float(amount), "unit": "USD"},
                    f"${amount / 1_000_000:g}M",
                    {"value": amount / 1_000_000, "unit": "million USD"},
                ],
                reject=[{"value": amount / 1_000_000, "unit": "USD"}],
            ),
        )
    if kind == "per_year":
        annual = (index % 4 + 2) * 100_000
        return (
            f"The Customer shall pay ${annual:,} per year for a term of four years, for a total "
            f"contract value of ${annual * 4:,}.",
            Trap(
                gold={"value": float(annual * 4), "unit": "USD"},
                kind=kind,
                accept=[{"value": float(annual * 4), "unit": "USD"}],
                reject=[{"value": float(annual), "unit": "USD"}],
            ),
        )
    if kind == "dollars_word":
        amount = (index + 2) * 150_000
        return (
            f"The total contract value is {amount:,} dollars, payable in quarterly instalments.",
            Trap(
                gold={"value": float(amount), "unit": "USD"},
                kind=kind,
                # a model that reads "dollars" literally may say so; it is still US dollars
                accept=[
                    {"value": float(amount), "unit": "USD"},
                    {"value": float(amount), "unit": "dollars"},
                    {"value": float(amount), "unit": "US dollars"},
                ],
                reject=[{"value": float(amount), "unit": "EUR"}],
            ),
        )
    amount = (index + 1) * 175_000
    return (
        f"The total contract value is ${amount:,}.",
        Trap(
            gold={"value": float(amount), "unit": "USD"},
            kind="plain",
            accept=[{"value": float(amount), "unit": "USD"}],
            reject=[{"value": float(amount) * 1.01, "unit": "USD"}],
        ),
    )


def _dates(kind: str, index: int) -> tuple[str, str | None, Trap]:
    """Return the preamble date, any separate effective-date sentence, and the trap."""
    if kind == "signed_vs_effective":
        month = index % 10 + 1
        return (
            _date_text(2024, month, 1),
            f'This Agreement shall become effective on {_date_text(2024, month + 1, 15)} (the "Effective Date").',
            Trap(
                gold=_iso(2024, month + 1, 15),
                kind=kind,
                accept=[_iso(2024, month + 1, 15), _date_text(2024, month + 1, 15)],
                reject=[_iso(2024, month, 1)],
            ),
        )
    if kind == "relative":
        month, day = index % 11 + 1, 3 + index % 20
        return (
            _date_text(2024, month, day),
            "This Agreement becomes effective on the first day of the calendar month following the "
            "date of its execution.",
            Trap(
                gold=_iso(2024, month + 1, 1),
                kind=kind,
                accept=[_iso(2024, month + 1, 1)],
                reject=[_iso(2024, month, day)],
            ),
        )
    if kind == "month_only":
        month = index % 12 + 1
        return (
            f"{MONTHS[month - 1]} 2025",
            None,
            Trap(
                # the document never names a day, so neither does the gold
                gold=f"2025-{month:02d}",
                kind=kind,
                accept=[f"2025-{month:02d}", f"{MONTHS[month - 1]} 2025"],
                reject=[f"2024-{month:02d}"],
            ),
        )
    if kind == "european":
        month, day = index % 12 + 1, 13 + index % 15
        written = f"{day:02d}/{month:02d}/2024 ({day} {MONTHS[month - 1]} 2024)"
        return (
            written,
            None,
            Trap(
                gold=_iso(2024, month, day),
                kind=kind,
                accept=[_iso(2024, month, day), f"{day} {MONTHS[month - 1]} 2024"],
                reject=[f"2024-{month:02d}"],
            ),
        )
    month, day = index % 12 + 1, 1 + (index * 3) % 27
    return (
        _date_text(2024, month, day),
        None,
        Trap(
            gold=_iso(2024, month, day),
            kind="plain",
            accept=[_iso(2024, month, day)],
            reject=["2024"],
        ),
    )


def _signatories(kind: str, index: int) -> tuple[list[str], Trap]:
    """Return the signature lines and the trap in them."""
    first, last = PEOPLE[index % len(PEOPLE)]
    other_first, other_last = PEOPLE[(index + 3) % len(PEOPLE)]
    name, other = f"{first} {last}", f"{other_first} {other_last}"
    if kind == "notary":
        return (
            [f"Signed: {name}, Chief Executive Officer", f"Subscribed and sworn before me: {other}, Notary Public"],
            Trap(gold=[name], kind=kind, accept=[[name]], reject=[[name, other]]),
        )
    if kind == "titles":
        # the labeler drops the honorific and the degree; a literal model may not
        return (
            [f"Signed: Dr. {name}, PhD, Chief Scientific Officer"],
            Trap(gold=[name], kind=kind, accept=[[name], [f"Dr. {name}"], [f"Dr. {name}, PhD"]], reject=[[other]]),
        )
    if kind == "middle_initial":
        return (
            [f"Signed: {first} J. {last}", f"Signed: {other}"],
            Trap(
                gold=[name, other],
                kind=kind,
                accept=[[name, other], [f"{first} J. {last}", other]],
                reject=[[name]],
            ),
        )
    if kind == "both_parties":
        return (
            [f"For the Customer: {name}", f"For the Provider: {other}"],
            Trap(gold=[name, other], kind=kind, accept=[[name, other], [other, name]], reject=[[name]]),
        )
    return (
        [f"Signed: {name}"],
        Trap(gold=[name], kind="plain", accept=[[name]], reject=[[other]]),
    )


def _products(kind: str) -> tuple[str, Trap]:
    """Return the scope clause and the trap in it."""
    if kind == "exclusion":
        return (
            "The Provider shall supply hardware and software to the Customer. Support services are "
            "expressly excluded from the scope of this Agreement.",
            Trap(
                gold=["hardware", "software"],
                kind=kind,
                accept=[["hardware", "software"]],
                reject=[["hardware", "software", "support"]],
            ),
        )
    if kind == "synonyms":
        return (
            "The Provider shall deliver professional services together with ongoing maintenance and "
            "technical support.",
            Trap(gold=["services", "support"], kind=kind, accept=[["services", "support"]], reject=[["services"]]),
        )
    if kind == "future_option":
        return (
            "The Provider shall supply software. The Customer may purchase hardware at a later date "
            "under a separate order form.",
            Trap(gold=["software"], kind=kind, accept=[["software"]], reject=[["software", "hardware"]]),
        )
    return (
        "The Provider shall supply the following product lines to the Customer: hardware, services.",
        Trap(gold=["hardware", "services"], kind="plain", accept=[["hardware", "services"]], reject=[["hardware"]]),
    )


def _contract_number(kind: str, index: int) -> tuple[str, Trap]:
    """Return the contract-number line and the trap in it."""
    number = f"MSA-2024-{1000 + index * 7}"
    if kind == "po_and_superseded":
        return (
            f"Contract Number: {number}. This Agreement references Purchase Order PO-{88000 + index} and "
            f"supersedes Master Services Agreement MSA-2023-{900 + index} in its entirety.",
            Trap(gold=number, kind=kind, accept=[number], reject=[f"PO-{88000 + index}", f"MSA-2023-{900 + index}"]),
        )
    if kind == "en_dash":
        # the document uses en dashes; the labeler typed ordinary hyphens
        return (
            f"Contract No. {number.replace('-', '–')}",
            Trap(gold=number, kind=kind, accept=[number, number.replace("-", "–")], reject=[f"MSA-2024-{999}"]),
        )
    return (
        f"Contract Number: {number}",
        Trap(gold=number, kind="plain", accept=[number], reject=[f"MSA-2023-{1000 + index * 7}"]),
    )


def _arbitration(kind: str) -> tuple[str | None, Trap]:
    """Return the dispute clause, if any, and the trap in it."""
    clauses = {
        "binding": (
            "Any dispute arising under this Agreement shall be finally resolved by binding arbitration.",
            True,
        ),
        "mediation": (
            "The parties shall first attempt to resolve any dispute through non-binding mediation. If "
            "mediation fails, either party may bring the dispute before the state or federal courts.",
            False,
        ),
        "optional": (
            "Either party may, at its option, elect to submit a dispute to arbitration; absent such an "
            "election, disputes shall be resolved in the courts.",
            False,
        ),
        "nonbinding": (
            "Disputes shall be submitted to non-binding arbitration, the outcome of which either party "
            "may reject in favour of litigation.",
            False,
        ),
    }
    if kind == "silent":
        return None, Trap(gold=None, kind=kind, accept=[None], reject=[True])
    text, gold = clauses[kind]
    return text, Trap(gold=gold, kind=kind, accept=[gold], reject=[not gold])


def _governing_law(state: str, index: int) -> tuple[str, str, str, str]:
    """Return the governing-law clause, the sentence the span covers, the variant, and its distractor.

    The distractor is the state a careless reading would pick: the venue for a clause that
    names courts in one state and law in another, Delaware for a carve-out that sends the
    arbitration provisions there, and otherwise Delaware from the Provider's address.
    """
    name = STATES[state]
    main = f"This Agreement shall be governed by and construed in accordance with the laws of the State of {name}."
    fallback_distractor = "CA" if state == "DE" else "DE"
    if index in FAA_DOCS:
        extra = (
            "Notwithstanding the foregoing, the arbitration provisions of this Agreement shall be governed "
            "by the Federal Arbitration Act and, to the extent not preempted, the laws of the State of Delaware."
        )
        return f"{main} {extra}", main, "faa_and_delaware", "DE"
    if index in VENUE_DOCS:
        venue = (
            "The parties consent to the exclusive jurisdiction of the state and federal courts located in "
            "San Francisco, California."
        )
        return f"{venue} {main}", main, "california_venue", "CA"
    if index in SEVERABILITY_DOCS:
        extra = (
            "If any provision of this Agreement is held unenforceable, the remaining provisions shall "
            "remain in full force and effect."
        )
        return f"{main} {extra}", main, "trailing_severability", fallback_distractor
    plain = f"{main[:-1]}, without regard to its conflict of laws principles."
    return plain, plain, "plain", fallback_distractor


def build_documents(count: int = 24) -> list[Document]:
    """Build the adversarial corpus specification.

    :param count: How many documents to build; the state plan covers 24
    :returns: One document per index, with its traps
    """
    if count > len(STATE_PLAN):
        raise ValueError(f"the state plan covers {len(STATE_PLAN)} documents, not {count}")
    documents = []
    for index in range(count):
        state = STATE_PLAN[index]
        arbitration_kind = "binding" if index in FAA_DOCS else ARBITRATION_CYCLE[index % 5]
        counterparty_text, counterparty = _counterparty(COUNTERPARTY_CYCLE[index % 7])
        value_text, value = _value(VALUE_CYCLE[index % 6], index)
        preamble_date, effective_sentence, date = _dates(DATE_CYCLE[index % 5], index)
        signature_lines, signatories = _signatories(SIGNATORY_CYCLE[index % 5], index)
        products_text, products = _products(PRODUCT_CYCLE[index % 4])
        number_text, number = _contract_number(NUMBER_CYCLE[index % 3], index)
        dispute_text, arbitration = _arbitration(arbitration_kind)
        governing_text, governing_sentence, governing_kind, state_distractor = _governing_law(state, index)

        paragraphs: list[tuple[str, str]] = [
            ("Title", "MASTER SERVICES AGREEMENT"),
            ("Body", number_text),
            (
                "Body",
                f"This Master Services Agreement is entered into as of {preamble_date}, by and between "
                f'{PROVIDER}, a corporation with offices in Wilmington, Delaware ("Provider"), and '
                f'{counterparty_text} ("Customer").',
            ),
        ]
        if effective_sentence:
            paragraphs.append(("Body", effective_sentence))
        paragraphs += [
            ("Heading", "1. Scope"),
            ("Body", products_text),
            ("Heading", "2. Consideration"),
            ("Body", value_text),
        ]
        if dispute_text:
            paragraphs += [("Heading", "3. Dispute Resolution"), ("Body", dispute_text)]
        paragraphs += [
            ("Heading", "4. Governing Law"),
            ("Body", governing_text),
            ("Heading", "5. Signatures"),
            ("Signatures", "\n".join(signature_lines)),
        ]
        documents.append(
            Document(
                doc_id=f"adv_{index:02d}",
                paragraphs=paragraphs,
                governing_sentence=governing_sentence,
                traps={
                    "has_arbitration_clause": arbitration,
                    "filing_state": Trap(
                        gold=state,
                        kind=governing_kind,
                        accept=[state, STATES[state]],
                        reject=[state_distractor],
                    ),
                    "covered_products": products,
                    "contract_number": number,
                    "counterparty": counterparty,
                    "signatories": signatories,
                    "contract_value": value,
                    "effective_date": date,
                },
            )
        )
    return documents


def render_pdf(document: Document, pdf_dir: Path) -> Path:
    """Render one document, escaping text so ampersands and angle brackets survive."""
    styles = getSampleStyleSheet()
    lookup = {
        "Title": ParagraphStyle("AdvTitle", parent=styles["Title"], fontSize=14, spaceAfter=12),
        "Heading": ParagraphStyle("AdvHeading", parent=styles["Heading2"], fontSize=11, spaceBefore=10),
        "Body": ParagraphStyle("AdvBody", parent=styles["BodyText"], fontSize=10, leading=14),
    }
    lookup["Signatures"] = lookup["Body"]
    path = pdf_dir / f"{document.doc_id}.pdf"
    template = SimpleDocTemplate(str(path), pagesize=LETTER, title=document.doc_id)
    flowables: list[Any] = []
    for style, text in document.paragraphs:
        markup = "<br/>".join(escape(line) for line in text.split("\n"))
        flowables += [Paragraph(markup, lookup[style]), Spacer(1, 4)]
    template.build(flowables)
    return path


def locate(sentence: str, text: str) -> dict[str, int] | None:
    """Find a sentence in extracted text, tolerating the line breaks extraction introduces.

    :param sentence: The sentence as it was written into the document
    :param text: The cached extracted text the program is shown
    :returns: Character offsets into that text, or None if the sentence is not there
    """
    pattern = r"\s+".join(re.escape(word) for word in sentence.split())
    match = re.search(pattern, text)
    return None if match is None else {"start": match.start(), "end": match.end()}


def generate(project_dir: Path, count: int = 24) -> list[Document]:
    """Render the corpus, extract it, and write gold labels including span offsets.

    Span offsets are computed against the cached text rather than the source, because a
    character offset only means anything relative to the text the program is shown.

    :param project_dir: The adversarial project's root
    :param count: How many documents to generate
    :returns: The documents, with their traps
    """
    data = project_dir / "data"
    pdf_dir = data / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    documents = build_documents(count)
    for document in documents:
        render_pdf(document, pdf_dir)
    extracted = extract_corpus(
        pdf_dir, data / "text", data / "extraction_manifest.csv", ExtractionConfig(transcription=TranscriptionConfig(mode="off")), force=True
    )
    texts = {item.doc_id: item.text for item in extracted}

    records, missing = [], []
    for document in documents:
        span = locate(document.governing_sentence, texts[document.doc_id])
        if span is None:
            missing.append(document.doc_id)
        labels = document.labels()
        labels["governing_law_span"] = span
        records.append(LabelRecord(doc_id=document.doc_id, labels=labels, labeling_mode="corrected"))
    if missing:
        raise RuntimeError(f"governing-law sentence not found in the extracted text of: {', '.join(missing)}")
    write_labels(data / "labels.jsonl", records)
    logger.info("wrote %d adversarial documents to %s", len(documents), data)
    return documents


def main() -> None:
    """Generate the adversarial corpus into a project directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="The adversarial project directory")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    documents = generate(args.project, args.count)
    print(f"Generated {len(documents)} adversarial documents into {args.project / 'data'}.")


if __name__ == "__main__":
    main()
