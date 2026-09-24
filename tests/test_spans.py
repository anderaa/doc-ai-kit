"""Spans by quoted text: the model quotes, the package finds the offsets.

A model asked for character offsets has to count characters. On the adversarial corpus it
found the right sentence and placed it 84 characters early, and counting only gets harder
as documents grow.
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from doc_ai_kit.dataset import DatasetError, LabelRecord, build_examples
from doc_ai_kit.metric import build_metric
from doc_ai_kit.normalize import span as normalize_span
from doc_ai_kit.program import build_program
from doc_ai_kit.registry import SPAN_QUOTE_INSTRUCTION, Registry
from doc_ai_kit.spans import locate_quote
from doc_ai_kit.values import QuotedSpan, Span

CLAUSE = "This Agreement shall be governed by and construed in accordance with the laws of the State of\nCalifornia."
DOCUMENT = "4. Governing Law\n" + CLAUSE + " The parties agree to the “exclusive” venue – San Francisco… only."
CLAUSE_START = DOCUMENT.index(CLAUSE)
CLAUSE_END = CLAUSE_START + len(CLAUSE)


@pytest.fixture
def registry() -> Registry:
    return Registry.from_mapping({"tasks": [{"id": "clause", "type": "span", "question": "The governing-law clause."}]})


# --- finding the quote ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,quote",
    [
        ("exact, line break and all", CLAUSE),
        ("line break flattened", CLAUSE.replace("\n", " ")),
        ("wrapped in quotation marks", '"' + CLAUSE.replace("\n", " ") + '"'),
        ("capitalised differently", CLAUSE.replace("\n", " ").upper()),
        ("surrounded by whitespace", "  " + CLAUSE + "\n"),
    ],
)
def test_formatting_differences_land_on_the_exact_offsets(label: str, quote: str) -> None:
    span = locate_quote(quote, DOCUMENT)
    assert span is not None, label
    assert (span.start, span.end) == (CLAUSE_START, CLAUSE_END), label


def test_a_partial_quote_marked_with_an_ellipsis_lands_on_the_excerpt() -> None:
    span = locate_quote("...shall be governed by and construed", DOCUMENT)
    assert span is not None
    assert DOCUMENT[span.start : span.end] == "shall be governed by and construed"


def test_typographic_substitutes_map_back_to_the_original_characters() -> None:
    """Folding changes length ("…" is three characters folded); offsets must still be the source's."""
    span = locate_quote('the "exclusive" venue - San Francisco... only.', DOCUMENT)
    assert span is not None
    assert DOCUMENT[span.start : span.end] == "the “exclusive” venue – San Francisco… only."
    assert span.text == DOCUMENT[span.start : span.end]


@pytest.mark.parametrize(
    "quote",
    [
        "This contract is governed by California law.",
        "This Agreement shall be governed by the laws of the State of Texas.",
        "",
        "   ",
        '""',
    ],
)
def test_a_paraphrase_or_an_absent_passage_is_not_found(quote: str) -> None:
    """A paraphrase is not a quote: stretching it to fit would score a wrong answer as right."""
    assert locate_quote(quote, DOCUMENT) is None


# --- the program ----------------------------------------------------------------------------


def test_the_model_is_asked_for_a_quote(registry: Registry) -> None:
    field = registry.signature_for("all").output_fields["clause"]
    assert field.annotation == str | None
    assert SPAN_QUOTE_INSTRUCTION in field.json_schema_extra["desc"]


def _predict(registry: Registry, answer: Any) -> Any:
    with dspy.context(lm=DummyLM([{"clause": answer}])):
        return build_program(registry)(document=DOCUMENT)


def test_a_quote_becomes_offsets(registry: Registry) -> None:
    prediction = _predict(registry, CLAUSE.replace("\n", " "))
    assert isinstance(prediction.clause, Span)
    assert (prediction.clause.start, prediction.clause.end) == (CLAUSE_START, CLAUSE_END)


def test_a_paraphrase_is_scored_wrong_not_as_an_abstention(registry: Registry) -> None:
    prediction = _predict(registry, "This contract is governed by California law.")
    assert isinstance(prediction.clause, str)
    gold = {"doc_id": "d0", "clause": {"start": CLAUSE_START, "end": CLAUSE_END}}
    result = build_metric(registry).score_task(registry.by_id("clause"), gold, prediction)
    assert not result.correct
    assert (result.fp, result.fn) == (1, 1)


def test_an_explicit_null_is_still_an_abstention(registry: Registry) -> None:
    assert _predict(registry, "null").clause is None


# --- gold spans as quotes -------------------------------------------------------------------


def _gold_example(registry: Registry, start: int = CLAUSE_START, end: int = CLAUSE_END) -> Any:
    records = [LabelRecord(doc_id="d0", labels={"clause": {"start": start, "end": end}})]
    return build_examples(records, {"d0": DOCUMENT}, registry=registry)[0]


def test_a_gold_span_is_shown_as_the_passage_it_covers(registry: Registry) -> None:
    """A labeled demonstration must show a quote -- not the offsets the model is told never to give."""
    gold = _gold_example(registry).clause
    assert isinstance(gold, QuotedSpan)
    assert str(gold) == CLAUSE
    assert "start" not in str(gold)


def test_a_gold_quote_is_still_scored_on_its_offsets(registry: Registry) -> None:
    example = _gold_example(registry)
    normalized = normalize_span(example.clause, {})
    assert isinstance(normalized, Span)
    assert (normalized.start, normalized.end) == (CLAUSE_START, CLAUSE_END)

    prediction = {"clause": Span(start=CLAUSE_START, end=CLAUSE_END)}
    assert build_metric(registry).score_example(example, prediction).all_correct


def test_a_gold_span_outside_the_text_fails_loudly(registry: Registry) -> None:
    """Offsets past the end mean the label was made against a different extraction."""
    with pytest.raises(DatasetError, match="falls outside the cached text"):
        _gold_example(registry, start=10, end=len(DOCUMENT) + 50)


def test_without_a_registry_gold_spans_are_untouched() -> None:
    records = [LabelRecord(doc_id="d0", labels={"clause": {"start": 1, "end": 5}})]
    example = build_examples(records, {"d0": DOCUMENT})[0]
    assert example.clause == {"start": 1, "end": 5}
