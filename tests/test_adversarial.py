"""The adversarial corpus's offline oracle.

Every trap lists answers a correct model could give, which the matcher must accept, and the
distractor it must refuse. This is free to run and caught three real normalizer bugs before
any API call was made: dotted legal suffixes, titles on people's names, and currency written
as a word -- plus a fourth, hidden one, where a typed model output skipped unit
normalization entirely.

It is still a set of guesses about model output. The live run is what checks them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import load_example

from doc_harness.dataset import LabelRecord
from doc_harness.metric import build_metric
from doc_harness.registry import Registry, TaskType
from doc_harness.splits import class_supports
from doc_harness.values import Quantity

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "adversarial"
_adversarial = load_example("adversarial")
build_documents, locate = _adversarial.build_documents, _adversarial.locate

# a known limitation, recorded rather than hidden: see test_similar_company_names_are_a_known_limit
KNOWN_FALSE_ACCEPTS = {("counterparty", "ampersand", repr("Smith & Hartley Ltd."))}


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.from_yaml(EXAMPLES / "tasks.yaml")


@pytest.fixture(scope="module")
def documents() -> list[Any]:
    return build_documents()


def _variants(form: Any) -> list[Any]:
    """Return a form as written, plus as the typed object a model actually returns."""
    if isinstance(form, dict) and set(form) == {"value", "unit"}:
        return [form, Quantity(**form)]
    return [form]


def _verdicts(registry: Registry, documents: list[Any], side: str) -> list[str]:
    """Return a line for every form the matcher got wrong on one side of the oracle."""
    metric = build_metric(registry)
    wrong: list[str] = []
    for document in documents:
        labels = document.labels()
        for task_id, trap in document.traps.items():
            task = registry.by_id(task_id)
            for form in getattr(trap, side):
                # forms can be lists or dicts, which are unhashable; compare by repr
                if side == "reject" and (task_id, trap.kind, repr(form)) in KNOWN_FALSE_ACCEPTS:
                    continue
                for variant in _variants(form):
                    result = metric.score_task(task, labels, {**labels, task_id: variant})
                    if result.correct != (side == "accept"):
                        wrong.append(f"{document.doc_id} {task_id}/{trap.kind}: {variant!r} -> {result.detail}")
    return wrong


def test_every_correct_answer_is_accepted(registry: Registry, documents: list[Any]) -> None:
    """A right answer scored as wrong makes a good prompt look broken."""
    wrong = _verdicts(registry, documents, "accept")
    assert not wrong, "correct answers rejected:\n" + "\n".join(wrong)


def test_every_distractor_is_rejected(registry: Registry, documents: list[Any]) -> None:
    """A wrong answer scored as right inflates every number built on it."""
    wrong = _verdicts(registry, documents, "reject")
    assert not wrong, "distractors accepted:\n" + "\n".join(wrong)


def test_similar_company_names_are_a_known_limit(registry: Registry, documents: list[Any]) -> None:
    """Character similarity cannot tell a typo from a different name.

    "Smith & Hartley Ltd." scores 0.903 against "Smith and Hart Limited" -- a different company,
    just over a theta of 0.90. No threshold separates "Hart"/"Harte" (a typo) from
    "Hart"/"Hartley" (another firm), which is why BUILD.md makes borderline fuzzy matches a
    human decision. This test pins the limitation so a change to it is noticed, and checks
    that adjudication lists the case rather than letting it pass silently.
    """
    from doc_harness.adjudicate import adjudicate

    document = next(d for d in documents if d.traps["counterparty"].kind == "ampersand")
    labels = {"doc_id": document.doc_id, **document.labels()}
    task = registry.by_id("counterparty")
    result = build_metric(registry).score_task(task, labels, {**labels, "counterparty": "Smith & Hartley Ltd."})
    assert result.correct, "if this now rejects, the limitation is fixed: remove it from KNOWN_FALSE_ACCEPTS"
    assert result.measure is not None and result.measure < 0.91

    report = adjudicate(
        registry, build_metric(registry), [labels], [{**labels, "counterparty": "Smith & Hartley Ltd."}]
    )
    assert any(decision.task_id == "counterparty" for decision in report.decisions), "the close call went unflagged"


def test_every_task_type_is_trapped(registry: Registry, documents: list[Any]) -> None:
    trapped = {task_id for document in documents for task_id in document.traps}
    untrapped = {task.id for task in registry} - trapped - {"governing_law_span"}
    assert not untrapped, f"tasks with no traps: {untrapped}"
    assert {str(task.type) for task in registry} == set(TaskType)


def test_every_trap_appears_more_than_once(documents: list[Any]) -> None:
    """One document per trap makes a single fluke look like a finding."""
    counts: dict[tuple[str, str], int] = {}
    for document in documents:
        for task_id, trap in document.traps.items():
            counts[(task_id, trap.kind)] = counts.get((task_id, trap.kind), 0) + 1
    rare = {key: count for key, count in counts.items() if count < 2}
    assert not rare, rare


def test_texas_falls_under_the_support_floor(registry: Registry, documents: list[Any]) -> None:
    """So that make-splits' support-floor stop fires on this corpus, which it never has."""
    records = [LabelRecord(doc_id=d.doc_id, labels=d.labels()) for d in documents]
    supports = class_supports(registry, records)["filing_state"]
    assert supports["TX"].count == 3
    assert all(supports[state].count >= 4 for state in ("CA", "NY", "DE"))


def test_the_venue_trap_names_the_wrong_state(documents: list[Any]) -> None:
    venue = [d for d in documents if d.traps["filing_state"].kind == "california_venue"]
    assert venue and all(d.traps["filing_state"].gold == "TX" for d in venue)
    assert all(d.traps["filing_state"].reject == ["CA"] for d in venue)


def test_locate_tolerates_extraction_line_breaks() -> None:
    sentence = "This Agreement shall be governed by the laws of Texas."
    text = "Intro.\nThis Agreement shall be governed\nby the laws of Texas. More text."
    span = locate(sentence, text)
    assert span is not None
    assert text[span["start"] : span["end"]].replace("\n", " ") == sentence
    assert locate("Not present at all.", text) is None
