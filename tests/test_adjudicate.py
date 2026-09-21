"""Adjudication: finding the decisions a threshold actually made."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc_harness.adjudicate import adjudicate, to_markdown, write_adjudication
from doc_harness.hooks import get_matcher
from doc_harness.metric import build_metric
from doc_harness.registry import Registry

PERFECT: dict[str, Any] = {
    "has_arbitration_clause": True,
    "filing_state": "CA",
    "covered_products": ["hardware"],
    "contract_number": "A-1",
    "counterparty": "Acme Holdings Inc",
    "signatories": ["Jane Doe"],
    "contract_value": {"value": 1000000.0, "unit": "USD"},
    "effective_date": "2024-06-15",
    "governing_law_span": {"start": 100, "end": 200},
}


@pytest.fixture
def registry(fixtures_dir: Path) -> Registry:
    return Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")


def _run(registry: Registry, **overrides: Any) -> Any:
    gold = {"doc_id": "d01", **PERFECT}
    pred = {**PERFECT, **overrides}
    return adjudicate(registry, build_metric(registry), [gold], [pred], run_id="test")


def _reasons(report: Any) -> dict[str, str]:
    return {decision.task_id: decision.reason for decision in report.decisions}


def test_a_clean_run_has_nothing_to_adjudicate(registry: Registry) -> None:
    report = _run(registry)
    assert report.decisions == []
    assert report.errors == []


def test_theta_overruling_strict_is_a_close_call(registry: Registry) -> None:
    """Word order defeats strict equality; token-sort similarity still accepts it."""
    report = _run(registry, counterparty="Holdings Acme Inc")
    assert "theta overruled strict equality" in _reasons(report)["counterparty"]
    decision = next(d for d in report.decisions if d.task_id == "counterparty")
    assert decision.verdict == "accepted"


def test_an_exact_fuzzy_match_is_not_a_close_call(registry: Registry) -> None:
    """A suffix that normalizes away never involved theta at all."""
    report = _run(registry, counterparty="Acme Holdings, LLC")
    assert "counterparty" not in _reasons(report)


def test_differing_units_are_listed(registry: Registry) -> None:
    report = _run(registry, contract_value={"value": 1000000.0, "unit": "EUR"})
    assert _reasons(report)["contract_value"] == "units differ"


def test_a_number_near_the_tolerance_is_listed(registry: Registry) -> None:
    """The fixture allows 1%; half a percent off is half the allowance."""
    report = _run(registry, contract_value={"value": 1005000.0, "unit": "USD"})
    assert "0.50x the tolerance" in _reasons(report)["contract_value"]


def test_a_number_far_outside_the_tolerance_is_just_an_error(registry: Registry) -> None:
    report = _run(registry, contract_value={"value": 2000000.0, "unit": "USD"})
    assert "contract_value" not in _reasons(report)
    assert any(error.task_id == "contract_value" for error in report.errors)


def test_granularity_mismatch_is_listed(registry: Registry) -> None:
    report = _run(registry, effective_date="2024-06")
    assert "granularity differs: gold day, predicted month" in _reasons(report)["effective_date"]


def test_a_span_near_its_threshold_is_listed(registry: Registry) -> None:
    """Gold 100-200 against 150-250: 50 shared of 150, an overlap of one third."""
    report = _run(registry, governing_law_span={"start": 148, "end": 200})
    assert "against a threshold of 0.50" in _reasons(report)["governing_law_span"]


def test_a_list_pair_near_theta_is_listed(registry: Registry) -> None:
    """ "jane doeson" scores 0.842 against "jane doe": just under a theta of 0.85."""
    report = _run(registry, signatories=["Jane Doeson"])
    assert "0.842 against a threshold of 0.85" in _reasons(report)["signatories"]


def test_classification_errors_are_errors_not_close_calls(registry: Registry) -> None:
    """Enum tasks have no threshold, so a wrong class is simply wrong."""
    report = _run(registry, filing_state="NY")
    assert "filing_state" not in _reasons(report)
    assert [error.task_id for error in report.errors] == ["filing_state"]


def test_report_shows_raw_and_normalized_side_by_side(registry: Registry) -> None:
    """The arrow is where a normalizer bug becomes visible."""
    report = _run(registry, counterparty="Holdings Acme Inc")
    markdown = to_markdown(report)
    assert "Holdings Acme Inc → `holdings acme`" in markdown
    assert "for humans only" in markdown


def test_report_escapes_table_pipes(registry: Registry, tmp_path: Path) -> None:
    report = _run(registry, contract_number="A|1")
    path = tmp_path / "adjudication.md"
    write_adjudication(path, report)
    assert "A\\|1" in path.read_text(encoding="utf-8")


def test_mismatched_lengths_fail_loudly(registry: Registry) -> None:
    with pytest.raises(ValueError, match="cannot adjudicate"):
        adjudicate(registry, build_metric(registry), [PERFECT], [])


@pytest.mark.parametrize(
    "matcher,gold,pred,params,measure",
    [
        ("entity_name", "acme holdings", "acme holdings", {"normalizer": "entity_name", "theta": 0.9}, 1.0),
        ("span", {"start": 0, "end": 100}, {"start": 0, "end": 50}, {"overlap_threshold": 0.5}, 0.5),
        ("numeric", 100, 105, {"normalizer": "numeric", "tolerance": 10, "tolerance_kind": "absolute"}, 0.5),
    ],
)
def test_matchers_record_how_close_the_call_was(
    matcher: str, gold: Any, pred: Any, params: dict[str, Any], measure: float
) -> None:
    result = get_matcher(matcher)(gold, pred, params)
    assert result.measure == pytest.approx(measure)
    assert result.threshold is not None


def test_a_list_accepted_only_by_theta_is_listed_however_far_from_it(registry: Registry) -> None:
    """Found by the adversarial run: "mei j tanaka" was accepted for "mei tanaka" at 0.909.

    That sits 0.059 from a theta of 0.85, outside the proximity margin, so it went unflagged --
    yet strict equality would have refused it, which makes it exactly a threshold decision.
    """
    report = _run(registry, signatories=["Jane Q. Doe"])
    decision = next(d for d in report.decisions if d.task_id == "signatories")
    assert decision.reason == "theta overruled strict equality"
    assert decision.verdict == "accepted"


def test_an_identical_list_is_not_a_close_call(registry: Registry) -> None:
    report = _run(registry, signatories=["JANE DOE"])
    assert "signatories" not in _reasons(report)
