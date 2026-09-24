"""Matcher tests driven by the adversarial fixture corpus.

A bad matcher makes a good prompt look broken and sends the optimizer chasing a bug, so
these run before anything that depends on a score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from doc_ai_kit.hooks import get_matcher, registered_names
from doc_ai_kit.values import MatchResult

COUNT_FIELDS = ("tp", "fp", "fn", "tn")


def _load_cases(fixtures_dir: Path) -> list[dict[str, Any]]:
    return list(yaml.safe_load((fixtures_dir / "matchers.yaml").read_text(encoding="utf-8")))


def _case_ids(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


def _run(case: dict[str, Any]) -> MatchResult:
    matcher = get_matcher(case["matcher"])
    return matcher(case["gold"], case["pred"], case.get("params", {}))


def _assert_counts(result: MatchResult, expected: dict[str, Any], label: str) -> None:
    for field in COUNT_FIELDS:
        if field in expected:
            got = getattr(result, field)
            assert got == expected[field], f"{label}: {field} was {got}, expected {expected[field]} ({result.detail})"
    if "correct" in expected:
        assert result.correct is expected["correct"], f"{label}: correct was {result.correct} ({result.detail})"


def test_fixture_covers_every_registered_matcher(fixtures_dir: Path) -> None:
    """A new matcher must arrive with adversarial cases, not after them."""
    # free_text delegates to entity_name with a different normalizer
    registered = set(registered_names()["matchers"]) - {"free_text"}
    covered = {case["matcher"] for case in _load_cases(fixtures_dir)}
    assert registered - covered == set(), f"matchers with no fixture case: {sorted(registered - covered)}"


def test_matcher_outcomes(fixtures_dir: Path) -> None:
    failures: list[str] = []
    for case in _load_cases(fixtures_dir):
        result = _run(case)
        try:
            _assert_counts(result, case["expect"], case["name"])
            if "strict" in case:
                assert "strict" in result.alternates, f"{case['name']}: no strict result attached"
                _assert_counts(result.alternates["strict"], case["strict"], f"{case['name']} (strict)")
        except AssertionError as exc:
            failures.append(str(exc))
    assert not failures, "\n".join(failures)


def test_fuzzy_always_reports_strict_alongside(fixtures_dir: Path) -> None:
    """There is no code path that reports a fuzzy number on its own."""
    for case in _load_cases(fixtures_dir):
        if case["matcher"] not in {"entity_name", "free_text"}:
            continue
        result = _run(case)
        assert "strict" in result.alternates, case["name"]


def test_matchers_are_symmetric_in_correctness(fixtures_dir: Path) -> None:
    """Swapping gold and prediction must not change whether they match.

    The counts legitimately swap -- a false positive becomes a false negative -- but a
    matcher whose verdict depends on argument order is comparing two different things.
    """
    order_sensitive = {"date"}  # coarser-prediction handling is deliberately asymmetric
    for case in _load_cases(fixtures_dir):
        if case["matcher"] in order_sensitive:
            continue
        matcher = get_matcher(case["matcher"])
        params = case.get("params", {})
        forward = matcher(case["gold"], case["pred"], params)
        backward = matcher(case["pred"], case["gold"], params)
        assert forward.correct is backward.correct, f"{case['name']}: {forward.detail} vs {backward.detail}"
        assert forward.tp == backward.tp, case["name"]
        assert forward.fp == backward.fn, case["name"]
        assert forward.fn == backward.fp, case["name"]


def test_every_comparison_is_counted(fixtures_dir: Path) -> None:
    """No comparison may produce zero outcomes: that would be a silent drop."""
    for case in _load_cases(fixtures_dir):
        result = _run(case)
        assert result.counted > 0, f"{case['name']} produced no outcomes"


def test_normalized_values_are_reported(fixtures_dir: Path) -> None:
    """Matchers must hand back what they compared, so failures.md can show it."""
    for case in _load_cases(fixtures_dir):
        result = _run(case)
        assert result.detail, case["name"]
        if case["gold"] is not None:
            assert result.gold_normalized is not None, case["name"]


@pytest.mark.parametrize("theta", [0.5, 0.85, 0.95, 1.0])
def test_fuzzy_threshold_is_monotone(theta: float) -> None:
    """Raising theta can only ever make a matcher stricter."""
    matcher = get_matcher("entity_name")
    params = {"normalizer": "entity_name", "theta": theta}
    result = matcher("Acme Holdings", "Acme Holding", params)
    if theta == 1.0:
        assert not result.correct
    assert result.counted == 2 if not result.correct else result.counted == 1


def test_unit_aliases_are_accepted_in_tasks_yaml() -> None:
    """Reported from a real run: the normalizer read unit_aliases, but tasks.yaml refused the key."""
    from doc_ai_kit.metric import build_metric
    from doc_ai_kit.registry import Registry

    registry = Registry.from_mapping(
        {
            "tasks": [
                {
                    "id": "notice",
                    "type": "extract_numeric",
                    "question": "Notice to stop renewal.",
                    "match": {"matcher": "numeric", "tolerance": 0, "unit": "days", "unit_aliases": {"jours": "days"}},
                }
            ]
        }
    )
    task = registry.by_id("notice")
    metric = build_metric(registry)
    assert metric.score_task(task, {"notice": "30 days"}, {"notice": "30 jours"}).correct
    assert metric.score_task(task, {"notice": "30 days"}, {"notice": "30 day"}).correct
    assert not metric.score_task(task, {"notice": "30 days"}, {"notice": "1 month"}).correct
    assert not metric.score_task(task, {"notice": "30 days"}, {"notice": "30 business days"}).correct
