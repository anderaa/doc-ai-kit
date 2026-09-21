"""Metric and evaluation tests against a toy split whose numbers were computed by hand.

Every expected value in this file was derived on paper from tests/fixtures/toy_examples.yaml
before the code ran. That is the point: an evaluation module verified against its own output
verifies nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from doc_harness.evaluate import (
    EvaluationResult,
    load_predictions,
    null_rates_from_metrics,
    score_split,
    write_run,
)
from doc_harness.metric import build_metric
from doc_harness.registry import Registry
from doc_harness.values import PartialDate, Quantity, Span

# support floors set low so the toy split's classes count as measurable
TOY_SUPPORT_FLOOR = 2
TOY_MEASURABLE_FLOOR = 2


@pytest.fixture
def toy(fixtures_dir: Path) -> tuple[Registry, list[dict[str, Any]], list[dict[str, Any]], EvaluationResult]:
    registry = Registry.from_yaml(fixtures_dir / "toy_tasks.yaml")
    raw = yaml.safe_load((fixtures_dir / "toy_examples.yaml").read_text(encoding="utf-8"))["examples"]
    golds = [{"doc_id": row["doc_id"], "document": f"document {row['doc_id']}", **row["gold"]} for row in raw]
    preds = [dict(row["pred"]) for row in raw]
    metric = build_metric(registry)
    result = score_split(
        registry,
        metric,
        golds,
        preds,
        support_floor=TOY_SUPPORT_FLOOR,
        measurable_floor=TOY_MEASURABLE_FLOOR,
    )
    return registry, golds, preds, result


def test_aggregate_is_the_hand_computed_mean(toy: Any) -> None:
    """Per-example aggregates: 1, 1, 1/3, 2/3, 1, 0, 1, 1/3, 1, 2/3 -> mean 0.70."""
    _registry, _golds, _preds, result = toy
    assert result.aggregate == pytest.approx(0.70)
    per_example = [score.aggregate for score in result.scores]
    assert per_example == pytest.approx([1.0, 1.0, 1 / 3, 2 / 3, 1.0, 0.0, 1.0, 1 / 3, 1.0, 2 / 3])


def test_binary_counts(toy: Any) -> None:
    """flag: 8 right, 2 wrong non-null answers, each counting as both FP and FN."""
    _registry, _golds, _preds, result = toy
    flag = result.tasks["flag"]
    assert (flag.tp, flag.fp, flag.fn, flag.tn) == (8, 2, 2, 0)
    assert flag.precision == pytest.approx(0.8)
    assert flag.recall == pytest.approx(0.8)
    assert flag.f1 == pytest.approx(0.8)
    assert flag.exact_match == pytest.approx(0.8)
    # the binary headline is F1 on the positive class
    assert flag.primary_metric == "f1_positive"
    assert flag.primary_value == pytest.approx(0.8)
    assert flag.classes["true"].support == 5
    assert flag.classes["true"].f1 == pytest.approx(0.8)
    assert flag.classes["false"].f1 == pytest.approx(0.8)


def test_multiclass_counts_and_abstention(toy: Any) -> None:
    """state: 5 TP, 4 FP (3 wrong values + 1 answer where gold is null), 3 FN, 1 TN."""
    _registry, _golds, _preds, result = toy
    state = result.tasks["state"]
    assert (state.tp, state.fp, state.fn, state.tn) == (5, 4, 3, 1)
    assert state.precision == pytest.approx(5 / 9)
    assert state.recall == pytest.approx(5 / 8)
    assert state.f1 == pytest.approx(2 * (5 / 9) * (5 / 8) / ((5 / 9) + (5 / 8)))
    assert state.exact_match == pytest.approx(0.6)


def test_multiclass_per_class_and_macro(toy: Any) -> None:
    """CA 2/2/1, NY 2/2/1, TX 1/0/1 as tp/fp/fn; macro-F1 = (4/7 + 4/7 + 2/3) / 3."""
    _registry, _golds, _preds, result = toy
    classes = result.tasks["state"].classes
    assert (classes["CA"].tp, classes["CA"].fp, classes["CA"].fn) == (2, 2, 1)
    assert (classes["NY"].tp, classes["NY"].fp, classes["NY"].fn) == (2, 2, 1)
    assert (classes["TX"].tp, classes["TX"].fp, classes["TX"].fn) == (1, 0, 1)
    assert classes["CA"].f1 == pytest.approx(4 / 7)
    assert classes["TX"].f1 == pytest.approx(2 / 3)
    assert classes["TX"].precision == pytest.approx(1.0)
    assert classes["TX"].recall == pytest.approx(0.5)
    assert result.tasks["state"].primary_metric == "macro_f1"
    assert result.tasks["state"].primary_value == pytest.approx((4 / 7 + 4 / 7 + 2 / 3) / 3)


def test_extraction_counts(toy: Any) -> None:
    """number: 6 TP, 1 TN, one abstention, one spurious answer, one wrong value."""
    _registry, _golds, _preds, result = toy
    number = result.tasks["number"]
    assert (number.tp, number.fp, number.fn, number.tn) == (6, 2, 2, 1)
    assert number.precision == pytest.approx(0.75)
    assert number.recall == pytest.approx(0.75)
    assert number.f1 == pytest.approx(0.75)
    assert number.primary_value == pytest.approx(0.75)
    assert number.exact_match == pytest.approx(0.7)


def test_confusion_matrix(toy: Any) -> None:
    """The state confusion table must place the null row and column explicitly."""
    _registry, _golds, _preds, result = toy
    confusion = dict(((gold, pred), count) for gold, pred, count in result.tasks["state"].confusion)
    assert confusion[("CA", "CA")] == 2
    assert confusion[("CA", "NY")] == 1
    assert confusion[("NY", "NY")] == 2
    assert confusion[("NY", "CA")] == 1
    assert confusion[("TX", "NY")] == 1
    assert confusion[("(null)", "(null)")] == 1
    assert confusion[("(null)", "CA")] == 1
    assert sum(confusion.values()) == 10


def test_per_task_vector_sits_beside_the_scalar(toy: Any) -> None:
    """A rising aggregate must never be able to hide a collapsing task."""
    _registry, _golds, _preds, result = toy
    assert set(result.per_task_primary) == {"flag", "state", "number"}
    assert result.to_dict()["aggregate"]["per_task_primary"]["flag"] == pytest.approx(0.8)


def test_bootstrap_interval_brackets_the_aggregate(toy: Any) -> None:
    _registry, _golds, _preds, result = toy
    low, high = result.aggregate_ci
    assert low <= result.aggregate <= high
    assert low >= 0.0 and high <= 1.0


def test_trace_makes_the_metric_strict(toy: Any) -> None:
    """Bootstrapping needs pass/fail, or the demo pool fills with half-right examples."""
    registry, golds, preds, _result = toy
    metric = build_metric(registry)
    partial_gold, partial_pred = golds[2], preds[2]  # d03: one of three tasks right
    assert metric(partial_gold, partial_pred) == pytest.approx(1 / 3)
    assert metric(partial_gold, partial_pred, trace=[]) is False
    assert metric(golds[0], preds[0], trace=[]) is True


def test_scoring_refuses_mismatched_lengths(toy: Any) -> None:
    """No silent drops: a short prediction list is an error, not a truncated split."""
    registry, golds, preds, _result = toy
    metric = build_metric(registry)
    with pytest.raises(ValueError, match="cannot score"):
        score_split(registry, metric, golds, preds[:5])


def test_written_metrics_json_matches(tmp_path: Path, toy: Any) -> None:
    registry, golds, _preds, result = toy
    write_run(tmp_path, registry, result, golds)
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["aggregate"]["score"] == pytest.approx(0.70)
    assert payload["tasks"]["flag"]["counts"] == {"tp": 8, "fp": 2, "fn": 2, "tn": 0}
    assert payload["tasks"]["number"]["precision"] == pytest.approx(0.75)
    # run metadata must be reproducible from the file alone
    assert payload["metadata"]["harness_version"]
    assert payload["metadata"]["n_examples"] == 10
    assert payload["metadata"]["support_floor"] == TOY_SUPPORT_FLOOR


def test_failures_md_caps_sampled_errors(tmp_path: Path, toy: Any) -> None:
    """state gets 4 errors but only 3 may be shown, or the optimizer overfits the split."""
    registry, golds, _preds, result = toy
    write_run(tmp_path, registry, result, golds)
    text = (tmp_path / "failures.md").read_text(encoding="utf-8")
    assert "Sampled errors (3 of 4)" in text
    assert "Sampled errors (2 of 2)" in text  # flag
    assert "Sampled errors (3 of 3)" in text  # number
    # every sampled error names its document so a human can go and look
    assert text.count("- **d") <= 3 + 2 + 3
    assert "| gold \\ predicted |" in text


def test_failure_sampling_is_reproducible(tmp_path: Path, toy: Any) -> None:
    registry, golds, preds, _result = toy
    metric = build_metric(registry)
    texts = []
    for _ in range(2):
        result = score_split(
            registry, metric, golds, preds, support_floor=TOY_SUPPORT_FLOOR, measurable_floor=TOY_MEASURABLE_FLOOR
        )
        write_run(tmp_path, registry, result, golds)
        text = (tmp_path / "failures.md").read_text(encoding="utf-8")
        texts.append("\n".join(line for line in text.splitlines() if not line.startswith("Split of")))
    assert texts[0] == texts[1]


def test_unmeasurable_tasks_are_named(toy: Any) -> None:
    """A task whose rarest class is tiny must be flagged, not quoted as a number."""
    registry, golds, preds, _result = toy
    metric = build_metric(registry)
    result = score_split(registry, metric, golds, preds, support_floor=30, measurable_floor=10)
    assert "state" in result.to_dict()["unmeasurable_tasks"]
    assert any("should not be quoted" in note for note in result.tasks["state"].notes)


def test_excluded_classes_leave_the_target_but_stay_reported(toy: Any) -> None:
    """A below-floor class drops out of the aggregate while keeping its own numbers."""
    registry, golds, preds, _result = toy
    metric = build_metric(registry, excluded_classes={"state": {"TX"}})
    result = score_split(
        registry, metric, golds, preds, support_floor=TOY_SUPPORT_FLOOR, measurable_floor=TOY_MEASURABLE_FLOOR
    )
    # d07 is all-correct with TX gold, so state stops contributing and the example still scores 1.0
    assert result.scores[6].aggregate == pytest.approx(1.0)
    assert "state" in result.scores[6].excluded
    # d08 has TX gold and a wrong answer; excluding it lifts that example from 1/3 to 1/2
    assert result.scores[7].aggregate == pytest.approx(0.5)
    assert result.tasks["state"].classes["TX"].support == 2
    assert result.to_dict()["excluded_classes"] == {"state": ["TX"]}


def test_weighted_estimates_are_labelled(toy: Any) -> None:
    """Corpus-level estimates use inverse-probability weights and say so."""
    registry, golds, preds, _result = toy
    metric = build_metric(registry)
    result = score_split(
        registry,
        metric,
        golds,
        preds,
        weights=[2.0] * 10,
        support_floor=TOY_SUPPORT_FLOOR,
        measurable_floor=TOY_MEASURABLE_FLOOR,
    )
    weighted = result.tasks["flag"].weighted
    assert weighted is not None
    assert weighted["counts"]["tp"] == pytest.approx(16.0)
    # doubling every weight cannot move a ratio
    assert weighted["f1"] == pytest.approx(result.tasks["flag"].f1)
    assert "inverse-probability" in weighted["basis"]


def test_best_guess_charges_abstention_as_a_wrong_answer(toy: Any) -> None:
    """Switching abstention policy changes the metric, not just the prompt."""
    registry, golds, preds, _result = toy
    scored = build_metric(registry, abstention="scored")
    guessing = build_metric(registry, abstention="best_guess")

    # d04: gold number A-4, prediction null -- a miss under either policy
    scored_result = scored.score_example(golds[3], preds[3]).results["number"]
    guess_result = guessing.score_example(golds[3], preds[3]).results["number"]
    assert (scored_result.fn, scored_result.fp) == (1, 0)
    assert (guess_result.fn, guess_result.fp) == (1, 1)
    assert "charged as a wrong answer" in guess_result.detail

    # a correct abstention is still a true negative: both sides agree there is nothing there
    assert guessing.score_example(golds[4], preds[4]).results["number"].tn == 1
    # and a wrong non-null answer is unaffected
    assert guessing.score_example(golds[7], preds[7]).results["number"].fp == 1


def test_best_guess_lowers_precision(toy: Any) -> None:
    registry, golds, preds, baseline = toy
    guessing = build_metric(registry, abstention="best_guess")
    result = score_split(
        registry, guessing, golds, preds, support_floor=TOY_SUPPORT_FLOOR, measurable_floor=TOY_MEASURABLE_FLOOR
    )
    assert result.tasks["number"].precision < baseline.tasks["number"].precision
    assert result.tasks["number"].recall == pytest.approx(baseline.tasks["number"].recall)


def test_unknown_abstention_policy_fails_loudly(toy: Any) -> None:
    registry, _golds, _preds, _result = toy
    with pytest.raises(ValueError, match="unknown abstention policy"):
        build_metric(registry, abstention="vibes")


def test_predictions_are_saved_for_free_rescoring(tmp_path: Path, toy: Any) -> None:
    """A matcher fix must be re-measurable without paying for the split again."""
    registry, golds, preds, result = toy
    write_run(tmp_path, registry, result, golds, preds)
    saved = load_predictions(tmp_path / "predictions.jsonl")
    assert set(saved) == {gold["doc_id"] for gold in golds}
    assert saved["d03"]["state"] == "NY"
    assert set(saved["d01"]) == set(registry.ids)

    # re-scoring from the saved predictions reproduces the run exactly
    rescored = score_split(
        registry,
        build_metric(registry),
        golds,
        [saved[gold["doc_id"]] for gold in golds],
        support_floor=TOY_SUPPORT_FLOOR,
        measurable_floor=TOY_MEASURABLE_FLOOR,
    )
    assert rescored.aggregate == pytest.approx(result.aggregate)
    assert rescored.per_task_primary == pytest.approx(result.per_task_primary)


def test_rescoring_a_run_without_predictions_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cannot be re-scored without inference"):
        load_predictions(tmp_path / "predictions.jsonl")


def test_abstention_rates_are_recorded_per_document(toy: Any) -> None:
    """Support counts labels and characters; abstention has to be counted in documents."""
    _registry, _golds, _preds, result = toy
    number = result.tasks["number"]
    # d05 and d06 have null gold; d04 and d05 predicted null
    assert number.gold_null_rate == pytest.approx(0.2)
    assert number.predicted_null_rate == pytest.approx(0.2)
    assert result.tasks["state"].gold_null_rate == pytest.approx(0.2)


def test_null_rate_reference_is_never_negative(fixtures_dir: Path, tmp_path: Path) -> None:
    """A multilabel or span task has more support than documents, which broke the old gate."""
    registry = Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")
    gold = {
        "has_arbitration_clause": True,
        "filing_state": "CA",
        "covered_products": ["hardware", "software", "services"],
        "contract_number": "A-1",
        "counterparty": "Acme Inc",
        "signatories": ["Jane Doe", "John Roe"],
        "contract_value": "$1M",
        "effective_date": "2024-06-15",
        "governing_law_span": {"start": 0, "end": 500},
    }
    metric = build_metric(registry)
    result = score_split(registry, metric, [gold] * 4, [gold] * 4, support_floor=1, measurable_floor=1)
    write_run(tmp_path, registry, result, [gold] * 4)
    rates = null_rates_from_metrics(json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")))
    assert set(rates) == set(registry.ids)
    assert all(0.0 <= rate <= 1.0 for rate in rates.values()), rates
    # support far exceeds the document count for these two, which is the case that broke
    assert result.tasks["governing_law_span"].support > result.tasks["governing_law_span"].n_examples
    assert result.tasks["covered_products"].support > result.tasks["covered_products"].n_examples


def test_typed_predictions_round_trip_through_json(fixtures_dir: Path, tmp_path: Path) -> None:
    """Saved predictions must reproduce the run, for every task type, not just strings.

    Typed outputs are pydantic models. Serialized through repr they come back as
    ``"value=375000.0 unit='USD'"``, which re-scores to a different number -- so the saved
    predictions would quietly stop being the predictions.
    """
    registry = Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")
    gold = {
        "doc_id": "d01",
        "has_arbitration_clause": True,
        "filing_state": "CA",
        "covered_products": ["hardware", "software"],
        "contract_number": "MSA-2024-1014",
        "counterparty": "Acme Holdings, Inc.",
        "signatories": ["Jane Doe", "John Roe"],
        "contract_value": Quantity(value=375000.0, unit="USD"),
        "effective_date": PartialDate(value="2024-06-15", granularity="day"),
        "governing_law_span": Span(start=100, end=250),
    }
    metric = build_metric(registry)
    original = score_split(registry, metric, [gold], [gold], support_floor=1, measurable_floor=1)
    assert original.aggregate == pytest.approx(1.0)

    write_run(tmp_path, registry, original, [gold], [gold])
    saved = load_predictions(tmp_path / "predictions.jsonl")["d01"]
    # structured, not stringified
    assert saved["contract_value"] == {"value": 375000.0, "unit": "USD"}
    assert saved["effective_date"] == {"value": "2024-06-15", "granularity": "day"}
    assert saved["governing_law_span"] == {"start": 100, "end": 250}

    rescored = score_split(registry, metric, [gold], [saved], support_floor=1, measurable_floor=1)
    assert rescored.aggregate == pytest.approx(original.aggregate)
    assert rescored.per_task_primary == pytest.approx(original.per_task_primary)
