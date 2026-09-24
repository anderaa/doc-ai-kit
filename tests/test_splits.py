"""Split, stratification and support-floor tests."""

from __future__ import annotations

import random

import pytest

from doc_ai_kit.dataset import LabelRecord
from doc_ai_kit.registry import Registry
from doc_ai_kit.splits import (
    ClassSupport,
    SplitError,
    SupportDecision,
    below_floor,
    class_supports,
    half_width_table,
    inclusion_probability,
    keyword_stratum,
    make_splits,
    model_nominated_stratum,
    plan_enrichment,
    support_floor_prompt,
)

SEED = 20260918


def corpus(n: int, rare_count: int = 3) -> list[LabelRecord]:
    """Build a labeled corpus where the class 'TX' is deliberately rare."""
    rng = random.Random(1)
    records = []
    for index in range(n):
        state = "TX" if index < rare_count else rng.choice(["CA", "NY"])
        records.append(
            LabelRecord(
                doc_id=f"d{index:03d}",
                labels={"flag": index % 2 == 0, "state": state, "number": f"A-{index}"},
            )
        )
    return records


def test_class_supports_counts_every_declared_class(toy_registry: Registry) -> None:
    supports = class_supports(toy_registry, corpus(40))
    assert set(supports) == {"flag", "state"}  # extract_exact has no class space
    assert supports["state"]["TX"].count == 3
    assert supports["state"]["CA"].count + supports["state"]["NY"].count == 37
    assert supports["flag"]["true"].count == 20


def test_support_reports_what_it_can_carry(toy_registry: Registry) -> None:
    support = class_supports(toy_registry, corpus(40))["state"]["TX"]
    assert support.readable_as == "not measurable"
    assert support.prevalence == pytest.approx(3 / 40)
    # at 7.5% prevalence, 30 examples needs about 400 documents
    assert support.documents_needed(30) == 400


def test_documents_needed_scales_with_rarity() -> None:
    """A 2% class needs roughly 1,500 documents for 30 examples."""
    support = ClassSupport(task_id="t", label="rare", count=2, total=100)
    assert support.documents_needed(30) == 1500


def test_below_floor_lists_rarest_first(toy_registry: Registry) -> None:
    rare = below_floor(class_supports(toy_registry, corpus(60)), support_floor=30)
    assert rare[0].label == "TX"
    assert all(support.count < 30 for support in rare)


def test_support_prompt_shows_the_numbers_and_all_four_options(toy_registry: Registry) -> None:
    support = class_supports(toy_registry, corpus(40))["state"]["TX"]
    text = support_floor_prompt(support, support_floor=30)
    for choice in ("enrich", "collapse", "binary_detection", "report_unmeasured"):
        assert choice in text
    assert "half-width" in text
    assert "400 documents" in text


def test_decisions_render_for_the_log() -> None:
    decision = SupportDecision(task_id="state", label="TX", count=3, choice="report_unmeasured", rationale="Too rare.")
    markdown = decision.to_markdown()
    assert "report_unmeasured" in markdown
    assert "Too rare." in markdown


def test_half_width_table_matches_the_protocol() -> None:
    rows = "\n".join(half_width_table())
    assert "+/-29 pts" in rows and "presence check only" in rows
    assert "+/-8 pts" in rows and "| 100 |" in rows


@pytest.mark.parametrize("size,expected", [(250, "50/25/25"), (150, "45/25/30"), (75, "5-fold")])
def test_strategy_follows_corpus_size(toy_registry: Registry, size: int, expected: str) -> None:
    splits = make_splits(toy_registry, corpus(size), seed=SEED)
    assert expected in splits.strategy


def test_ratios_are_respected(toy_registry: Registry) -> None:
    splits = make_splits(toy_registry, corpus(400), seed=SEED)
    total = len(splits.train) + len(splits.val) + len(splits.holdout)
    assert total == 400
    assert len(splits.train) / total == pytest.approx(0.50, abs=0.04)
    assert len(splits.val) / total == pytest.approx(0.25, abs=0.04)
    assert len(splits.holdout) / total == pytest.approx(0.25, abs=0.04)


def test_splits_are_reproducible_from_the_seed(toy_registry: Registry) -> None:
    records = corpus(200)
    first = make_splits(toy_registry, records, seed=SEED)
    second = make_splits(toy_registry, records, seed=SEED)
    assert first.assignments == second.assignments
    different = make_splits(toy_registry, records, seed=SEED + 1)
    assert different.assignments != first.assignments


def test_every_document_lands_in_exactly_one_split(toy_registry: Registry) -> None:
    records = corpus(137)
    splits = make_splits(toy_registry, records, seed=SEED)
    assigned = splits.train + splits.val + splits.holdout
    assert sorted(assigned) == sorted(record.doc_id for record in records)
    assert len(assigned) == len(set(assigned))


@pytest.mark.parametrize("rare_count", [1, 2, 3, 5, 9])
def test_no_class_reaches_the_holdout_without_reaching_train(toy_registry: Registry, rare_count: int) -> None:
    """The invariant that makes a holdout number mean anything."""
    records = corpus(120, rare_count=rare_count)
    splits = make_splits(toy_registry, records, seed=SEED)
    by_id = {record.doc_id: record for record in records}
    train_states = {by_id[doc_id].labels["state"] for doc_id in splits.train}
    holdout_states = {by_id[doc_id].labels["state"] for doc_id in splits.holdout}
    assert holdout_states <= train_states


def test_rare_class_is_spread_not_clustered(toy_registry: Registry) -> None:
    """Stratification must put the rare class in train, not leave it to chance."""
    records = corpus(200, rare_count=12)
    splits = make_splits(toy_registry, records, seed=SEED)
    by_id = {record.doc_id: record for record in records}
    in_train = sum(1 for doc_id in splits.train if by_id[doc_id].labels["state"] == "TX")
    assert in_train >= 4


def test_cross_validation_regime_populates_folds(toy_registry: Registry) -> None:
    splits = make_splits(toy_registry, corpus(80), seed=SEED)
    assert len(splits.folds) == 5
    assert sorted(doc_id for fold in splits.folds for doc_id in fold) == sorted(splits.optimization_pool)
    assert len(splits.holdout) / 80 == pytest.approx(0.30, abs=0.05)


def test_small_corpus_warns(toy_registry: Registry, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        make_splits(toy_registry, corpus(30), seed=SEED)
    assert "too few points for automated search" in caplog.text


def test_empty_corpus_fails_loudly(toy_registry: Registry) -> None:
    with pytest.raises(SplitError, match="zero labeled documents"):
        make_splits(toy_registry, [], seed=SEED)


def test_keyword_stratum_is_model_independent() -> None:
    texts = {"a": "governed by the laws of Texas", "b": "governed by California law", "c": "no state named"}
    assert keyword_stratum(texts, r"\btexas\b") == ["a"]
    assert keyword_stratum(texts, r"governed by") == ["a", "b"]


def test_model_nominated_stratum_reads_predictions() -> None:
    predictions = {"a": {"state": "TX"}, "b": {"state": "CA"}, "c": {"state": None}}
    assert model_nominated_stratum(predictions, "state", "TX") == ["a"]


def test_inclusion_probability() -> None:
    assert inclusion_probability(25, 100) == 0.25
    with pytest.raises(SplitError, match="empty stratum"):
        inclusion_probability(1, 0)
    with pytest.raises(SplitError, match="cannot select"):
        inclusion_probability(10, 5)


def test_enrichment_plan_is_reproducible_and_carries_its_probability() -> None:
    candidates = [f"d{i:03d}" for i in range(40)]
    chosen, probability = plan_enrichment(candidates, already_labeled=["d000"], target=10, stratum="keyword", seed=SEED)
    again, _ = plan_enrichment(candidates, already_labeled=["d000"], target=10, stratum="keyword", seed=SEED)
    assert chosen == again
    assert len(chosen) == 10
    assert "d000" not in chosen
    # 10 drawn from the 39 that remain unlabeled
    assert probability == pytest.approx(10 / 39)


def test_enrichment_caps_at_what_exists(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        chosen, probability = plan_enrichment(["a", "b"], [], target=10, stratum="random", seed=SEED)
    assert chosen == ["a", "b"]
    assert probability == 1.0
    assert "only had 2" in caplog.text


def test_unknown_stratum_fails_loudly() -> None:
    with pytest.raises(SplitError, match="unknown stratum"):
        plan_enrichment(["a"], [], target=1, stratum="vibes", seed=1)  # type: ignore[arg-type]


def test_exhausted_stratum_fails_loudly() -> None:
    with pytest.raises(SplitError, match="no unlabeled documents left"):
        plan_enrichment(["a"], ["a"], target=1, stratum="keyword", seed=1)
