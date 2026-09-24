"""Labels and splits: loading, validation and the read-only expectations around them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_ai_kit.dataset import (
    DatasetError,
    LabelRecord,
    Splits,
    build_examples,
    load_labels,
    load_splits,
    select,
    validate_against_registry,
    weights_for,
    write_labels,
    write_splits,
)
from doc_ai_kit.registry import Registry


def _records() -> list[LabelRecord]:
    return [
        LabelRecord(doc_id="d01", labels={"flag": True, "state": "CA", "number": "A-1"}),
        LabelRecord(doc_id="d02", labels={"flag": False, "state": None, "number": None}, stratum="keyword"),
        LabelRecord(
            doc_id="d03",
            labels={"flag": True, "state": "NY", "number": "A-3"},
            stratum="model_nominated",
            inclusion_probability=0.25,
            labeling_mode="blind",
        ),
    ]


def test_labels_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    write_labels(path, _records())
    loaded = load_labels(path)
    assert [record.doc_id for record in loaded] == ["d01", "d02", "d03"]
    assert loaded[1].stratum == "keyword"
    assert loaded[2].labeling_mode == "blind"
    assert loaded[2].inclusion_probability == 0.25


def test_inverse_probability_weights() -> None:
    """A document drawn with probability 0.25 stands for four documents in the corpus."""
    assert weights_for(_records()) == [1.0, 1.0, 4.0]


def test_zero_inclusion_probability_is_an_error() -> None:
    with pytest.raises(DatasetError, match="must be positive"):
        _ = LabelRecord(doc_id="x", labels={}, inclusion_probability=0.0).weight


def test_duplicate_doc_id_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps({"doc_id": "d01", "labels": {}}) + "\n" + json.dumps({"doc_id": "d01", "labels": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate doc_id"):
        load_labels(path)


def test_malformed_label_rows_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"doc_id": "d01"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="no 'labels' mapping"):
        load_labels(path)

    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_labels(path)


def test_missing_labels_file_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="label some documents first"):
        load_labels(tmp_path / "nope.jsonl")


def test_validate_reports_every_problem_at_once(toy_registry: Registry) -> None:
    records = [
        LabelRecord(doc_id="d01", labels={"flag": True, "state": "ZZ", "number": "A-1"}),
        LabelRecord(doc_id="d02", labels={"flag": True, "typo_task": 1}),
    ]
    problems = validate_against_registry(toy_registry, records)
    joined = "\n".join(problems)
    assert "outside the enum" in joined
    assert "undeclared task" in joined
    assert "no label for task(s)" in joined


def test_splits_round_trip(tmp_path: Path) -> None:
    splits = Splits(
        seed=7,
        strategy="50/25/25",
        assignments={"train": ["a", "b"], "val": ["c"], "holdout": ["d"]},
    )
    path = tmp_path / "splits.json"
    write_splits(path, splits)
    loaded = load_splits(path)
    assert loaded.seed == 7
    assert loaded.optimization_pool == ["a", "b", "c"]
    assert loaded.holdout == ["d"]
    # the holdout is never part of what an optimizer may see
    assert "d" not in loaded.optimization_pool


def test_a_document_cannot_be_in_two_splits() -> None:
    with pytest.raises(DatasetError, match="appears in both"):
        Splits(seed=1, strategy="x", assignments={"train": ["a"], "val": ["a"], "holdout": []})


def test_missing_split_fails_loudly() -> None:
    with pytest.raises(DatasetError, match="missing: holdout"):
        Splits(seed=1, strategy="x", assignments={"train": ["a"], "val": ["b"]})


def test_missing_splits_file_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="doc-ai-kit make-splits"):
        load_splits(tmp_path / "splits.json")


def test_select_refuses_unlabeled_documents() -> None:
    with pytest.raises(DatasetError, match="are not labeled"):
        select(_records(), ["d01", "d99"])


def test_select_preserves_requested_order() -> None:
    assert [record.doc_id for record in select(_records(), ["d03", "d01"])] == ["d03", "d01"]


def test_build_examples_marks_only_the_document_as_input(toy_registry: Registry) -> None:
    records = _records()
    texts = {record.doc_id: f"text for {record.doc_id}" for record in records}
    examples = build_examples(records, texts)
    assert len(examples) == 3
    assert set(examples[0].inputs().keys()) == {"document"}
    assert examples[0].doc_id == "d01"
    assert examples[0].state == "CA"


def test_build_examples_refuses_missing_text() -> None:
    with pytest.raises(DatasetError, match="no cached text"):
        build_examples(_records(), {"d01": "only one"})
