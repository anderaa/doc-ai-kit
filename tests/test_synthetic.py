"""The synthetic corpus generator, tested offline.

The dogfood run needs a real model; the corpus it runs on does not, and a broken generator
would make the acceptance test meaningless rather than failing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import load_example

from doc_harness.dataset import load_labels, load_splits
from doc_harness.metric import build_metric
from doc_harness.registry import Registry, TaskType
from doc_harness.splits import class_supports, make_splits

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "synthetic"
_synthetic = load_example("synthetic")
build_specs, generate, mark_holdout_blind, span_for = (
    _synthetic.build_specs,
    _synthetic.generate,
    _synthetic.mark_holdout_blind,
    _synthetic.span_for,
)


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.from_yaml(EXAMPLES / "tasks.yaml")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    project = tmp_path_factory.mktemp("synthetic")
    generate(project, count=20)
    return project


def test_tasks_cover_every_task_type(registry: Registry) -> None:
    """The acceptance test is only an acceptance test if it exercises the whole matrix."""
    assert {str(task.type) for task in registry} == set(TaskType)


def test_specs_are_deterministic() -> None:
    first = [spec.doc_id for spec in build_specs(20)]
    second = [spec.doc_id for spec in build_specs(20)]
    assert first == second == [f"doc_{i:02d}" for i in range(20)]


def test_every_class_clears_the_dogfood_floor(registry: Registry, corpus: Path) -> None:
    """Twenty documents is few; a random draw would leave classes with one example."""
    records = load_labels(corpus / "data" / "labels.jsonl")
    for task_id, supports in class_supports(registry, records).items():
        for label, support in supports.items():
            assert support.count >= 3, f"{task_id}/{label} has only {support.count}"


def test_gold_labels_are_complete(registry: Registry, corpus: Path) -> None:
    records = load_labels(corpus / "data" / "labels.jsonl")
    assert len(records) == 20
    for record in records:
        assert set(record.labels) == set(registry.ids), record.doc_id


def test_span_offsets_point_at_the_governing_law_sentence(corpus: Path) -> None:
    """A character offset only means anything against the exact text the program is shown."""
    records = load_labels(corpus / "data" / "labels.jsonl")
    for record in records:
        text = (corpus / "data" / "text" / f"{record.doc_id}.md").read_text(encoding="utf-8")
        span = record.labels["governing_law_span"]
        quoted = text[span["start"] : span["end"]]
        assert quoted.startswith("This Agreement shall be governed by")
        assert quoted.endswith("principles.")


def test_span_for_returns_none_when_absent() -> None:
    assert span_for("not in here", "some other text") is None


def test_gold_labels_score_perfectly_against_themselves(registry: Registry, corpus: Path) -> None:
    """If gold does not score 1.0 against gold, a matcher is broken, not the model."""
    records = load_labels(corpus / "data" / "labels.jsonl")
    metric = build_metric(registry)
    for record in records:
        score = metric.score_example(record.labels, record.labels)
        assert score.aggregate == pytest.approx(1.0), (record.doc_id, list(score.failures()))


def test_blind_marking_follows_the_splits(registry: Registry, corpus: Path) -> None:
    """The holdout is labeled blind after the splits exist, never before."""
    records = load_labels(corpus / "data" / "labels.jsonl")
    assert all(record.labeling_mode == "corrected" for record in records)

    splits = make_splits(registry, records, seed=20260918)
    (corpus / "data" / "splits.json").write_text(json.dumps(splits.to_dict()), encoding="utf-8")
    marked = mark_holdout_blind(corpus)

    assert marked == len(splits.holdout)
    updated = {record.doc_id: record.labeling_mode for record in load_labels(corpus / "data" / "labels.jsonl")}
    assert all(updated[doc_id] == "blind" for doc_id in splits.holdout)
    assert all(updated[doc_id] == "corrected" for doc_id in load_splits(corpus / "data" / "splits.json").train)


def test_extraction_manifest_is_written(corpus: Path) -> None:
    manifest = (corpus / "data" / "extraction_manifest.csv").read_text(encoding="utf-8")
    assert manifest.count("\n") == 21  # header plus twenty documents
    assert "doc_00" in manifest
