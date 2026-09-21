"""Holdout reading, report assembly and the shared ledger."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from conftest import document_for, scripted_lm

from doc_harness.config import Config
from doc_harness.dataset import LabelRecord, build_examples
from doc_harness.guards import GuardError
from doc_harness.metric import build_metric
from doc_harness.program import build_program
from doc_harness.registry import Registry
from doc_harness.report import (
    LEDGER_COLUMNS,
    TaskGap,
    append_ledger,
    read_gaps,
    run_holdout,
    write_report,
)

PERFECT = {"flag": 1.0, "state": 1.0, "number": 1.0}


def gap(task_id: str, validation: float, holdout: float, ci: tuple[float, float], support: int = 20) -> TaskGap:
    return TaskGap(
        task_id=task_id,
        validation=validation,
        holdout=holdout,
        holdout_ci=ci,
        support=support,
        measurable=True,
    )


def test_within_ci_reads_as_no_overfitting() -> None:
    gaps = [gap("a", 0.80, 0.78, (0.70, 0.88)), gap("b", 0.90, 0.91, (0.85, 0.95))]
    assert read_gaps(gaps) == "within_ci"


def test_modest_gap_outside_ci_reads_as_mild() -> None:
    gaps = [gap("a", 0.80, 0.77, (0.72, 0.79)), gap("b", 0.90, 0.88, (0.84, 0.89))]
    assert read_gaps(gaps) == "modest"


def test_a_large_gap_on_one_task_of_four() -> None:
    gaps = [
        gap("a", 0.90, 0.60, (0.50, 0.70)),
        gap("b", 0.85, 0.84, (0.80, 0.90)),
        gap("c", 0.80, 0.79, (0.74, 0.85)),
        gap("d", 0.75, 0.76, (0.70, 0.82)),
    ]
    assert read_gaps(gaps) == "large_few"


def test_a_large_gap_across_the_board() -> None:
    gaps = [
        gap("a", 0.90, 0.60, (0.50, 0.70)),
        gap("b", 0.88, 0.55, (0.45, 0.65)),
        gap("c", 0.80, 0.40, (0.30, 0.50)),
    ]
    assert read_gaps(gaps) == "large_widespread"


def test_no_tasks_is_not_a_crash() -> None:
    assert read_gaps([]) == "within_ci"


def test_gap_sign_is_validation_minus_holdout() -> None:
    assert gap("a", 0.9, 0.7, (0.6, 0.8)).gap == pytest.approx(0.2)
    # a holdout that beats validation is a negative gap, not a large one
    assert gap("a", 0.7, 0.9, (0.85, 0.95)).is_large is False


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    return tmp_path


@pytest.fixture
def setup(toy_registry: Registry) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    records = [
        LabelRecord(
            doc_id=f"h{index:02d}",
            labels={"flag": index % 2 == 0, "state": ["CA", "NY"][index % 2], "number": f"A-{index}"},
            labeling_mode="blind",
        )
        for index in range(12)
    ]
    texts = {record.doc_id: document_for(record.doc_id) for record in records}
    examples = build_examples(records, texts)
    answers = {
        example.doc_id: {
            "flag": "true" if example.flag else "false",
            "state": example.state,
            "number": example.number,
        }
        for example in examples
    }
    return examples, answers


def _config() -> Config:
    return Config.from_mapping(
        {
            "models": {"task": "fake/scripted", "reflection": "fake/scripted"},
            "optimization": {"num_threads": 1},
            "splits": {"support_floor": 2, "measurable_floor": 2},
            "metric": {"bootstrap_resamples": 200},
        }
    )


def test_holdout_runs_once_and_writes_its_lock(
    project: Path, toy_registry: Registry, setup: tuple[list[Any], Any]
) -> None:
    examples, answers = setup
    metric = build_metric(toy_registry)
    program = build_program(toy_registry)
    config = _config()
    with scripted_lm(answers):
        report = run_holdout(
            toy_registry,
            config,
            metric,
            program,
            examples,
            project,
            validation_aggregate=1.0,
            validation_per_task={"flag": 1.0, "state": 1.0, "number": 1.0},
            opened_by="holdout command",
        )
    assert (project / "runs" / "holdout" / ".lock").exists()
    assert (project / "runs" / "holdout" / "metrics.json").exists()
    assert (project / "runs" / "holdout" / "failures.md").exists()
    assert report.result.aggregate == pytest.approx(1.0)
    assert report.verdict == "within_ci"
    assert "# Holdout" in report.to_markdown()
    assert "Measured once" in report.to_markdown()


def test_second_holdout_refuses(project: Path, toy_registry: Registry, setup: tuple[list[Any], Any]) -> None:
    examples, answers = setup
    metric = build_metric(toy_registry)
    program = build_program(toy_registry)
    config = _config()
    with scripted_lm(answers):
        run_holdout(toy_registry, config, metric, program, examples, project, 1.0, PERFECT, opened_by="first")
        with pytest.raises(GuardError, match="one-shot measurement"):
            run_holdout(toy_registry, config, metric, program, examples, project, 1.0, PERFECT, opened_by="second")


def test_override_is_stamped_into_the_report(
    project: Path, toy_registry: Registry, setup: tuple[list[Any], Any]
) -> None:
    examples, answers = setup
    metric = build_metric(toy_registry)
    program = build_program(toy_registry)
    config = _config()
    with scripted_lm(answers):
        run_holdout(toy_registry, config, metric, program, examples, project, 1.0, PERFECT, opened_by="first")
        report = run_holdout(
            toy_registry,
            config,
            metric,
            program,
            examples,
            project,
            1.0,
            PERFECT,
            opened_by="second",
            override=True,
            reason="the first run used stale labels",
        )
    markdown = report.to_markdown()
    assert "Holdout override" in markdown
    assert "stale labels" in markdown
    assert "weaker as an out-of-sample estimate" in markdown


def test_unmeasurable_tasks_are_named_in_the_report(
    project: Path, toy_registry: Registry, setup: tuple[list[Any], Any]
) -> None:
    examples, answers = setup
    metric = build_metric(toy_registry)
    program = build_program(toy_registry)
    config = Config.from_mapping(
        {
            "models": {"task": "fake/scripted", "reflection": "fake/scripted"},
            "optimization": {"num_threads": 1},
            "splits": {"support_floor": 30, "measurable_floor": 10},
            "metric": {"bootstrap_resamples": 200},
        }
    )
    with scripted_lm(answers):
        report = run_holdout(
            toy_registry, config, metric, program, examples, project, 1.0, PERFECT, opened_by="holdout"
        )
    assert report.unmeasurable
    assert "cannot measure" in report.to_markdown()


def test_write_report_assembles_sections(tmp_path: Path) -> None:
    path = write_report(tmp_path, ["# Baselines\n\nfirst", "# Holdout\n\nsecond"], title="Toy project")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Toy project")
    assert "doc-harness" in text
    assert text.index("# Baselines") < text.index("# Holdout")


def test_ledger_appends_with_a_header(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    append_ledger(ledger, {"project": "alpha", "n_documents": 500, "compiled_holdout": 0.81})
    append_ledger(ledger, {"project": "beta", "n_documents": 120, "compiled_holdout": 0.74})
    with ledger.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["project"] for row in rows] == ["alpha", "beta"]
    assert set(rows[0]) == set(LEDGER_COLUMNS)


def test_ledger_rejects_unknown_columns(tmp_path: Path) -> None:
    """A typo must not silently vanish into an unwritten column."""
    with pytest.raises(ValueError, match="unknown ledger column"):
        append_ledger(tmp_path / "ledger.csv", {"projct": "typo"})
