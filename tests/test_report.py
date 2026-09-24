"""Holdout reading, report assembly and the shared ledger."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from conftest import document_for, scripted_lm

from doc_ai_kit.config import Config
from doc_ai_kit.dataset import LabelRecord, build_examples
from doc_ai_kit.guards import GuardError
from doc_ai_kit.metric import build_metric
from doc_ai_kit.program import build_program
from doc_ai_kit.registry import Registry
from doc_ai_kit.report import (
    LEDGER_COLUMNS,
    NOTES_FILE,
    TaskGap,
    append_ledger,
    artifact_links,
    best_baseline,
    holdout_aggregate,
    read_gaps,
    run_holdout,
    write_prompt,
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
    assert "# How well it works" in report.to_markdown()
    assert "used once, at the end" in report.to_markdown()


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
    assert "used more than once" in markdown
    assert "stale labels" in markdown
    assert "a weaker\nestimate of how the program behaves" in markdown


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
    assert "cannot tell you" in report.to_markdown()


def test_write_report_assembles_sections(tmp_path: Path) -> None:
    path = write_report(tmp_path, ["# Baselines\n\nfirst", "# Holdout\n\nsecond"], title="Toy project")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Toy project")
    assert "doc-ai-kit" in text
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


def test_the_report_links_to_every_artifact(tmp_path: Path) -> None:
    """Reported from a real run: numbers with nothing to click."""
    (tmp_path / "runs" / "holdout").mkdir(parents=True)
    (tmp_path / "runs" / "holdout" / "metrics.json").write_text('{"aggregate": {"score": 0.822}}', encoding="utf-8")
    (tmp_path / "decisions.md").write_text("decided", encoding="utf-8")
    links = artifact_links(tmp_path)
    assert "[`runs/holdout/metrics.json`](runs/holdout/metrics.json)" in links
    assert "[`decisions.md`](decisions.md)" in links
    assert "Not written by this project" in links and "`PROMPT.md`" in links

    text = write_report(tmp_path, ["# Holdout\n\nsecond"], title="Toy").read_text(encoding="utf-8")
    assert "## Where everything is" in text


def test_hand_written_notes_survive_a_regenerated_report(tmp_path: Path) -> None:
    """Reported from a real run: close regenerates REPORT.md and silently drops what was added."""
    (tmp_path / NOTES_FILE).write_text("The cap clause task is not fit for automated use.", encoding="utf-8")
    text = write_report(tmp_path, ["# Holdout\n\nsecond"], title="Toy").read_text(encoding="utf-8")
    assert "## Notes" in text and "not fit for automated use" in text

    again = write_report(tmp_path, ["# Holdout\n\nsecond"], title="Toy").read_text(encoding="utf-8")
    assert again.count("not fit for automated use") == 1


def test_the_shipped_prompt_is_written_as_markdown(
    tmp_path: Path, toy_registry: Registry, setup: tuple[list[Any], Any]
) -> None:
    """Reported from a real run: the only copy was a JSON string inside the compiled program."""
    examples, _answers = setup
    program = build_program(toy_registry, instructions={"all": "Answer from the contract only."})
    program.predict_all.demos = [examples[0]]
    path = write_prompt(tmp_path, toy_registry, program)
    text = path.read_text(encoding="utf-8")
    assert path.name == "PROMPT.md"
    assert "Answer from the contract only." in text
    assert "**state** (multiclass, one of: CA, NY, TX)" in text
    assert "Which state governs." in text
    assert "### Demonstrations (1)" in text and "`state`:" in text


def test_the_ledger_numbers_come_from_the_runs(tmp_path: Path) -> None:
    """Reported from a real run: the columns were declared and never filled, so 0.822 went in by hand."""
    runs = tmp_path / "runs"
    (runs / "holdout").mkdir(parents=True)
    (runs / "holdout" / "metrics.json").write_text('{"aggregate": {"score": 0.8221}}', encoding="utf-8")
    assert holdout_aggregate(runs) == pytest.approx(0.8221)
    assert holdout_aggregate(tmp_path / "empty") is None

    for name, score in (("baseline_zero_shot", 0.61), ("baseline_bootstrap_few_shot", 0.74)):
        (runs / name).mkdir()
        (runs / name / "metrics.json").write_text(f'{{"aggregate": {{"score": {score}}}}}', encoding="utf-8")
    assert best_baseline(runs) == ("baseline_bootstrap_few_shot", 0.74)
