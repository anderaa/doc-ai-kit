"""Program construction and the three baselines, run end to end against a scripted LM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import document_for, scripted_lm

from doc_harness.baseline import BOOTSTRAP, ZERO_SHOT, find_upstream_problems, run_baselines
from doc_harness.config import Config
from doc_harness.dataset import LabelRecord, build_examples
from doc_harness.evaluate import run_program, score_split
from doc_harness.metric import build_metric
from doc_harness.program import build_program, instructions_of, load_program, save_program
from doc_harness.registry import Registry


@pytest.fixture
def toy_examples(toy_registry: Registry, toy_rows: list[dict[str, Any]]) -> list[Any]:
    records = [LabelRecord(doc_id=row["doc_id"], labels=dict(row["gold"])) for row in toy_rows]
    texts = {row["doc_id"]: document_for(row["doc_id"]) for row in toy_rows}
    return build_examples(records, texts)


@pytest.fixture
def scripted_answers(toy_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Script the LM to return exactly the toy fixture's predictions."""
    answers: dict[str, dict[str, Any]] = {}
    for row in toy_rows:
        answers[row["doc_id"]] = {
            key: ("null" if value is None else ("true" if value is True else ("false" if value is False else value)))
            for key, value in row["pred"].items()
        }
    return answers


@pytest.fixture
def config() -> Config:
    return Config.from_mapping(
        {
            "models": {"task": "fake/scripted", "reflection": "fake/scripted"},
            "optimization": {"num_threads": 1, "max_bootstrapped_demos": 2, "max_labeled_demos": 2},
            "splits": {"support_floor": 2, "measurable_floor": 2},
        }
    )


def test_program_has_one_predictor_per_group(fixtures_dir: Path) -> None:
    registry = Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")
    program = build_program(registry)
    names = [name for name, _ in program.named_predictors()]
    assert len(names) == 2, names
    assert any("spans" in name for name in names)


def test_chain_of_thought_is_a_choice_not_a_default(toy_registry: Registry) -> None:
    plain = build_program(toy_registry, module_type="predict")
    reasoning = build_program(toy_registry, module_type="chain_of_thought")
    assert len(list(reasoning.named_predictors())) == len(list(plain.named_predictors()))
    assert "reasoning" in str(next(iter(reasoning.named_predictors()))[1].signature.output_fields)


def test_unknown_module_type_fails_loudly(toy_registry: Registry) -> None:
    with pytest.raises(ValueError, match="unknown module type"):
        build_program(toy_registry, module_type="magic")


def test_program_merges_every_task(toy_registry: Registry, toy_examples: list[Any], scripted_answers: Any) -> None:
    program = build_program(toy_registry)
    with scripted_lm(scripted_answers):
        prediction = program(document=document_for("d01"))
    assert prediction.flag is True
    assert prediction.state == "CA"
    assert prediction.number == "A-1"


def test_save_and_load_round_trip(tmp_path: Path, toy_registry: Registry, toy_examples: list[Any]) -> None:
    program = build_program(toy_registry)
    for _name, predictor in program.named_predictors():
        predictor.signature = predictor.signature.with_instructions("A tuned instruction.")
    path = tmp_path / "compiled.json"
    save_program(program, path)
    loaded = load_program(toy_registry, path)
    assert all(text == "A tuned instruction." for text in instructions_of(loaded).values())


def test_load_missing_program_fails_loudly(tmp_path: Path, toy_registry: Registry) -> None:
    with pytest.raises(FileNotFoundError, match="no compiled program"):
        load_program(toy_registry, tmp_path / "absent.json")


def test_predictions_match_the_hand_computed_split(
    toy_registry: Registry, toy_examples: list[Any], scripted_answers: Any
) -> None:
    """Running the program end to end must reproduce the toy split's scored numbers."""
    program = build_program(toy_registry)
    metric = build_metric(toy_registry)
    with scripted_lm(scripted_answers):
        predictions = run_program(program, toy_examples, metric, num_threads=1)
    result = score_split(toy_registry, metric, toy_examples, predictions, support_floor=2, measurable_floor=2)
    assert result.aggregate == pytest.approx(0.70)
    assert result.tasks["flag"].f1 == pytest.approx(0.8)
    assert result.tasks["number"].precision == pytest.approx(0.75)


def test_run_baselines_records_three(
    tmp_path: Path,
    toy_registry: Registry,
    toy_examples: list[Any],
    scripted_answers: Any,
    config: Config,
) -> None:
    demos = [toy_examples[0], toy_examples[1]]
    runs_dir = tmp_path / "runs"
    with scripted_lm(scripted_answers):
        report = run_baselines(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=toy_examples[:6],
            valset=toy_examples,
            runs_dir=runs_dir,
            demos={"all": demos},
        )
    assert [run.name for run in report.runs] == [ZERO_SHOT, "few_shot", BOOTSTRAP]
    assert report.best.score == pytest.approx(0.70)
    for name in (ZERO_SHOT, "few_shot", BOOTSTRAP):
        run_dir = runs_dir / f"baseline_{name}"
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "failures.md").exists()
    payload = json.loads((runs_dir / f"baseline_{ZERO_SHOT}" / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["baseline"] == ZERO_SHOT
    assert payload["metadata"]["config_hash"]
    assert payload["metadata"]["instructions"]
    assert "Best baseline" in report.to_markdown()


def test_few_shot_is_skipped_loudly_without_demos(
    tmp_path: Path,
    toy_registry: Registry,
    toy_examples: list[Any],
    scripted_answers: Any,
    config: Config,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with scripted_lm(scripted_answers), caplog.at_level("WARNING"):
        report = run_baselines(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=toy_examples[:6],
            valset=toy_examples,
            runs_dir=tmp_path / "runs",
        )
    assert [run.name for run in report.runs] == [ZERO_SHOT, BOOTSTRAP]
    assert "hand-written demonstrations" in caplog.text


def test_upstream_problems_are_surfaced(
    tmp_path: Path,
    toy_registry: Registry,
    toy_examples: list[Any],
    config: Config,
) -> None:
    """A task nobody can answer is an extraction, definition or matcher problem."""
    broken = {row: {"flag": "true", "state": "null", "number": "null"} for row in [e.doc_id for e in toy_examples]}
    with scripted_lm(broken):
        report = run_baselines(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=toy_examples[:6],
            valset=toy_examples,
            runs_dir=tmp_path / "runs",
        )
    assert "number" in report.upstream_problems
    assert "Tasks scoring near zero" in report.to_markdown()


def test_find_upstream_problems_uses_the_best_baseline() -> None:
    """A task that one baseline answers well is not an upstream problem."""

    class _Result:
        def __init__(self, values: dict[str, float]) -> None:
            self.per_task_primary = values

    class _Run:
        def __init__(self, values: dict[str, float]) -> None:
            self.result = _Result(values)

    runs = [_Run({"a": 0.0, "b": 0.0}), _Run({"a": 0.9, "b": 0.0})]
    assert find_upstream_problems(runs) == {"b": 0.0}  # type: ignore[arg-type]
