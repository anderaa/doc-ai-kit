"""Optimizer tests: budgets, class coverage, the leaderboard and champion pinning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from conftest import document_for, scripted_lm

from doc_harness.config import Config
from doc_harness.dataset import LabelRecord, build_examples
from doc_harness.guards import GuardError, RolloutBudgetExceeded
from doc_harness.metric import build_metric
from doc_harness.optimize import (
    MINIBATCH_FLOOR,
    CountingMetric,
    OptimizeError,
    _compile,
    build_optimizer,
    class_coverage_order,
    demo_class_coverage,
    leaderboard_rows,
    next_experiment_id,
    read_champion,
    require_champion,
    run_experiment,
)
from doc_harness.registry import Registry


def _records(n: int, rare_at: int = 0) -> list[LabelRecord]:
    """Build labeled records where 'TX' appears exactly once, at ``rare_at``."""
    records = []
    for index in range(n):
        state = "TX" if index == rare_at else ("CA" if index % 2 else "NY")
        records.append(
            LabelRecord(
                doc_id=f"d{index:03d}",
                labels={"flag": index % 2 == 0, "state": state, "number": f"A-{index}"},
            )
        )
    return records


@pytest.fixture
def examples(toy_registry: Registry) -> list[Any]:
    records = _records(12)
    texts = {record.doc_id: document_for(record.doc_id) for record in records}
    return build_examples(records, texts)


@pytest.fixture
def answers(examples: list[Any]) -> dict[str, dict[str, Any]]:
    """Script the LM to answer every document correctly."""
    return {
        example.doc_id: {
            "flag": "true" if example.flag else "false",
            "state": example.state,
            "number": example.number,
        }
        for example in examples
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "labels.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "data" / "splits.json").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _config(**overrides: Any) -> Config:
    payload: dict[str, Any] = {
        "models": {"task": "fake/scripted", "reflection": "fake/scripted"},
        "optimization": {
            "optimizer": "BootstrapFewShot",
            "num_threads": 1,
            "max_bootstrapped_demos": 2,
            "max_labeled_demos": 2,
            "max_experiments": 3,
            "max_rollouts": 5000,
        },
        "splits": {"support_floor": 2, "measurable_floor": 2},
    }
    for key, value in overrides.items():
        payload.setdefault(key, {})
        payload[key].update(value)
    return Config.from_mapping(payload)


def test_class_coverage_order_front_loads_every_class(toy_registry: Registry, examples: list[Any]) -> None:
    """A rare class buried at the back never reaches a demonstration otherwise."""
    ordered = class_coverage_order(toy_registry, examples, seed=1)
    assert len(ordered) == len(examples)
    assert {example.doc_id for example in ordered} == {example.doc_id for example in examples}
    front_states = {example.state for example in ordered[:4]}
    assert {"TX", "CA", "NY"} <= front_states


def test_class_coverage_order_is_reproducible(toy_registry: Registry, examples: list[Any]) -> None:
    first = [e.doc_id for e in class_coverage_order(toy_registry, examples, seed=7)]
    second = [e.doc_id for e in class_coverage_order(toy_registry, examples, seed=7)]
    assert first == second


def test_class_coverage_warns_when_a_class_is_absent(
    toy_registry: Registry, examples: list[Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The enum declares TX; a trainset without it cannot teach it."""
    without_tx = [example for example in examples if example.state != "TX"]
    with caplog.at_level("INFO"):
        class_coverage_order(toy_registry, without_tx, seed=1)
    assert "reordered trainset" in caplog.text


def test_counting_metric_enforces_the_rollout_budget(toy_registry: Registry, examples: list[Any]) -> None:
    counting = CountingMetric(metric=build_metric(toy_registry), max_rollouts=2)
    counting(examples[0], examples[0])
    counting(examples[0], examples[0])
    with pytest.raises(RolloutBudgetExceeded, match="rollout budget of 2 is spent"):
        counting(examples[0], examples[0])
    assert counting.calls == 3


def test_unknown_optimizer_fails_loudly(toy_registry: Registry) -> None:
    config = _config()
    object.__setattr__(config.optimization, "optimizer", "Telepathy")
    counting = CountingMetric(metric=build_metric(toy_registry), max_rollouts=10)
    with pytest.raises(OptimizeError, match="unknown optimizer"):
        build_optimizer(config, counting)


@pytest.mark.parametrize(
    "optimizer",
    ["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2", "GEPA", "SIMBA"],
)
def test_every_supported_optimizer_constructs(toy_registry: Registry, optimizer: str) -> None:
    """Constructor signatures are verified against the installed DSPy, not assumed."""
    config = _config(optimization={"optimizer": optimizer})
    counting = CountingMetric(metric=build_metric(toy_registry), max_rollouts=10)
    assert build_optimizer(config, counting) is not None


def test_experiment_ids_increment(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    assert next_experiment_id(runs) == "exp_001"
    (runs / "exp_001").mkdir(parents=True)
    assert next_experiment_id(runs) == "exp_002"


def test_run_experiment_writes_everything(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    config = _config()
    with scripted_lm(answers):
        record, compiled, result = run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="baseline optimizer",
        )
    run_dir = project / "runs" / record.exp_id
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "failures.md").exists()
    assert (project / "programs" / "compiled" / f"{record.exp_id}.json").exists()

    stored = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert stored["config"]["variable"] == "baseline optimizer"
    assert stored["config"]["optimizer"] == "BootstrapFewShot"
    assert stored["results"]["rollouts"] > 0
    assert record.aggregate == pytest.approx(1.0)
    assert set(record.per_task) == {"flag", "state", "number"}


def test_leaderboard_gets_a_row_per_experiment(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    config = _config()
    with scripted_lm(answers):
        for variable in ("first", "second"):
            run_experiment(
                toy_registry,
                config,
                build_metric(toy_registry),
                trainset=examples[:8],
                valset=examples[8:],
                project_dir=project,
                variable=variable,
            )
    text = (project / "runs" / "leaderboard.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| exp_0")]
    assert len(rows) == 2
    assert "BootstrapFewShot" in text
    assert len(leaderboard_rows(project / "runs")) == 2


def test_champion_is_pinned_and_branched_from(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    config = _config()
    with scripted_lm(answers):
        first, _program, _result = run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="first",
        )
        champion = read_champion(project / "runs")
        assert champion is not None and champion["exp_id"] == first.exp_id

        second, _program2, _result2 = run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="second",
        )
    # the second experiment branched from the pinned champion, not from whatever ran last
    assert second.parent == first.exp_id


def test_experiment_budget_stops_the_run(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    config = _config(optimization={"max_experiments": 1})
    with scripted_lm(answers):
        run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="only one",
        )
        with pytest.raises(GuardError, match="experiment budget of 1 is spent"):
            run_experiment(
                toy_registry,
                config,
                build_metric(toy_registry),
                trainset=examples[:8],
                valset=examples[8:],
                project_dir=project,
                variable="one too many",
            )


def test_labels_stay_read_only_during_an_experiment(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    config = _config()
    before = (project / "data" / "labels.jsonl").read_bytes()
    with scripted_lm(answers):
        run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="check",
        )
    assert (project / "data" / "labels.jsonl").read_bytes() == before


def test_demo_coverage_is_reported(project: Path, toy_registry: Registry, examples: list[Any], answers: Any) -> None:
    config = _config()
    with scripted_lm(answers):
        record, compiled, _result = run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="coverage",
        )
    coverage = demo_class_coverage(toy_registry, compiled)
    assert set(coverage) == {"flag", "state"}
    # whatever the demos cover, any gap is written into the experiment record
    stored = json.loads((project / "runs" / record.exp_id / "config.json").read_text(encoding="utf-8"))
    assert isinstance(stored["notes"], list)


def test_require_champion_refuses_an_unoptimized_project(tmp_path: Path) -> None:
    with pytest.raises(GuardError, match="no champion is pinned"):
        require_champion(tmp_path / "runs")


def test_class_coverage_order_keeps_duplicate_examples(toy_registry: Registry) -> None:
    """Two examples with identical values are two examples, not one."""
    records = [LabelRecord(doc_id=f"d{i}", labels={"flag": True, "state": "CA", "number": "A"}) for i in range(4)]
    texts = {record.doc_id: document_for(record.doc_id) for record in records}
    examples = build_examples(records, texts)
    ordered = class_coverage_order(toy_registry, examples, seed=1)
    assert len(ordered) == len(examples)
    assert sorted(example.doc_id for example in ordered) == ["d0", "d1", "d2", "d3"]


def test_trial_and_candidate_budgets_reach_the_optimizer(toy_registry: Registry) -> None:
    """max_rollouts is a tripwire; num_trials and num_candidates are the actual brakes."""
    config = _config(optimization={"optimizer": "MIPROv2", "num_candidates": 2, "num_trials": 3})
    counting = CountingMetric(metric=build_metric(toy_registry), max_rollouts=10)
    optimizer = build_optimizer(config, counting)
    assert optimizer.num_candidates == 2

    random_search = build_optimizer(
        _config(optimization={"optimizer": "BootstrapFewShotWithRandomSearch", "num_candidates": 2}), counting
    )
    assert random_search.num_candidate_sets == 2

    simba = build_optimizer(_config(optimization={"optimizer": "SIMBA", "max_steps": 2, "num_candidates": 2}), counting)
    assert simba.max_steps == 2


def test_experiment_record_pins_the_budget(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    """One variable changes per experiment, and the run says what every knob was set to."""
    config = _config(optimization={"num_candidates": 2, "num_trials": 3})
    with scripted_lm(answers):
        record, _program, _result = run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="bounded budget",
        )
    stored = json.loads((project / "runs" / record.exp_id / "config.json").read_text(encoding="utf-8"))
    assert stored["config"]["num_trials"] == 3
    assert stored["config"]["num_candidates"] == 2
    assert stored["config"]["max_rollouts"] == config.optimization.max_rollouts


def test_small_validation_splits_skip_minibatching(caplog: pytest.LogCaptureFixture) -> None:
    """MIPROv2's default minibatch size exceeds a small valset, and that is a crash."""

    class Recording:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def compile(self, student: Any, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return student

    small = Recording()
    with caplog.at_level("INFO"):
        _compile(small, "MIPROv2", student="program", trainset=["t"] * 8, valset=["v"] * 4, num_trials=3)
    assert small.kwargs["minibatch"] is False
    assert small.kwargs["num_trials"] == 3
    assert small.kwargs["requires_permission_to_run"] is False
    assert "below the minibatch floor" in caplog.text

    large = Recording()
    _compile(large, "MIPROv2", student="program", trainset=["t"] * 8, valset=["v"] * MINIBATCH_FLOOR)
    assert large.kwargs["minibatch"] is True


def test_compile_passes_only_what_each_optimizer_takes() -> None:
    """Optimizer compile signatures differ; passing a valset to BootstrapFewShot is an error."""

    class Recording:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def compile(self, student: Any, **kwargs: Any) -> Any:
            self.kwargs = kwargs
            return student

    bootstrap = Recording()
    _compile(bootstrap, "BootstrapFewShot", student="p", trainset=["t"], valset=["v"])
    assert set(bootstrap.kwargs) == {"trainset"}

    simba = Recording()
    _compile(simba, "SIMBA", student="p", trainset=["t"], valset=["v"])
    assert set(simba.kwargs) == {"trainset"}

    gepa = Recording()
    _compile(gepa, "GEPA", student="p", trainset=["t"], valset=["v"])
    assert set(gepa.kwargs) == {"trainset", "valset"}


def test_a_spent_rollout_budget_stops_the_run_cleanly(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    """Reported from a real run: the guard raised inside DSPy, which logged it and carried on."""
    config = _config(optimization={"max_rollouts": 1})
    with scripted_lm(answers), pytest.raises(GuardError, match="rollout budget of 1 is spent"):
        run_experiment(
            toy_registry,
            config,
            build_metric(toy_registry),
            trainset=examples[:8],
            valset=examples[8:],
            project_dir=project,
            variable="a budget too small to finish",
        )
    assert not (project / "runs" / "champion.json").exists(), "a run that was stopped must record nothing"


def test_a_champion_of_another_module_type_is_not_branched_from(
    project: Path, toy_registry: Registry, examples: list[Any], answers: Any
) -> None:
    """Reported from a real run: compile crashed loading a predict champion as chain_of_thought."""
    # a chain-of-thought program asks for its reasoning field too
    reasoned = {doc_id: {"reasoning": "the document says so", **values} for doc_id, values in answers.items()}
    with scripted_lm(reasoned):
        first, _program, _result = run_experiment(
            toy_registry, _config(), build_metric(toy_registry), trainset=examples[:8], valset=examples[8:],
            project_dir=project, variable="predict champion",
        )  # fmt: skip
        champion = json.loads((project / "runs" / "champion.json").read_text(encoding="utf-8"))
        assert champion["module"] == "predict"

        second, _program, _result = run_experiment(
            toy_registry, _config(optimization={"module": "chain_of_thought"}), build_metric(toy_registry),
            trainset=examples[:8], valset=examples[8:], project_dir=project, variable="chain of thought",
        )  # fmt: skip
    assert second.parent is None, f"{second.exp_id} branched from a {champion['module']} champion"
    assert first.exp_id != second.exp_id
