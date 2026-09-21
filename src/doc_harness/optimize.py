"""Optimizer runs, the leaderboard, and the budget that bounds them.

One variable changes per experiment, every experiment writes down what it was, and the
champion is pinned so later experiments branch from the best program rather than from
whatever happened to run last.

Two things here are less obvious than they look:

* the trainset is reordered so every class appears early. Bootstrapped selection otherwise
  omits rare classes entirely, and the compiled program then behaves as if they do not
  exist -- while the aggregate barely moves, because rare classes are rare.
* rollouts are counted by wrapping the metric, so the budget is enforced against what
  actually happened rather than against an estimate.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from doc_harness.config import Config
from doc_harness.evaluate import EvaluationResult, run_program, score_split, write_run
from doc_harness.guards import GuardError, check_experiment_budget, check_rollout_budget, readonly
from doc_harness.metric import Metric, get_field
from doc_harness.program import build_program, instructions_of, load_program, save_program
from doc_harness.registry import Registry, TaskType
from doc_harness.splits import CLASSIFICATION_TYPES, _labels_of

logger = logging.getLogger(__name__)

# MIPROv2 minibatches its trials by default at a size that exceeds a small validation
# split, which is a crash rather than a warning. Below this, evaluate the split in full --
# it is cheap at that size, and it is also the only way to use every example.
MINIBATCH_FLOOR = 50

CHAMPION_FILE = "champion.json"
LEADERBOARD_FILE = "leaderboard.md"

OPTIMIZERS = (
    "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch",
    "MIPROv2",
    "GEPA",
    "SIMBA",
)


class OptimizeError(RuntimeError):
    """Raised when an experiment cannot be run as configured."""


@dataclass
class CountingMetric:
    """Wraps the metric so rollouts are counted against the budget as they happen."""

    metric: Metric
    max_rollouts: int
    calls: int = 0

    def __call__(self, gold: Any, pred: Any, trace: Any = None, *args: Any, **kwargs: Any) -> Any:
        """Score one example, refusing once the rollout budget is spent."""
        self.calls += 1
        check_rollout_budget(self.calls - 1, self.max_rollouts)
        return self.metric(gold, pred, trace)

    def gepa(self, gold: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None) -> Any:
        """Score with feedback, counted the same way."""
        self.calls += 1
        check_rollout_budget(self.calls - 1, self.max_rollouts)
        return self.metric.gepa_metric(gold, pred, trace, pred_name, pred_trace)


@dataclass
class ExperimentRecord:
    """What one experiment fixed, and what it produced."""

    exp_id: str
    optimizer: str
    config: dict[str, Any]
    aggregate: float
    per_task: dict[str, float]
    rollouts: int
    wall_seconds: float
    parent: str | None = None
    notes: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_dict(self) -> dict[str, Any]:
        """Return the config.json payload for this experiment."""
        return {
            "exp_id": self.exp_id,
            "optimizer": self.optimizer,
            "parent": self.parent,
            "started_at": self.started_at,
            "config": self.config,
            "results": {
                "aggregate": self.aggregate,
                "per_task": self.per_task,
                "rollouts": self.rollouts,
                "wall_seconds": self.wall_seconds,
            },
            "notes": self.notes,
        }


def next_experiment_id(runs_dir: Path) -> str:
    """Return the next free experiment id, in run order."""
    existing = sorted(path.name for path in runs_dir.glob("exp_*") if path.is_dir()) if runs_dir.exists() else []
    return f"exp_{len(existing) + 1:03d}"


def class_coverage_order(
    registry: Registry,
    examples: Sequence[Any],
    seed: int = 0,
) -> list[Any]:
    """Reorder a trainset so the first examples cover every class at least once.

    ``BootstrapFewShot`` walks the trainset in order and stops once it has enough
    demonstrations. Left alone it draws them from whatever is common, so a rare class never
    appears in a demonstration and the compiled program stops predicting it at all.

    :param registry: The parsed tasks.yaml
    :param examples: The training examples
    :param seed: Fixed so the ordering is reproducible
    :returns: The same examples, reordered
    """
    import random

    classification = [task for task in registry if TaskType(task.type) in CLASSIFICATION_TYPES]
    if not classification:
        return list(examples)

    remaining = list(examples)
    rng = random.Random(seed)
    rng.shuffle(remaining)

    needed: set[tuple[str, str]] = set()
    for task in classification:
        for example in remaining:
            for label in _labels_of(get_field(example, task.id)):
                needed.add((task.id, label))

    front: list[Any] = []
    # tracked by identity, not equality: two examples can carry the same values, and dropping
    # one of them as a duplicate would silently shrink the trainset
    chosen: set[int] = set()
    covered: set[tuple[str, str]] = set()
    for target in sorted(needed):
        if target in covered:
            continue
        task_id, label = target
        carrier = next(
            (
                example
                for example in remaining
                if id(example) not in chosen and label in _labels_of(get_field(example, task_id))
            ),
            None,
        )
        if carrier is None:
            continue
        front.append(carrier)
        chosen.add(id(carrier))
        for task in classification:
            for carried in _labels_of(get_field(carrier, task.id)):
                covered.add((task.id, carried))

    tail = [example for example in remaining if id(example) not in chosen]
    missing = sorted(needed - covered)
    if missing:
        logger.warning(
            "no training example carries %s; the compiled program cannot learn to predict it",
            ", ".join(f"{task_id}={label}" for task_id, label in missing),
        )
    logger.info("reordered trainset so %d class(es) appear in the first %d examples", len(covered), len(front))
    return front + tail


def demo_class_coverage(registry: Registry, program: Any) -> dict[str, set[str]]:
    """Return which classes actually made it into the compiled demonstrations."""
    classification = {task.id for task in registry if TaskType(task.type) in CLASSIFICATION_TYPES}
    found: dict[str, set[str]] = {task_id: set() for task_id in classification}
    for _name, predictor in program.named_predictors():
        for demo in getattr(predictor, "demos", []) or []:
            for task_id in classification:
                for label in _labels_of(get_field(demo, task_id)):
                    found[task_id].add(label)
    return found


def build_optimizer(config: Config, metric: CountingMetric) -> Any:
    """Construct the optimizer named in config.yaml.

    :param config: The project configuration
    :param metric: The rollout-counting metric wrapper
    :returns: A configured DSPy optimizer
    """
    import dspy
    from dspy.teleprompt import (
        GEPA,
        SIMBA,
        BootstrapFewShot,
        BootstrapFewShotWithRandomSearch,
        MIPROv2,
    )

    settings = config.optimization
    name = settings.optimizer
    if name == "BootstrapFewShot":
        return BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=settings.max_bootstrapped_demos,
            max_labeled_demos=settings.max_labeled_demos,
        )
    if name == "BootstrapFewShotWithRandomSearch":
        return BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=settings.max_bootstrapped_demos,
            max_labeled_demos=settings.max_labeled_demos,
            num_candidate_programs=settings.num_candidates or 16,
            num_threads=settings.num_threads,
        )
    if name == "MIPROv2":
        # auto and the explicit budgets are mutually exclusive: MIPROv2 refuses to construct
        # with both, because auto would silently overwrite whatever was asked for
        explicit = settings.num_candidates is not None
        return MIPROv2(
            metric=metric,
            auto=None if explicit else settings.auto,
            max_bootstrapped_demos=settings.max_bootstrapped_demos,
            max_labeled_demos=settings.max_labeled_demos,
            num_candidates=settings.num_candidates,
            num_threads=settings.num_threads,
            seed=config.splits.seed,
            # both are required: MIPROv2 refuses to construct with only one of them set
            prompt_model=dspy.LM(config.models.require_reflection()),
            task_model=dspy.LM(config.models.require_task()),
        )
    if name == "GEPA":
        return GEPA(
            metric=metric.gepa,
            auto=settings.auto,
            reflection_lm=dspy.LM(config.models.require_reflection()),
            num_threads=settings.num_threads,
            track_stats=True,
        )
    if name == "SIMBA":
        return SIMBA(
            metric=metric,
            max_demos=settings.max_bootstrapped_demos,
            max_steps=settings.max_steps,
            num_candidates=settings.num_candidates or 6,
            num_threads=settings.num_threads,
        )
    raise OptimizeError(f"unknown optimizer {name!r}; supported: {', '.join(OPTIMIZERS)}")


def _compile(
    optimizer: Any,
    name: str,
    student: Any,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    num_trials: int | None = None,
) -> Any:
    """Call the optimizer's compile with the arguments that optimizer actually takes."""
    if name in {"BootstrapFewShot", "SIMBA"}:
        return optimizer.compile(student, trainset=list(trainset))
    if name == "MIPROv2":
        minibatch = len(valset) >= MINIBATCH_FLOOR
        if not minibatch:
            logger.info(
                "validation split has %d examples, below the minibatch floor of %d: "
                "evaluating it in full on every trial",
                len(valset),
                MINIBATCH_FLOOR,
            )
        return optimizer.compile(
            student,
            trainset=list(trainset),
            valset=list(valset),
            num_trials=num_trials,
            minibatch=minibatch,
            requires_permission_to_run=False,
        )
    return optimizer.compile(student, trainset=list(trainset), valset=list(valset))


def read_champion(runs_dir: Path) -> dict[str, Any] | None:
    """Return the pinned champion, if one has been recorded."""
    path = runs_dir / CHAMPION_FILE
    if not path.exists():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def pin_champion(runs_dir: Path, record: ExperimentRecord, program_path: Path) -> None:
    """Pin an experiment as the champion that later experiments branch from."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp_id": record.exp_id,
        "optimizer": record.optimizer,
        "aggregate": record.aggregate,
        "per_task": record.per_task,
        "program": str(program_path),
        "pinned_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (runs_dir / CHAMPION_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("pinned %s as champion at %.3f", record.exp_id, record.aggregate)


def append_leaderboard(runs_dir: Path, record: ExperimentRecord) -> None:
    """Append one row to runs/leaderboard.md, creating it with a header if needed."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / LEADERBOARD_FILE
    if not path.exists():
        header = [
            "# Leaderboard",
            "",
            "One row per experiment. One variable changes per experiment; the `config` column",
            "names it. Validation scores only -- the holdout appears in REPORT.md and nowhere else.",
            "",
            "| exp_id | optimizer | aggregate | per-task | rollouts | wall (s) | parent |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        path.write_text("\n".join(header) + "\n", encoding="utf-8")
    per_task = ", ".join(f"{task_id} {value:.3f}" for task_id, value in record.per_task.items())
    row = (
        f"| {record.exp_id} | {record.optimizer} | {record.aggregate:.3f} | {per_task} | "
        f"{record.rollouts} | {record.wall_seconds:.0f} | {record.parent or '-'} |"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")


def run_experiment(
    registry: Registry,
    config: Config,
    metric: Metric,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    project_dir: Path,
    variable: str,
    branch_from_champion: bool = True,
    instructions: Mapping[str, str] | None = None,
) -> tuple[ExperimentRecord, Any, EvaluationResult]:
    """Run one optimization experiment end to end.

    :param registry: The parsed tasks.yaml
    :param config: The project configuration
    :param metric: The metric built from the registry
    :param trainset: The training examples
    :param valset: The validation examples
    :param project_dir: The project root
    :param variable: The single thing this experiment changes, recorded in the run
    :param branch_from_champion: Start from the pinned champion rather than from scratch
    :param instructions: Optional starting instructions per group
    :returns: The experiment record, the compiled program and its validation result
    """
    runs_dir = project_dir / "runs"
    data_dir = project_dir / "data"
    check_experiment_budget(runs_dir, config.optimization.max_experiments)

    exp_id = next_experiment_id(runs_dir)
    champion = read_champion(runs_dir) if branch_from_champion else None
    parent = champion["exp_id"] if champion else None

    if champion and Path(champion["program"]).exists():
        student = load_program(registry, Path(champion["program"]), module_type=config.optimization.module)
        logger.info("%s branches from champion %s", exp_id, parent)
    else:
        student = build_program(registry, module_type=config.optimization.module, instructions=instructions)

    ordered_train = class_coverage_order(registry, trainset, seed=config.splits.seed)
    counting = CountingMetric(metric=metric, max_rollouts=config.optimization.max_rollouts)
    optimizer = build_optimizer(config, counting)

    protected = [data_dir / "labels.jsonl", data_dir / "splits.json"]
    started = time.monotonic()
    with readonly(protected, reason="labels.jsonl and splits.json are read-only to every optimization path"):
        compiled = _compile(
            optimizer,
            config.optimization.optimizer,
            student,
            ordered_train,
            valset,
            num_trials=config.optimization.num_trials,
        )
        predictions = run_program(
            compiled,
            valset,
            metric,
            num_threads=config.optimization.num_threads,
            max_retries=config.evaluation.max_retries,
            max_failure_rate=config.evaluation.max_failure_rate,
        )
    wall_seconds = time.monotonic() - started

    result = score_split(
        registry,
        metric,
        valset,
        predictions,
        metadata={
            "run": exp_id,
            "split": "val",
            "optimizer": config.optimization.optimizer,
            "module": config.optimization.module,
            "task_model": config.models.task,
            "reflection_model": config.models.reflection,
            "config_hash": config.fingerprint(),
            "instructions": instructions_of(compiled),
            "variable": variable,
            "parent": parent,
        },
        support_floor=config.splits.support_floor,
        measurable_floor=config.splits.measurable_floor,
    )

    notes = _coverage_notes(registry, compiled)
    record = ExperimentRecord(
        exp_id=exp_id,
        optimizer=config.optimization.optimizer,
        config={
            "optimizer": config.optimization.optimizer,
            "auto": config.optimization.auto,
            "module": config.optimization.module,
            "groups": sorted(registry.groups()),
            "max_bootstrapped_demos": config.optimization.max_bootstrapped_demos,
            "max_labeled_demos": config.optimization.max_labeled_demos,
            "num_candidates": config.optimization.num_candidates,
            "num_trials": config.optimization.num_trials,
            "max_rollouts": config.optimization.max_rollouts,
            "task_model": config.models.task,
            "reflection_model": config.models.reflection,
            "trainset_size": len(trainset),
            "valset_size": len(valset),
            "variable": variable,
            "config_hash": config.fingerprint(),
        },
        aggregate=result.aggregate,
        per_task=result.per_task_primary,
        rollouts=counting.calls,
        wall_seconds=wall_seconds,
        parent=parent,
        notes=notes,
    )

    run_dir = runs_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(record.to_dict(), indent=2) + "\n", encoding="utf-8")
    write_run(run_dir, registry, result, valset, predictions)
    program_path = project_dir / "programs" / "compiled" / f"{exp_id}.json"
    save_program(compiled, program_path)
    append_leaderboard(runs_dir, record)

    if champion is None or result.aggregate > float(champion["aggregate"]):
        pin_champion(runs_dir, record, program_path)
    else:
        logger.info(
            "%s scored %.3f against champion %s at %.3f; champion unchanged",
            exp_id,
            result.aggregate,
            champion["exp_id"],
            float(champion["aggregate"]),
        )
    return record, compiled, result


def _coverage_notes(registry: Registry, compiled: Any) -> list[str]:
    """Note any class that never made it into a demonstration."""
    notes: list[str] = []
    coverage = demo_class_coverage(registry, compiled)
    for task in registry:
        if TaskType(task.type) not in CLASSIFICATION_TYPES:
            continue
        declared = set(task.enum_members or ["true", "false"])
        missing = sorted(declared - coverage.get(task.id, set()))
        if missing:
            notes.append(f"{task.id}: no demonstration covers {', '.join(missing)}")
    return notes


def leaderboard_rows(runs_dir: Path) -> list[dict[str, Any]]:
    """Read every experiment's record, for `status` and the final report."""
    rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("exp_*/config.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def require_champion(runs_dir: Path) -> dict[str, Any]:
    """Return the pinned champion, refusing to continue without one."""
    champion = read_champion(runs_dir)
    if champion is None:
        raise GuardError("no champion is pinned; run at least one experiment before evaluating or producing")
    return champion
