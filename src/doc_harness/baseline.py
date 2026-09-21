"""The three baselines a project records before it optimizes anything.

Three, not one: zero-shot, hand-written few-shot, and ``BootstrapFewShot``. The compiled
program has to beat the best of them on the holdout, or the honest move is to ship the
baseline. Recording only the weakest baseline makes any later number look like progress.

The baseline report also names the tasks scoring near zero. Those are upstream problems --
information absent from the extracted text, an ambiguous definition, or a broken matcher --
and none of them is fixed by optimization.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc_harness.config import Config
from doc_harness.evaluate import EvaluationResult, run_program, score_split, write_run
from doc_harness.metric import Metric
from doc_harness.program import build_program, instructions_of
from doc_harness.registry import Registry

logger = logging.getLogger(__name__)

# a task at or below this is not an optimization problem
NEAR_ZERO = 0.05

ZERO_SHOT = "zero_shot"
FEW_SHOT = "few_shot"
BOOTSTRAP = "bootstrap_few_shot"
BASELINE_NAMES: tuple[str, ...] = (ZERO_SHOT, FEW_SHOT, BOOTSTRAP)


@dataclass
class BaselineRun:
    """One baseline variant and how it scored."""

    name: str
    program: Any
    result: EvaluationResult
    note: str = ""

    @property
    def score(self) -> float:
        """Return the aggregate validation score."""
        return self.result.aggregate


@dataclass
class BaselineReport:
    """Every baseline, and what they say about the project."""

    runs: list[BaselineRun]
    upstream_problems: dict[str, float]

    @property
    def best(self) -> BaselineRun:
        """Return the strongest baseline: the bar a compiled program has to clear."""
        return max(self.runs, key=lambda run: run.score)

    def to_markdown(self) -> str:
        """Render the baseline section of the report."""
        lines = [
            "# Baselines",
            "",
            "| baseline | aggregate | " + " | ".join(self.runs[0].result.per_task_primary) + " |",
            "| --- | --- |" + " --- |" * len(self.runs[0].result.per_task_primary),
        ]
        for run in self.runs:
            per_task = " | ".join(f"{value:.3f}" for value in run.result.per_task_primary.values())
            lines.append(f"| {run.name} | {run.score:.3f} | {per_task} |")
        lines += [
            "",
            f"Best baseline: **{self.best.name}** at {self.best.score:.3f}. "
            "A compiled program has to beat this on the holdout, or shipping the baseline is the honest move.",
            "",
        ]
        if self.upstream_problems:
            lines += [
                "## Tasks scoring near zero",
                "",
                "These are upstream problems, not optimization problems. Check, in order: is the answer",
                "present in the extracted text at all; is the question unambiguous; is the matcher correct.",
                "",
            ]
            for task_id, value in sorted(self.upstream_problems.items(), key=lambda item: item[1]):
                lines.append(f"- **{task_id}**: {value:.3f}")
            lines.append("")
        return "\n".join(lines)


def _evaluate(
    name: str,
    program: Any,
    registry: Registry,
    metric: Metric,
    examples: Sequence[Any],
    config: Config,
    runs_dir: Path,
    note: str = "",
) -> BaselineRun:
    """Run one baseline over the validation split and write its run directory."""
    predictions = run_program(
        program,
        examples,
        metric,
        num_threads=config.optimization.num_threads,
        max_retries=config.evaluation.max_retries,
        max_failure_rate=config.evaluation.max_failure_rate,
    )
    result = score_split(
        registry,
        metric,
        examples,
        predictions,
        metadata={
            "run": f"baseline_{name}",
            "baseline": name,
            "split": "val",
            "module": config.optimization.module,
            "task_model": config.models.task,
            "config_hash": config.fingerprint(),
            "instructions": instructions_of(program),
        },
        support_floor=config.splits.support_floor,
        measurable_floor=config.splits.measurable_floor,
    )
    run_dir = runs_dir / f"baseline_{name}"
    write_run(run_dir, registry, result, examples, predictions)
    logger.info("baseline %s scored %.3f on validation", name, result.aggregate)
    return BaselineRun(name=name, program=program, result=result, note=note)


def run_baselines(
    registry: Registry,
    config: Config,
    metric: Metric,
    trainset: Sequence[Any],
    valset: Sequence[Any],
    runs_dir: Path,
    demos: Mapping[str, Sequence[Any]] | None = None,
    instructions: Mapping[str, str] | None = None,
) -> BaselineReport:
    """Record all three baselines against the validation split.

    :param registry: The parsed tasks.yaml
    :param config: The project configuration
    :param metric: The metric built from the registry
    :param trainset: The training examples, used only by the bootstrap baseline
    :param valset: The validation examples every baseline is scored on
    :param runs_dir: The project's runs directory
    :param demos: Hand-written demonstrations per group, for the few-shot baseline
    :param instructions: Optional starting instructions per group
    :returns: The three runs plus any upstream problems found
    """
    from dspy.teleprompt import BootstrapFewShot

    module_type = config.optimization.module
    runs: list[BaselineRun] = []

    zero_shot = build_program(registry, module_type=module_type, instructions=instructions)
    runs.append(_evaluate(ZERO_SHOT, zero_shot, registry, metric, valset, config, runs_dir))

    if demos:
        few_shot = build_program(registry, module_type=module_type, instructions=instructions, demos=demos)
        runs.append(_evaluate(FEW_SHOT, few_shot, registry, metric, valset, config, runs_dir))
    else:
        logger.warning(
            "no hand-written demonstrations supplied; the few-shot baseline is skipped. "
            "Add them in programs/baseline.py so the compiled program has a real bar to clear"
        )

    student = build_program(registry, module_type=module_type, instructions=instructions)
    bootstrap = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=config.optimization.max_bootstrapped_demos,
        max_labeled_demos=config.optimization.max_labeled_demos,
    )
    compiled = bootstrap.compile(student, trainset=list(trainset))
    runs.append(
        _evaluate(
            BOOTSTRAP,
            compiled,
            registry,
            metric,
            valset,
            config,
            runs_dir,
            note=f"bootstrapped from {len(trainset)} training examples",
        )
    )

    upstream = find_upstream_problems(runs)
    if upstream:
        logger.warning(
            "%d task(s) score near zero at every baseline: %s. This is an extraction, definition "
            "or matcher problem, and optimization will not fix it",
            len(upstream),
            ", ".join(sorted(upstream)),
        )
    return BaselineReport(runs=runs, upstream_problems=upstream)


def find_upstream_problems(runs: Sequence[BaselineRun], threshold: float = NEAR_ZERO) -> dict[str, float]:
    """Return tasks that score near zero across every baseline.

    :param runs: The baseline runs
    :param threshold: The score at or below which a task counts as broken rather than weak
    :returns: Task id to its best score across the baselines
    """
    if not runs:
        return {}
    best_per_task: dict[str, float] = {}
    for run in runs:
        for task_id, value in run.result.per_task_primary.items():
            best_per_task[task_id] = max(best_per_task.get(task_id, 0.0), value)
    return {task_id: value for task_id, value in best_per_task.items() if value <= threshold}
