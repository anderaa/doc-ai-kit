"""The holdout reading, REPORT.md, and the shared ledger.

The holdout is one number, measured once. What makes it useful is not the number but the
gap between it and the validation number, read against a fixed table rather than against
whatever the reader hopes is true:

| gap (val - holdout) | reading | action |
| --- | --- | --- |
| within CI | no detectable overfitting | ship |
| modest, consistent | mild overfitting, normal | ship, quote holdout |
| large, few tasks | those tasks memorized specifics | revert them to baseline |
| large, across the board | fitted the validation split | fall back to a simpler champion |

The report also names the tasks the holdout **cannot** measure, so nobody downstream builds
on a number that rests on three documents.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from doc_harness import __version__
from doc_harness.config import Config
from doc_harness.evaluate import EvaluationResult, run_program, score_split, task_primary_ci, write_run
from doc_harness.guards import HoldoutLock, open_holdout
from doc_harness.metric import Metric
from doc_harness.registry import Registry
from doc_harness.stats import readable_at

logger = logging.getLogger(__name__)

# a gap at or below this is ordinary; above it, the program learned something split-specific
MODEST_GAP = 0.05
LARGE_GAP = 0.10
# when more than this share of tasks show a large gap, the problem is the champion, not a task
WIDESPREAD_SHARE = 0.5

Verdict = Literal["within_ci", "modest", "large_few", "large_widespread"]

READINGS: dict[Verdict, tuple[str, str]] = {
    "within_ci": ("no detectable overfitting", "ship"),
    "modest": ("mild overfitting, normal", "ship, quoting the holdout number"),
    "large_few": ("these tasks memorized split specifics", "revert them to baseline"),
    "large_widespread": ("the champion fitted the validation split", "fall back to a simpler champion"),
}

LEDGER_COLUMNS = (
    "closed_at",
    "project",
    "harness_version",
    "task_types",
    "n_documents",
    "k_labeled",
    "baseline_holdout",
    "compiled_holdout",
    "labeling_hours",
    "cost_usd",
    "notes",
)


@dataclass(frozen=True)
class TaskGap:
    """One task's validation number, holdout number and the distance between them."""

    task_id: str
    validation: float
    holdout: float
    holdout_ci: tuple[float, float]
    support: int
    measurable: bool

    @property
    def gap(self) -> float:
        """Return validation minus holdout; positive means the holdout is worse."""
        return self.validation - self.holdout

    @property
    def within_ci(self) -> bool:
        """Return whether the validation number falls inside the holdout's interval."""
        return self.holdout_ci[0] <= self.validation <= self.holdout_ci[1]

    @property
    def is_large(self) -> bool:
        """Return whether this gap is large enough to act on."""
        return not self.within_ci and self.gap > LARGE_GAP


@dataclass
class HoldoutReport:
    """Everything the one-shot holdout measurement produced."""

    result: EvaluationResult
    gaps: list[TaskGap]
    verdict: Verdict
    lock: HoldoutLock
    validation_aggregate: float
    unmeasurable: list[str] = field(default_factory=list)

    @property
    def reading(self) -> str:
        """Return the plain-language reading of the gap."""
        return READINGS[self.verdict][0]

    @property
    def action(self) -> str:
        """Return what the table says to do about it."""
        return READINGS[self.verdict][1]

    @property
    def large_gap_tasks(self) -> list[str]:
        """Return the tasks whose holdout number fell well short of validation."""
        return [gap.task_id for gap in self.gaps if gap.is_large]

    def to_markdown(self) -> str:
        """Render the holdout section of REPORT.md."""
        aggregate_low, aggregate_high = self.result.aggregate_ci
        lines = [
            "# Holdout",
            "",
            f"Measured once on {len(self.result.scores)} documents "
            f"({self.result.metadata.get('generated_at', '')}).",
            "",
            f"Aggregate **{self.result.aggregate:.3f}** "
            f"(95% CI {aggregate_low:.3f}-{aggregate_high:.3f}) "
            f"against {self.validation_aggregate:.3f} on validation, "
            f"a gap of {self.validation_aggregate - self.result.aggregate:+.3f}.",
            "",
            f"Reading: **{self.reading}**. Action: **{self.action}**.",
            "",
            "| task | holdout | 95% CI | validation | gap | support | reads as |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for gap in self.gaps:
            metrics = self.result.tasks[gap.task_id]
            lines.append(
                f"| {gap.task_id} | {gap.holdout:.3f} | "
                f"{gap.holdout_ci[0]:.3f}-{gap.holdout_ci[1]:.3f} | "
                f"{gap.validation:.3f} | {gap.gap:+.3f} | {gap.support} | "
                f"{'measurable' if gap.measurable else readable_at(gap.support)} |"
            )
        lines.append("")
        lines += [
            "Per task, precision and recall with their intervals:",
            "",
            "| task | P | P 95% CI | R | R 95% CI | F1 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for task_id, metrics in self.result.tasks.items():
            lines.append(
                f"| {task_id} | {metrics.precision:.3f} | "
                f"{metrics.precision_ci[0]:.3f}-{metrics.precision_ci[1]:.3f} | "
                f"{metrics.recall:.3f} | {metrics.recall_ci[0]:.3f}-{metrics.recall_ci[1]:.3f} | "
                f"{metrics.f1:.3f} |"
            )
        lines.append("")
        if self.large_gap_tasks:
            lines += [
                "## Tasks with a large gap",
                "",
                "These learned something specific to the validation split: "
                + ", ".join(f"`{task_id}`" for task_id in self.large_gap_tasks)
                + ".",
                "",
            ]
        if self.unmeasurable:
            lines += [
                "## What this holdout cannot measure",
                "",
                "The rarest class in each of these tasks has too few examples in the holdout for the",
                "number above to mean much. Do not quote them on their own.",
                "",
            ]
            for task_id in self.unmeasurable:
                metrics = self.result.tasks[task_id]
                note = metrics.notes[0] if metrics.notes else "support is too low"
                lines.append(f"- **{task_id}**: {note}")
            lines.append("")
        if self.lock.was_overridden:
            lines += [
                "## Holdout override",
                "",
                "This holdout has been evaluated more than once. Every number above is correspondingly",
                "weaker as an out-of-sample estimate.",
                "",
            ]
            for override in self.lock.overrides:
                lines.append(f"- {override['at']} by {override['by']}: {override['reason']}")
            lines.append("")
        return "\n".join(lines)


def read_gaps(gaps: Sequence[TaskGap]) -> Verdict:
    """Read the whole set of gaps against the table.

    :param gaps: One gap per task
    :returns: Which row of the table applies
    """
    if not gaps:
        return "within_ci"
    large = [gap for gap in gaps if gap.is_large]
    if not large:
        if all(gap.within_ci for gap in gaps):
            return "within_ci"
        return "modest"
    if len(large) / len(gaps) > WIDESPREAD_SHARE:
        return "large_widespread"
    return "large_few"


def run_holdout(
    registry: Registry,
    config: Config,
    metric: Metric,
    program: Any,
    holdout_examples: Sequence[Any],
    project_dir: Path,
    validation_aggregate: float,
    validation_per_task: Mapping[str, float],
    opened_by: str,
    override: bool = False,
    reason: str = "",
) -> HoldoutReport:
    """Evaluate the holdout once and read the result against the gap table.

    :param registry: The parsed tasks.yaml
    :param config: The project configuration
    :param metric: The metric built from the registry
    :param program: The compiled champion program
    :param holdout_examples: The blind-labeled holdout examples
    :param project_dir: The project root
    :param validation_aggregate: The champion's recorded validation aggregate
    :param validation_per_task: The champion's recorded per-task validation vector
    :param opened_by: What is opening the holdout, recorded in the lock and the report
    :param override: Whether this is a deliberate second evaluation
    :param reason: Why, required when overriding
    :returns: The holdout report
    """
    holdout_dir = project_dir / "runs" / "holdout"
    # predictions come before the lock: if too many replies fail, the run refuses before any
    # number exists, and a refusal must not spend the one-shot measurement. Nothing is scored
    # or written until the lock is held, so no holdout number can be seen without taking it.
    predictions = run_program(
        program,
        holdout_examples,
        metric,
        num_threads=config.optimization.num_threads,
        max_retries=config.evaluation.max_retries,
        max_failure_rate=config.evaluation.max_failure_rate,
    )
    lock = open_holdout(holdout_dir, opened_by=opened_by, override=override, reason=reason)
    result = score_split(
        registry,
        metric,
        holdout_examples,
        predictions,
        metadata={
            "run": "holdout",
            "split": "holdout",
            "module": config.optimization.module,
            "task_model": config.models.task,
            "config_hash": config.fingerprint(),
            "opened_by": opened_by,
            "overrides": lock.overrides,
        },
        support_floor=config.splits.support_floor,
        measurable_floor=config.splits.measurable_floor,
        seed=config.splits.seed,
    )
    write_run(holdout_dir, registry, result, holdout_examples, predictions)

    gaps = []
    for task_id, metrics in result.tasks.items():
        gaps.append(
            TaskGap(
                task_id=task_id,
                validation=validation_per_task.get(task_id, 0.0),
                holdout=metrics.primary_value,
                holdout_ci=task_primary_ci(
                    registry,
                    task_id,
                    result.scores,
                    support_floor=config.splits.support_floor,
                    measurable_floor=config.splits.measurable_floor,
                    resamples=config.metric.bootstrap_resamples // 2,
                    seed=config.splits.seed,
                ),
                support=metrics.support,
                measurable=metrics.measurable,
            )
        )
    report = HoldoutReport(
        result=result,
        gaps=gaps,
        verdict=read_gaps(gaps),
        lock=lock,
        validation_aggregate=validation_aggregate,
        unmeasurable=[task_id for task_id, metrics in result.tasks.items() if not metrics.measurable],
    )
    logger.info(
        "holdout aggregate %.3f against validation %.3f: %s -> %s",
        result.aggregate,
        validation_aggregate,
        report.reading,
        report.action,
    )
    return report


def write_report(project_dir: Path, sections: Sequence[str], title: str = "Project report") -> Path:
    """Assemble REPORT.md from its sections.

    :param project_dir: The project root
    :param sections: Markdown blocks, in the order they should appear
    :param title: The report title
    :returns: The path written
    """
    path = project_dir / "REPORT.md"
    header = [
        f"# {title}",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} " f"by doc-harness {__version__}.",
        "",
    ]
    path.write_text("\n".join(header) + "\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def append_ledger(ledger_path: Path, row: Mapping[str, Any]) -> None:
    """Append one project to the shared ledger.

    :param ledger_path: The shared ledger.csv
    :param row: The values to record; unknown keys are rejected rather than dropped
    """
    unknown = sorted(set(row) - set(LEDGER_COLUMNS))
    if unknown:
        raise ValueError(f"unknown ledger column(s): {', '.join(unknown)}")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not ledger_path.exists()
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LEDGER_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in LEDGER_COLUMNS})
    logger.info("appended %s to %s", row.get("project", "(unnamed)"), ledger_path)
