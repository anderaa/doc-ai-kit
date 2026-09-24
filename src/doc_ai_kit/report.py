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
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from doc_ai_kit import __version__
from doc_ai_kit.config import Config
from doc_ai_kit.evaluate import EvaluationResult, run_program, score_split, task_primary_ci, write_run
from doc_ai_kit.guards import HoldoutLock, open_holdout
from doc_ai_kit.metric import Metric
from doc_ai_kit.registry import Registry
from doc_ai_kit.stats import readable_at, wilson_interval

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
    "package_version",
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
        """Render the holdout section of REPORT.md, for a reader who did not build the program."""
        aggregate_low, aggregate_high = self.result.aggregate_ci
        difference = self.validation_aggregate - self.result.aggregate
        against_tuning = "the same" if abs(difference) < 0.005 else f"a difference of {difference:+.3f}"
        lines = [
            "# How well it works",
            "",
            f"These numbers come from **{len(self.result.scores)} documents the program never saw while it "
            "was being built**. They were set aside at the start, labeled by hand, and used once, at the "
            "end. That is what makes them a fair estimate of how it behaves on new documents "
            f"({self.result.metadata.get('generated_at', '')}).",
            "",
            f"**Score: {self.result.aggregate:.3f} out of 1.0.** The true value is very likely between "
            f"{aggregate_low:.3f} and {aggregate_high:.3f}. While it was being built it scored "
            f"{self.validation_aggregate:.3f} on the documents used for tuning, {against_tuning}. "
            "A score well below the tuning number means the program learned quirks of those documents "
            "rather than the task.",
            "",
            f"Reading: **{self.reading}**. What to do: **{self.action}**.",
            "",
            "## Question by question",
            "",
            "**Score** is this question's headline number, from 0 to 1. **Range** is where its true value "
            "very likely sits: a wide range means too few examples to be sure, not a worse program. "
            "**While tuning** is what the same question scored on the documents used to build it.",
            "",
            "| question | score | range | while tuning | difference | examples | what this many can tell you |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for gap in self.gaps:
            metrics = self.result.tasks[gap.task_id]
            lines.append(
                f"| {gap.task_id} | {gap.holdout:.3f} | "
                f"{gap.holdout_ci[0]:.3f}-{gap.holdout_ci[1]:.3f} | "
                f"{gap.validation:.3f} | {gap.gap:+.3f} | {gap.support} | "
                f"{'enough to measure' if gap.measurable else readable_at(gap.support)} |"
            )
        lines.append("")
        lines += [
            "## How often it is right, and how much it finds",
            "",
            "**Right when it answers** is how often an answer it gave was correct. **Found** is how much "
            "of what was there it picked up. A program can be right whenever it answers while missing most "
            "of the cases, so both matter. **Combined** balances the two.",
            "",
            "**Counts** says what each row is about. A yes/no question is shown on its **yes** answers, "
            "which is what its headline score measures. Counting both answers together flatters such a "
            "question: a program that finds 6 of 14 liability caps, and says no correctly the rest of the "
            "time, reads as 0.750 both ways when counted together, and 1.000 right / 0.429 found on the "
            "answer anyone is asking about.",
            "",
            "| question | counts | right when it answers | range | found | range | combined |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for task_id, metrics in self.result.tasks.items():
            for scored_as, precision, precision_ci, recall, recall_ci, f1 in _reported_rows(metrics):
                lines.append(
                    f"| {task_id} | {scored_as} | {precision:.3f} | "
                    f"{precision_ci[0]:.3f}-{precision_ci[1]:.3f} | "
                    f"{recall:.3f} | {recall_ci[0]:.3f}-{recall_ci[1]:.3f} | "
                    f"{f1:.3f} |"
                )
        lines.append("")
        if self.large_gap_tasks:
            lines += [
                "## Questions that did much worse than while tuning",
                "",
                "These scored far lower here than on the documents used to build the program, which means "
                "what they learned was specific to those documents rather than to the task: "
                + ", ".join(f"`{task_id}`" for task_id in self.large_gap_tasks)
                + ". Quote the number above, not the tuning one.",
                "",
            ]
        if self.unmeasurable:
            lines += [
                "## What these documents cannot tell you",
                "",
                "Each of these questions has an answer that appears too rarely here for its number to mean",
                "much. The number is still shown, but do not quote it on its own.",
                "",
            ]
            for task_id in self.unmeasurable:
                metrics = self.result.tasks[task_id]
                note = metrics.notes[0] if metrics.notes else "support is too low"
                lines.append(f"- **{task_id}**: {note}")
            lines.append("")
        if self.lock.was_overridden:
            lines += [
                "## These documents were used more than once",
                "",
                "They were meant to be scored once. Each further use makes every number above a weaker",
                "estimate of how the program behaves on documents it has never seen.",
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
            **config.models.describe(),
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
        "What was built, how well it works, and what still needs a person. Every number here comes from "
        "the files linked at the end, which hold the same numbers in full detail.",
        "",
        f"Written by doc-ai-kit {__version__} on {datetime.now(UTC).isoformat(timespec='seconds')}.",
        "",
    ]
    blocks = list(sections) + [artifact_links(project_dir)]
    # regenerated every time, so anything written into REPORT.md by hand is lost; notes kept in
    # their own file survive, which is why close carries them in rather than asking for trust
    notes = project_dir / NOTES_FILE
    if notes.exists() and notes.read_text(encoding="utf-8").strip():
        blocks.append(f"## Notes\n\n{notes.read_text(encoding='utf-8').strip()}")
    path.write_text("\n".join(header) + "\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)
    return path


# what close links to, in reading order: the path, and why a reader would open it
ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("PROMPT.md", "the prompt that shipped: instructions, questions and demonstrations"),
    ("runs/holdout/metrics.json", "the holdout numbers this report quotes, per task and per class"),
    ("runs/holdout/failures.md", "every holdout error, sampled per task"),
    ("runs/baseline_report.md", "the three baselines the champion had to beat"),
    ("runs/leaderboard.md", "one row per experiment: what changed, and what it scored"),
    ("runs/champion.json", "which experiment shipped, and the program file it points at"),
    ("runs/production/qa_report.md", "the production gates, and what was routed to review"),
    ("runs/production/outputs.jsonl", "the answers for every document in the corpus"),
    ("runs/spend.json", "what each paid step cost"),
    ("decisions.md", "the human decisions behind these numbers, as they were made"),
    ("data/annotation_rules.md", "the labeling rules the gold labels follow"),
    ("data/labels.jsonl", "the gold labels"),
    ("data/splits.json", "the train/validation/holdout assignment, and its seed"),
    ("data/extraction_manifest.csv", "per document: pages, extractor, transcription, truncation"),
)

NOTES_FILE = "REPORT_NOTES.md"


def _reported_rows(
    metrics: Any,
) -> list[tuple[str, float, tuple[float, float], float, tuple[float, float], float]]:
    """Return the precision/recall rows to print for one task.

    A yes/no task is reported on its positive answer, because that is what its headline F1
    measures and what the question is asking about. Everything else is reported over all of
    its answers, as the pooled counts.
    """
    if str(metrics.task_type) == "binary":
        positive = metrics.classes.get("true")
        if positive is not None:
            precision_ci = wilson_interval(positive.tp, positive.tp + positive.fp)
            return [("yes", positive.precision, precision_ci, positive.recall, positive.recall_ci, positive.f1)]
    return [
        (
            "all answers",
            metrics.precision,
            metrics.precision_ci,
            metrics.recall,
            metrics.recall_ci,
            metrics.f1,
        )
    ]


def artifact_links(project_dir: Path) -> str:
    """List the files this project wrote, so the numbers have something to click.

    :param project_dir: The project root
    :returns: A markdown section naming every artifact that exists
    """
    lines = ["## Where everything is", ""]
    for relative, description in ARTIFACTS:
        if (project_dir / relative).exists():
            lines.append(f"- [`{relative}`]({relative}) -- {description}")
    missing = [relative for relative, _ in ARTIFACTS if not (project_dir / relative).exists()]
    if missing:
        lines += ["", f"Not written by this project: {', '.join(f'`{name}`' for name in missing)}."]
    return "\n".join(lines)


def write_prompt(project_dir: Path, registry: Registry, program: Any, demo_chars: int = 600) -> Path:
    """Write PROMPT.md: the shipped prompt as a person can read it.

    The only other copy is a JSON string inside the compiled program, which is the artifact
    people ask for most and the hardest one to find.

    :param project_dir: The project root
    :param registry: The parsed tasks.yaml
    :param program: The compiled champion
    :param demo_chars: How much of each demonstration's document to quote
    :returns: The path written
    """
    from doc_ai_kit.program import attribute_for

    lines = [
        "# The prompt that ships",
        "",
        f"Extracted from the compiled program by doc-ai-kit {__version__} on "
        f"{datetime.now(UTC).isoformat(timespec='seconds')}. Generated: edit `tasks.yaml` and re-compile, "
        "never this file.",
        "",
    ]
    for group, tasks in registry.groups().items():
        predictor = getattr(program, attribute_for(group), None)
        signature = getattr(predictor, "signature", None) or getattr(
            getattr(predictor, "predict", None), "signature", None
        )
        lines += [f"## Predictor `{group}`", "", "### Instructions", ""]
        instructions = getattr(signature, "instructions", "") if signature is not None else ""
        lines += ["```", (instructions or "(none)").strip(), "```", "", "### Questions asked", ""]
        for task in tasks:
            described = getattr(signature, "output_fields", {}).get(task.id) if signature is not None else None
            question = getattr(described, "description", None) or task.question
            allowed = ", ".join(task.enum_members) if task.enum_members else ""
            lines.append(f"**{task.id}** ({task.type}{', one of: ' + allowed if allowed else ''})")
            lines += ["", "```", str(question).strip(), "```", ""]
        demos = list(getattr(predictor, "demos", []) or [])
        lines += [f"### Demonstrations ({len(demos)})", ""]
        for index, demo in enumerate(demos, start=1):
            values = demo.toDict() if hasattr(demo, "toDict") else dict(demo)
            document = str(values.get("document", ""))
            shortened = document[:demo_chars] + (" [... truncated ...]" if len(document) > demo_chars else "")
            lines += [
                f"#### Demonstration {index}",
                "",
                "Document:",
                "",
                "```",
                shortened.strip(),
                "```",
                "",
                "Answers:",
                "",
            ]
            for task in tasks:
                if task.id in values:
                    lines.append(f"- `{task.id}`: {values[task.id]!r}")
            lines.append("")
    path = project_dir / "PROMPT.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    logger.info("wrote %s", path)
    return path


def holdout_aggregate(runs_dir: Path) -> float | None:
    """Return the champion's holdout aggregate, for the ledger."""
    path = runs_dir / "holdout" / "metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["aggregate"]["score"])


def best_baseline(runs_dir: Path) -> tuple[str, float] | None:
    """Return the best baseline and its validation aggregate, for the ledger's notes.

    Validation, not holdout: the baselines are never run against the holdout, so there is no
    holdout number for them to report, and inventing one would be worse than leaving it blank.
    """
    scored: list[tuple[str, float]] = []
    for path in sorted(runs_dir.glob("baseline_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scored.append((path.parent.name, float(payload["aggregate"]["score"])))
    return max(scored, key=lambda item: item[1]) if scored else None


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
