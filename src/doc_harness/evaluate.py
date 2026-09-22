"""Corpus-level scoring, and the two files every scoring run writes.

Per-example correctness comes from :mod:`doc_harness.metric`; this module accumulates the
outcome counts into precision, recall and F1 over the whole split, and writes:

* ``metrics.json`` -- every number, plus the metadata needed to reproduce the run;
* ``failures.md`` -- confusion matrices, per-class error counts, and a small sample of
  errors.

The error sample is capped on purpose. An optimizer handed the full error dump writes rules
keyed to individual documents, and those rules do not survive the holdout.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from doc_harness import __version__
from doc_harness.guards import GuardError
from doc_harness.metric import ExampleScore, Metric, _context_window, get_field
from doc_harness.program import fresh_generation
from doc_harness.registry import Registry, TaskType, _BaseTask
from doc_harness.stats import bootstrap_interval, precision_recall_f1, readable_at, wilson_interval

logger = logging.getLogger(__name__)

# at most this many sampled errors per task reach failures.md
MAX_SAMPLED_ERRORS = 3
# classes with fewer than this many gold examples are excluded from the optimization target
DEFAULT_SUPPORT_FLOOR = 30
# below this, a class cannot carry a reported number at all
DEFAULT_MEASURABLE_FLOOR = 10
# how much document text to quote beside a sampled error
ERROR_CONTEXT_CHARS = 300

NULL_LABEL = "(null)"


@dataclass(frozen=True)
class ClassMetrics:
    """One class of one task, scored one-vs-rest."""

    label: str
    tp: int
    fp: int
    fn: int
    support: int
    precision: float
    recall: float
    f1: float
    recall_ci: tuple[float, float]
    measurable: bool
    above_floor: bool
    readable_as: str

    def to_dict(self) -> dict[str, Any]:
        """Return this class's numbers as plain JSON types."""
        return {
            "label": self.label,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "recall_ci95": list(self.recall_ci),
            "measurable": self.measurable,
            "above_support_floor": self.above_floor,
            "readable_as": self.readable_as,
        }


@dataclass(frozen=True)
class TaskMetrics:
    """Corpus-level numbers for one task."""

    task_id: str
    task_type: str
    primary_metric: str
    primary_value: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    support: int
    n_examples: int
    exact_match: float
    # how often gold says nothing, and how often the program said nothing. The second is the
    # reference the production null-rate gate compares against.
    gold_null_rate: float
    predicted_null_rate: float
    precision_ci: tuple[float, float]
    recall_ci: tuple[float, float]
    classes: dict[str, ClassMetrics] = field(default_factory=dict)
    confusion: list[tuple[str, str, int]] = field(default_factory=list)
    # for fuzzy tasks, the same numbers under strict equality, never reported alone
    strict: dict[str, Any] | None = None
    weighted: dict[str, Any] | None = None
    measurable: bool = True
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return this task's numbers as plain JSON types."""
        payload: dict[str, Any] = {
            "type": self.task_type,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "counts": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "n_examples": self.n_examples,
            "exact_match": self.exact_match,
            "gold_null_rate": self.gold_null_rate,
            "predicted_null_rate": self.predicted_null_rate,
            "precision_ci95": list(self.precision_ci),
            "recall_ci95": list(self.recall_ci),
            "measurable": self.measurable,
            "notes": self.notes,
        }
        if self.classes:
            payload["classes"] = {label: metrics.to_dict() for label, metrics in self.classes.items()}
        if self.confusion:
            payload["confusion"] = [[gold, pred, count] for gold, pred, count in self.confusion]
        if self.strict is not None:
            payload["strict"] = self.strict
        if self.weighted is not None:
            payload["weighted"] = self.weighted
        return payload


@dataclass(frozen=True)
class EvaluationResult:
    """Everything one scoring run produced."""

    aggregate: float
    aggregate_ci: tuple[float, float]
    tasks: dict[str, TaskMetrics]
    scores: list[ExampleScore]
    metadata: dict[str, Any]
    excluded_classes: dict[str, list[str]] = field(default_factory=dict)
    # documents whose reply could not be read; each is scored as wrong on every task
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def per_task_primary(self) -> dict[str, float]:
        """Return the per-task primary metric vector that sits beside the aggregate.

        Always reported with the scalar: a rising aggregate hiding a collapsing task is a
        common and expensive failure.
        """
        return {task_id: metrics.primary_value for task_id, metrics in self.tasks.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return the full metrics.json payload."""
        return {
            "metadata": self.metadata,
            "aggregate": {
                "score": self.aggregate,
                "ci95": list(self.aggregate_ci),
                "per_task_primary": self.per_task_primary,
            },
            "tasks": {task_id: metrics.to_dict() for task_id, metrics in self.tasks.items()},
            "excluded_classes": self.excluded_classes,
            "unmeasurable_tasks": [task_id for task_id, m in self.tasks.items() if not m.measurable],
            "failures": self.failures,
        }


def is_abstention(value: Any) -> bool:
    """Return whether a normalized value means "no answer".

    An empty set or list counts: a multilabel task abstains by naming nothing.
    """
    if value is None:
        return True
    if isinstance(value, set | frozenset | list | tuple):
        return len(value) == 0
    return False


def _label(value: Any) -> str:
    """Render a normalized value as a confusion-matrix label."""
    if value is None:
        return NULL_LABEL
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _label_set(value: object | None) -> set[str]:
    """Render a set-valued answer as a set of labels."""
    if value is None:
        return set()
    if isinstance(value, set | frozenset | list | tuple):
        return {_label(item) for item in value}
    return {_label(value)}


def _declared_labels(task: _BaseTask) -> list[str]:
    """Return the label space to report for a classification task."""
    if TaskType(task.type) is TaskType.BINARY:
        return ["true", "false"]
    return list(task.enum_members or [])


def _class_counts(task: _BaseTask, scores: Sequence[ExampleScore]) -> dict[str, dict[str, int]]:
    """Accumulate one-vs-rest counts per class for a classification task."""
    task_type = TaskType(task.type)
    counts: dict[str, dict[str, int]] = {}

    def bucket(label: str) -> dict[str, int]:
        return counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0})

    for label in _declared_labels(task):
        bucket(label)
    for score in scores:
        result = score.results[task.id]
        if task_type is TaskType.MULTILABEL:
            gold_set = _label_set(result.gold_normalized)
            pred_set = _label_set(result.pred_normalized)
        else:
            gold_set = {_label(result.gold_normalized)} - {NULL_LABEL}
            pred_set = {_label(result.pred_normalized)} - {NULL_LABEL}
        for label in gold_set & pred_set:
            bucket(label)["tp"] += 1
        for label in pred_set - gold_set:
            bucket(label)["fp"] += 1
        for label in gold_set - pred_set:
            bucket(label)["fn"] += 1
    return counts


def _build_classes(
    task: _BaseTask,
    scores: Sequence[ExampleScore],
    support_floor: int,
    measurable_floor: int,
) -> dict[str, ClassMetrics]:
    """Turn per-class counts into per-class metrics with intervals and readability."""
    classes: dict[str, ClassMetrics] = {}
    for label, counts in _class_counts(task, scores).items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        support = tp + fn
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        classes[label] = ClassMetrics(
            label=label,
            tp=tp,
            fp=fp,
            fn=fn,
            support=support,
            precision=precision,
            recall=recall,
            f1=f1,
            recall_ci=wilson_interval(tp, support) if support else (0.0, 1.0),
            measurable=support >= measurable_floor,
            above_floor=support >= support_floor,
            readable_as=readable_at(support),
        )
    return classes


def _confusion(task: _BaseTask, scores: Sequence[ExampleScore]) -> list[tuple[str, str, int]]:
    """Build a gold-versus-predicted table for a single-label classification task."""
    tally: dict[tuple[str, str], int] = {}
    for score in scores:
        result = score.results[task.id]
        key = (_label(result.gold_normalized), _label(result.pred_normalized))
        tally[key] = tally.get(key, 0) + 1
    return [(gold, pred, count) for (gold, pred), count in sorted(tally.items())]


def _abstention_rate(scores: Sequence[ExampleScore], task_id: str, side: str) -> float:
    """Return how often one side of the comparison declined to answer."""
    if not scores:
        return 0.0
    silent = sum(1 for score in scores if is_abstention(getattr(score.results[task_id], side)))
    return silent / len(scores)


def _counts_from(
    scores: Sequence[ExampleScore], task_id: str, alternate: str | None = None
) -> tuple[int, int, int, int]:
    """Sum outcome counts for one task, optionally from a side-by-side alternate result."""
    tp = fp = fn = tn = 0
    for score in scores:
        result = score.results[task_id]
        if alternate is not None:
            result = result.alternates.get(alternate, result)
        tp += result.tp
        fp += result.fp
        fn += result.fn
        tn += result.tn
    return tp, fp, fn, tn


def _weighted_counts(scores: Sequence[ExampleScore], task_id: str, weights: Sequence[float]) -> dict[str, Any]:
    """Sum outcome counts under inverse-probability weights, for corpus-level estimates."""
    tp = fp = fn = tn = 0.0
    for score, weight in zip(scores, weights, strict=True):
        result = score.results[task_id]
        tp += weight * result.tp
        fp += weight * result.fp
        fn += weight * result.fn
        tn += weight * result.tn
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "basis": "inverse-probability weighted; corpus-level estimate",
    }


def _primary_value(
    task: _BaseTask,
    f1: float,
    classes: Mapping[str, ClassMetrics],
) -> tuple[float, list[str]]:
    """Return the task's headline number and any caveat that belongs beside it."""
    task_type = TaskType(task.type)
    notes: list[str] = []
    if task_type is TaskType.BINARY:
        positive = classes.get("true")
        return (positive.f1 if positive else 0.0), notes
    if task_type in {TaskType.MULTICLASS, TaskType.MULTILABEL}:
        above = [metrics for metrics in classes.values() if metrics.above_floor]
        if above:
            return sum(metrics.f1 for metrics in above) / len(above), notes
        # a class that is in neither the gold labels nor the predictions was not measured at
        # all, and averaging its F1 of zero reports a failure that never happened: a task
        # declaring 51 states and answering all 29 documents correctly scored 0.173
        seen = [metrics for metrics in classes.values() if metrics.tp + metrics.fp + metrics.fn > 0]
        # macro-F1 weights every class equally, so a four-example class otherwise swings the
        # headline number as hard as a four-hundred-example one
        if not seen:
            notes.append("no class appears in the gold labels or the predictions")
            return 0.0, notes
        notes.append(
            f"no class reaches the support floor; macro-F1 over the {len(seen)} class(es) that appear"
            + (f", of {len(classes)} declared" if len(classes) > len(seen) else "")
        )
        return sum(metrics.f1 for metrics in seen) / len(seen), notes
    return f1, notes


def score_split(
    registry: Registry,
    metric: Metric,
    golds: Sequence[Any],
    preds: Sequence[Any],
    metadata: Mapping[str, Any] | None = None,
    weights: Sequence[float] | None = None,
    support_floor: int = DEFAULT_SUPPORT_FLOOR,
    measurable_floor: int = DEFAULT_MEASURABLE_FLOOR,
    seed: int = 0,
) -> EvaluationResult:
    """Score a split and build every number that goes into metrics.json.

    :param registry: The parsed tasks.yaml
    :param metric: The metric built from that registry
    :param golds: Labeled examples
    :param preds: Predictions in the same order
    :param metadata: Run metadata merged into the report, such as model strings
    :param weights: Optional inverse-probability weights for corpus-level estimates
    :param support_floor: Classes below this are excluded from the optimization target
    :param measurable_floor: Classes below this cannot carry a reported number
    :param seed: Fixed so the sampled errors and bootstrap are reproducible
    :returns: The full evaluation result
    """
    if len(golds) != len(preds):
        raise ValueError(f"cannot score {len(preds)} predictions against {len(golds)} gold examples")
    if weights is not None and len(weights) != len(golds):
        raise ValueError(f"got {len(weights)} weights for {len(golds)} examples")
    scores = [metric.score_example(gold, pred) for gold, pred in zip(golds, preds, strict=True)]

    tasks: dict[str, TaskMetrics] = {}
    for task in registry:
        task_type = TaskType(task.type)
        tp, fp, fn, tn = _counts_from(scores, task.id)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        classes = (
            _build_classes(task, scores, support_floor, measurable_floor)
            if task_type in {TaskType.BINARY, TaskType.MULTICLASS, TaskType.MULTILABEL}
            else {}
        )
        primary_value, notes = _primary_value(task, f1, classes)
        strict: dict[str, Any] | None = None
        if task_type is TaskType.EXTRACT_FUZZY:
            s_tp, s_fp, s_fn, s_tn = _counts_from(scores, task.id, alternate="strict")
            s_precision, s_recall, s_f1 = precision_recall_f1(s_tp, s_fp, s_fn)
            # always beside the fuzzy number, so nobody reads a threshold as an accuracy
            strict = {
                "counts": {"tp": s_tp, "fp": s_fp, "fn": s_fn, "tn": s_tn},
                "precision": s_precision,
                "recall": s_recall,
                "f1": s_f1,
                "basis": "strict equality after normalization",
            }
        rarest = min((metrics.support for metrics in classes.values()), default=tp + fn)
        measurable = rarest >= measurable_floor
        if not measurable:
            notes.append(
                f"rarest class has {rarest} example(s): {readable_at(rarest)}; "
                "this task's number should not be quoted on its own"
            )
        tasks[task.id] = TaskMetrics(
            task_id=task.id,
            task_type=str(task.type),
            primary_metric=str(task.primary_metric),
            primary_value=primary_value,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=precision,
            recall=recall,
            f1=f1,
            support=tp + fn,
            n_examples=len(scores),
            exact_match=sum(1 for s in scores if s.results[task.id].correct) / len(scores) if scores else 0.0,
            gold_null_rate=_abstention_rate(scores, task.id, "gold_normalized"),
            predicted_null_rate=_abstention_rate(scores, task.id, "pred_normalized"),
            precision_ci=wilson_interval(tp, tp + fp) if (tp + fp) else (0.0, 1.0),
            recall_ci=wilson_interval(tp, tp + fn) if (tp + fn) else (0.0, 1.0),
            classes=classes,
            confusion=(_confusion(task, scores) if task_type in {TaskType.BINARY, TaskType.MULTICLASS} else []),
            strict=strict,
            weighted=_weighted_counts(scores, task.id, weights) if weights is not None else None,
            measurable=measurable,
            notes=notes,
        )

    failures = [
        {"doc_id": get_field(gold, "doc_id"), "error": pred.error, "attempts": pred.attempts}
        for gold, pred in zip(golds, preds, strict=True)
        if isinstance(pred, FailedReply)
    ]
    aggregates = [score.aggregate for score in scores]
    aggregate = sum(aggregates) / len(aggregates) if aggregates else 0.0
    merged_metadata = {
        "harness_version": __version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_examples": len(scores),
        "tasks_source": str(registry.source) if registry.source else None,
        "support_floor": support_floor,
        "measurable_floor": measurable_floor,
        "seed": seed,
        "n_failed_replies": len(failures),
        **dict(metadata or {}),
    }
    # counts must reconcile: every example scored every task, or something was dropped
    expected = len(scores) * len(registry)
    actual = sum(len(score.results) for score in scores)
    if expected != actual:
        raise RuntimeError(f"scored {actual} task results, expected {expected}: predictions were dropped")

    return EvaluationResult(
        aggregate=aggregate,
        aggregate_ci=bootstrap_interval(aggregates, seed=seed),
        tasks=tasks,
        scores=scores,
        metadata=merged_metadata,
        excluded_classes={task_id: sorted(labels) for task_id, labels in metric.excluded_classes.items()},
        failures=failures,
    )


def _sample_errors(
    task_id: str,
    golds: Sequence[Any],
    scores: Sequence[ExampleScore],
    seed: int,
    limit: int = MAX_SAMPLED_ERRORS,
) -> list[tuple[Any, ExampleScore]]:
    """Pick a small, reproducible sample of this task's errors."""
    failing = [(gold, score) for gold, score in zip(golds, scores, strict=True) if not score.results[task_id].correct]
    if len(failing) <= limit:
        return failing
    rng = random.Random(f"{seed}:{task_id}")
    return rng.sample(failing, limit)


def _confusion_table(confusion: Sequence[tuple[str, str, int]]) -> list[str]:
    """Render a confusion matrix as a markdown table."""
    if not confusion:
        return []
    golds = sorted({gold for gold, _pred, _count in confusion})
    preds = sorted({pred for _gold, pred, _count in confusion})
    lookup = {(gold, pred): count for gold, pred, count in confusion}
    lines = ["| gold \\ predicted | " + " | ".join(preds) + " |", "| --- |" + " --- |" * len(preds)]
    for gold in golds:
        cells = [str(lookup.get((gold, pred), 0)) for pred in preds]
        lines.append(f"| {gold} | " + " | ".join(cells) + " |")
    return lines


def write_failures(
    path: Path,
    registry: Registry,
    result: EvaluationResult,
    golds: Sequence[Any],
) -> None:
    """Write failures.md: confusion matrices, error counts and a capped error sample.

    :param path: Where to write the file
    :param registry: The parsed tasks.yaml
    :param result: The scored split
    :param golds: The labeled examples, for quoting document context
    """
    seed = int(result.metadata.get("seed", 0))
    lines: list[str] = [
        "# Failure analysis",
        "",
        f"Split of {result.metadata['n_examples']} examples, "
        f"aggregate {result.aggregate:.3f} "
        f"(95% CI {result.aggregate_ci[0]:.3f}-{result.aggregate_ci[1]:.3f}).",
        "",
        f"At most {MAX_SAMPLED_ERRORS} errors are sampled per task. That cap is deliberate: rules written",
        "against a full error dump key themselves to individual documents and do not survive the holdout.",
        "",
    ]
    if result.failures:
        lines += [
            f"## Failed replies ({len(result.failures)})",
            "",
            "These replies could not be read even after fresh retries. Each is scored as a wrong answer",
            "on every task, not as an abstention. They are not the model's judgement and say nothing",
            "about the prompt -- fix the cause, usually models.max_tokens, before reading anything below.",
            "",
            "| doc | attempts | error |",
            "| --- | --- | --- |",
        ]
        for failure in result.failures:
            error = " ".join(str(failure["error"]).split())[:200].replace("|", "\\|")
            lines.append(f"| {failure['doc_id']} | {failure['attempts']} | {error} |")
        lines.append("")
    for task in registry:
        metrics = result.tasks[task.id]
        errors = sum(1 for score in result.scores if not score.results[task.id].correct)
        lines += [
            f"## {task.id} ({metrics.task_type})",
            "",
            f"{metrics.primary_metric} **{metrics.primary_value:.3f}** | "
            f"P {metrics.precision:.3f} R {metrics.recall:.3f} F1 {metrics.f1:.3f} | "
            f"support {metrics.support} | {errors} of {metrics.n_examples} examples wrong",
            "",
        ]
        for note in metrics.notes:
            lines += [f"> {note}", ""]
        if metrics.strict is not None:
            lines += [
                f"Strict equality for the same task: P {metrics.strict['precision']:.3f} "
                f"R {metrics.strict['recall']:.3f} F1 {metrics.strict['f1']:.3f}.",
                "",
            ]
        if metrics.confusion:
            lines += ["### Confusion", ""] + _confusion_table(metrics.confusion) + [""]
        if metrics.classes:
            lines += [
                "### Per class",
                "",
                "| class | support | P | R | F1 | errors | reads as |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for label, class_metrics in sorted(metrics.classes.items()):
                lines.append(
                    f"| {label} | {class_metrics.support} | {class_metrics.precision:.3f} | "
                    f"{class_metrics.recall:.3f} | {class_metrics.f1:.3f} | "
                    f"{class_metrics.fp + class_metrics.fn} | {class_metrics.readable_as} |"
                )
            lines.append("")
        sampled = _sample_errors(task.id, golds, result.scores, seed)
        if sampled:
            lines += [f"### Sampled errors ({len(sampled)} of {errors})", ""]
            for gold, score in sampled:
                match_result = score.results[task.id]
                doc_id = get_field(gold, "doc_id") or "(unknown doc)"
                lines.append(f"- **{doc_id}**: {match_result.detail}")
                window = _context_window(get_field(gold, "document"), match_result.gold_normalized)
                if window:
                    lines.append(f"  - document: ...{window[:ERROR_CONTEXT_CHARS]}...")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", path)


def write_metrics(path: Path, result: EvaluationResult) -> None:
    """Write metrics.json."""
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def write_predictions(path: Path, registry: Registry, golds: Sequence[Any], preds: Sequence[Any]) -> None:
    """Write the raw predictions a run produced, so it can be re-scored for free.

    Matcher and normalizer bugs are found after the fact -- that is what the adversarial
    fixtures are for, and they still do not catch everything. Keeping the predictions means
    fixing one and re-measuring costs nothing, instead of paying for the whole split again
    and quietly tempting everyone to leave the bug in.
    """
    rows = []
    for gold, pred in zip(golds, preds, strict=True):
        row: dict[str, Any] = {
            "doc_id": get_field(gold, "doc_id"),
            "predicted": {task.id: to_json_value(get_field(pred, task.id)) for task in registry},
        }
        if isinstance(pred, FailedReply):
            row["failed"] = {"error": pred.error, "attempts": pred.attempts}
        rows.append(row)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def to_json_value(value: Any) -> Any:
    """Render a value as structured JSON, for every file the harness writes.

    Typed outputs are pydantic models. Falling back to ``str`` would write
    ``\"value=375000.0 unit='USD'\"``, which re-scoring would then have to parse back out of a
    repr -- so the saved predictions would no longer reproduce the run they came from.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [to_json_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(to_json_value(item) for item in value)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Read a run's saved predictions, keyed by document id."""
    if not path.exists():
        raise FileNotFoundError(f"no saved predictions at {path}; that run cannot be re-scored without inference")
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "failed" in row:
            failure = row["failed"]
            rows[str(row["doc_id"])] = FailedReply(list(row["predicted"]), failure["error"], failure["attempts"])
        else:
            rows[str(row["doc_id"])] = dict(row["predicted"])
    return rows


def write_run(
    run_dir: Path,
    registry: Registry,
    result: EvaluationResult,
    golds: Sequence[Any],
    preds: Sequence[Any] | None = None,
) -> None:
    """Write the output files for one scoring run.

    :param run_dir: The run directory, created if missing
    :param registry: The parsed tasks.yaml
    :param result: The scored split
    :param golds: The labeled examples
    :param preds: The predictions, saved so the run can be re-scored without inference
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(run_dir / "metrics.json", result)
    write_failures(run_dir / "failures.md", registry, result, golds)
    if preds is not None:
        write_predictions(run_dir / "predictions.jsonl", registry, golds, preds)


# what a reply that could not be read is scored as, on every task. Chosen so that no
# normalizer reads it as null and no matcher can match it: it scores as a wrong answer.
FAILED_REPLY = "<failed reply>"


class EvaluationError(GuardError):
    """Raised when too many replies failed for a scoring run's numbers to mean anything."""


class FailedReply(dict[str, Any]):
    """A document whose reply could not be read, even after fresh retries.

    Every task maps to :data:`FAILED_REPLY`, so the ordinary matchers score it as a wrong
    answer -- including where the gold is null, which an abstention would have got right.
    The error travels with it into metrics.json and failures.md.
    """

    def __init__(self, task_ids: Sequence[str], error: str, attempts: int) -> None:
        super().__init__({task_id: FAILED_REPLY for task_id in task_ids})
        self.error = error
        self.attempts = attempts


def _is_failed(prediction: Any) -> bool:
    """Return whether dspy.Evaluate handed back its empty placeholder for a raised error."""
    keys = prediction.keys() if hasattr(prediction, "keys") else None
    return keys is not None and len(list(keys)) == 0


def _retry(program: Any, example: Any, task_ids: Sequence[str], max_retries: int) -> Any:
    """Run one example again, the first time from cache and then with fresh generations.

    The first replay is free -- DSPy caches the reply that failed to parse -- and recovers
    the error that dspy.Evaluate swallowed. Each later attempt asks the model afresh.
    """
    last_error = "the reply could not be read"
    for attempt in range(1, max_retries + 2):
        try:
            with fresh_generation(attempt):
                return program(**example.inputs())
        except Exception as exc:  # noqa: BLE001 - recorded on the failure, never swallowed
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s failed on attempt %d: %s", get_field(example, "doc_id"), attempt, last_error)
    return FailedReply(task_ids, last_error, attempts=max_retries + 1)


def run_program(
    program: Any,
    examples: Sequence[Any],
    metric: Metric,
    num_threads: int = 8,
    display_progress: bool = False,
    max_retries: int = 2,
    max_failure_rate: float = 0.0,
) -> list[Any]:
    """Run a program over a split and return its predictions in example order.

    Wraps ``dspy.Evaluate`` for its threading, but not its error handling: on a raised error
    it substitutes an empty prediction, which reads as a null on every task and would be
    scored as the model choosing to abstain. Here a failed reply is retried with fresh
    generations; one that never succeeds becomes a :class:`FailedReply`, scored as wrong and
    listed by document. Above ``max_failure_rate`` the run refuses to report at all.

    The scoring that matters happens in :func:`score_split` afterwards, against saved
    predictions, which is what lets a project re-score without paying for inference again.

    :param program: The DSPy program to run
    :param examples: The examples to run it over
    :param metric: The metric, passed through to dspy.Evaluate
    :param num_threads: How many examples to run concurrently
    :param display_progress: Whether to show DSPy's progress bar
    :param max_retries: Fresh attempts for a reply that could not be read
    :param max_failure_rate: Share of still-failing replies above which the run refuses
    :returns: One prediction per example, in the same order
    """
    import dspy

    evaluator = dspy.Evaluate(
        devset=list(examples),
        metric=metric,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=False,
        provide_traceback=True,
        # every failure is handed back to be retried and accounted for, never an abort midway
        max_errors=len(examples) + 1,
    )
    outcome = evaluator(program)
    by_doc: dict[str, Any] = {}
    predictions: list[Any] = []
    for example, prediction, _score in outcome.results:
        doc_id = get_field(example, "doc_id")
        if doc_id is not None:
            by_doc[str(doc_id)] = prediction
        predictions.append(prediction)
    if len(predictions) != len(examples):
        raise RuntimeError(f"program returned {len(predictions)} predictions for {len(examples)} examples")
    if len(by_doc) == len(examples):
        # reorder by doc_id, because a threaded run does not guarantee input order
        predictions = [by_doc[str(get_field(example, "doc_id"))] for example in examples]

    task_ids = metric.registry.ids
    for index, (example, prediction) in enumerate(zip(examples, predictions, strict=True)):
        if _is_failed(prediction):
            predictions[index] = _retry(program, example, task_ids, max_retries)

    failed = [
        (get_field(example, "doc_id"), prediction)
        for example, prediction in zip(examples, predictions, strict=True)
        if isinstance(prediction, FailedReply)
    ]
    if failed:
        rate = len(failed) / len(examples)
        listing = "; ".join(f"{doc_id}: {reply.error.splitlines()[0][:160]}" for doc_id, reply in failed)
        if rate > max_failure_rate:
            raise EvaluationError(
                f"{len(failed)} of {len(examples)} replies ({rate:.0%}) could not be read after "
                f"{max_retries} fresh retries, above evaluation.max_failure_rate of {max_failure_rate:.0%}. "
                "Numbers from this run would describe the failures as much as the program, so none are "
                f"reported. Usually this is truncation: check models.max_tokens. Failed: {listing}"
            )
        logger.warning("%d reply(ies) failed and are scored as wrong answers: %s", len(failed), listing)
    return predictions


def task_primary_ci(
    registry: Registry,
    task_id: str,
    scores: Sequence[ExampleScore],
    support_floor: int = DEFAULT_SUPPORT_FLOOR,
    measurable_floor: int = DEFAULT_MEASURABLE_FLOOR,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a confidence interval for one task's primary metric.

    F1 and macro-F1 are not binomial proportions, so Wilson does not apply to them. The
    interval is built by resampling examples and recomputing the metric, which is what makes
    the validation-to-holdout gap readable rather than a bare difference of two numbers.

    :param registry: The parsed tasks.yaml
    :param task_id: The task to bound
    :param scores: The per-example scores from :func:`score_split`
    :param support_floor: Classes below this leave the optimization target
    :param measurable_floor: Classes below this cannot carry a number
    :param resamples: How many bootstrap resamples to draw
    :param seed: Fixed so the interval is reproducible
    :returns: The 2.5th and 97.5th percentile of the resampled metric
    """
    import numpy as np

    task = registry.by_id(task_id)
    if not scores:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    size = len(scores)
    draws: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, size, size=size)
        sample = [scores[int(index)] for index in indices]
        tp, fp, fn, _tn = _counts_from(sample, task_id)
        _precision, _recall, f1 = precision_recall_f1(tp, fp, fn)
        classes = (
            _build_classes(task, sample, support_floor, measurable_floor)
            if TaskType(task.type) in {TaskType.BINARY, TaskType.MULTICLASS, TaskType.MULTILABEL}
            else {}
        )
        value, _notes = _primary_value(task, f1, classes)
        draws.append(value)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def load_metrics(path: Path) -> dict[str, Any]:
    """Read a stored metrics.json.

    Reading a recorded run is always preferable to re-running it: it costs nothing, and it
    is the number that was actually recorded rather than a fresh sample of a stochastic
    program that happens to be called the same thing.
    """
    if not path.exists():
        raise FileNotFoundError(f"no metrics at {path}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def per_task_from_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Return the per-task primary metric vector from a stored metrics.json."""
    return {task_id: float(value) for task_id, value in payload["aggregate"]["per_task_primary"].items()}


def null_rates_from_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Return how often the program abstained per task, from a stored metrics.json.

    The reference for the production null-rate gate: a task the program answered on nine
    validation documents in ten, and declines on two thirds of the corpus, has stopped
    answering, and that is worth knowing before anyone quotes the outputs.

    Read from the recorded rate rather than derived from support, because support counts
    labels for a multilabel task and characters for a span task, not documents.
    """
    return {
        task_id: float(task["predicted_null_rate"])
        for task_id, task in payload["tasks"].items()
        if "predicted_null_rate" in task
    }
