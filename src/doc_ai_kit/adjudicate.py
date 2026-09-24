"""Borderline decisions, laid out for a human to adjudicate.

A threshold does its work silently. When a fuzzy match accepts a name at 0.91 against a theta
of 0.90, nothing records that it was close, and if the normalizer that fed it was wrong, no
number anywhere looks odd -- the accuracy is simply wrong. This module finds every decision a
threshold actually made and lays it out with the raw and normalized values side by side,
which is where a normalizer bug becomes visible.

This is a human review artifact, and it is uncapped on purpose. It is not failures.md: that
file samples three errors per task because an optimizer shown every error writes rules keyed
to individual documents. Do not feed this report to an optimizer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc_ai_kit.match import _render
from doc_ai_kit.metric import Metric, get_field
from doc_ai_kit.registry import Registry, TaskType
from doc_ai_kit.values import MatchResult, PartialDate

logger = logging.getLogger(__name__)

# how close a similarity or overlap has to sit to its threshold to count as a close call
BORDERLINE_MARGIN = 0.05
# for numeric matches, the band of distance-to-tolerance ratios worth a look: 1.0 is exactly
# on the boundary, and anything from half the allowance to twice it could have gone either way
NUMERIC_BAND = (0.5, 2.0)

THRESHOLDED_TYPES = frozenset(
    {
        TaskType.EXTRACT_FUZZY,
        TaskType.EXTRACT_LIST,
        TaskType.EXTRACT_NUMERIC,
        TaskType.EXTRACT_DATE,
        TaskType.SPAN,
    }
)


@dataclass(frozen=True)
class Decision:
    """One matcher decision worth a human look."""

    doc_id: str
    task_id: str
    gold_raw: Any
    pred_raw: Any
    result: MatchResult
    reason: str

    @property
    def verdict(self) -> str:
        """Return what the matcher decided, in words."""
        return "accepted" if self.result.correct else "rejected"


@dataclass
class Adjudication:
    """Every close call and every remaining error in one run."""

    run_id: str
    decisions: list[Decision]
    errors: list[Decision]
    n_examples: int

    def by_task(self, items: Sequence[Decision]) -> dict[str, list[Decision]]:
        """Group decisions by task, preserving document order."""
        grouped: dict[str, list[Decision]] = {}
        for item in items:
            grouped.setdefault(item.task_id, []).append(item)
        return grouped


def _why_borderline(task_type: TaskType, result: MatchResult) -> str | None:
    """Return why a decision is a close call, or None if the threshold was never in play."""
    strict = result.alternates.get("strict")
    if task_type is TaskType.EXTRACT_FUZZY and strict is not None and strict.correct != result.correct:
        # the one case the fuzzy number exists for: strict said one thing, theta another
        return "theta overruled strict equality"
    if task_type is TaskType.EXTRACT_LIST and result.correct:
        gold, pred = result.gold_normalized, result.pred_normalized
        # the list analogue of the rule above: accepted, yet the items are not identical, so
        # theta made the call however far from theta the score happened to land
        if isinstance(gold, list) and isinstance(pred, list) and sorted(gold) != sorted(pred):
            return "theta overruled strict equality"
    if task_type is TaskType.EXTRACT_NUMERIC and "unit mismatch" in result.detail:
        return "units differ"
    if task_type is TaskType.EXTRACT_DATE:
        gold, pred = result.gold_normalized, result.pred_normalized
        if isinstance(gold, PartialDate) and isinstance(pred, PartialDate) and gold.granularity != pred.granularity:
            return f"granularity differs: gold {gold.granularity.value}, predicted {pred.granularity.value}"
        return None
    if result.measure is None or result.threshold is None:
        return None
    if task_type is TaskType.EXTRACT_NUMERIC:
        low, high = NUMERIC_BAND
        if low <= result.measure <= high:
            return f"delta is {result.measure:.2f}x the tolerance"
        return None
    if abs(result.measure - result.threshold) <= BORDERLINE_MARGIN:
        return f"{result.measure:.3f} against a threshold of {result.threshold:.2f}"
    return None


def adjudicate(
    registry: Registry,
    metric: Metric,
    golds: Sequence[Any],
    preds: Sequence[Any],
    run_id: str = "",
) -> Adjudication:
    """Find every close call and every error in a scored run.

    :param registry: The parsed tasks.yaml
    :param metric: The metric, whose matchers make the decisions being reviewed
    :param golds: The labeled examples
    :param preds: The predictions, in the same order
    :param run_id: The run being reviewed, for the report heading
    :returns: The close calls and the remaining errors, kept separate
    """
    if len(golds) != len(preds):
        raise ValueError(f"cannot adjudicate {len(preds)} predictions against {len(golds)} gold examples")
    decisions: list[Decision] = []
    errors: list[Decision] = []
    for gold, pred in zip(golds, preds, strict=True):
        doc_id = str(get_field(gold, "doc_id") or "(unknown doc)")
        for task in registry:
            result = metric.score_task(task, gold, pred)
            reason = _why_borderline(TaskType(task.type), result) if TaskType(task.type) in THRESHOLDED_TYPES else None
            item = Decision(
                doc_id=doc_id,
                task_id=task.id,
                gold_raw=get_field(gold, task.id),
                pred_raw=get_field(pred, task.id),
                result=result,
                reason=reason or "error",
            )
            if reason is not None:
                decisions.append(item)
            elif not result.correct:
                errors.append(item)
    return Adjudication(run_id=run_id, decisions=decisions, errors=errors, n_examples=len(golds))


def _cell(value: Any) -> str:
    """Render a raw value for a markdown table cell."""
    if value is None:
        text = "null"
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, default=str, sort_keys=True)
    return " ".join(text.split()).replace("|", "\\|")


def _raw_and_normalized(raw: Any, normalized: Any) -> str:
    """Show the value as written and as the matcher saw it, when those differ.

    This is the column that exposes a normalizer bug: two raw values a person would call the
    same, normalized into two different things.
    """
    raw_text = _cell(raw)
    normalized_text = _cell(_render(normalized))
    if raw_text.casefold() == normalized_text.casefold():
        return raw_text
    return f"{raw_text} → `{normalized_text}`"


def to_markdown(report: Adjudication) -> str:
    """Render the adjudication report."""
    lines = [
        f"# Adjudication: {report.run_id or 'run'}",
        "",
        f"{report.n_examples} documents. {len(report.decisions)} decision(s) a threshold actually made, "
        f"and {len(report.errors)} other error(s).",
        "",
        "For each close call, decide whether the matcher was right. Values are shown as written",
        "and, after the arrow, as the matcher compared them -- a pair a person would call the same",
        "but that normalizes into two different strings is a normalizer bug, not a model error.",
        "",
        "Record anything that changes a threshold or a normalizer in `decisions.md`.",
        "",
        "> Uncapped, and for humans only. Unlike failures.md this lists every error; shown to an",
        "> optimizer it would teach rules keyed to individual documents.",
        "",
    ]
    lines += ["## Close calls", ""]
    if not report.decisions:
        lines += ["No threshold decided anything in this run.", ""]
    for task_id, items in report.by_task(report.decisions).items():
        accepted = sum(1 for item in items if item.result.correct)
        lines += [
            f"### {task_id} ({len(items)}: {accepted} accepted, {len(items) - accepted} rejected)",
            "",
            "| doc | gold | predicted | verdict | why it is here |",
            "| --- | --- | --- | --- | --- |",
        ]
        for item in items:
            lines.append(
                f"| {item.doc_id} | {_raw_and_normalized(item.gold_raw, item.result.gold_normalized)} | "
                f"{_raw_and_normalized(item.pred_raw, item.result.pred_normalized)} | "
                f"{item.verdict} | {item.reason} |"
            )
        lines.append("")
    lines += ["## Other errors", ""]
    if not report.errors:
        lines += ["None.", ""]
    for task_id, items in report.by_task(report.errors).items():
        lines += [f"### {task_id} ({len(items)})", "", "| doc | gold | predicted |", "| --- | --- | --- |"]
        for item in items:
            lines.append(
                f"| {item.doc_id} | {_raw_and_normalized(item.gold_raw, item.result.gold_normalized)} | "
                f"{_raw_and_normalized(item.pred_raw, item.result.pred_normalized)} |"
            )
        lines.append("")
    return "\n".join(lines)


def write_adjudication(path: Path, report: Adjudication) -> None:
    """Write adjudication.md."""
    path.write_text(to_markdown(report), encoding="utf-8")
    logger.info("wrote %s", path)
