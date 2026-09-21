"""The scoring metric, built from the task registry.

Responsibility is split deliberately:

* this module scores **one example at a time** -- a weighted mean of per-task correctness
  in [0, 1], which is what DSPy optimizes against;
* :mod:`doc_harness.evaluate` accumulates the per-task outcome counts across examples into
  corpus-level precision, recall and F1.

Getting that split wrong produces averages of per-document F1 scores, which is not the same
number as corpus F1 and is not the number anyone downstream means.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from doc_harness.hooks import get_matcher
from doc_harness.registry import Registry, _BaseTask
from doc_harness.values import MatchResult

logger = logging.getLogger(__name__)

# how much document text to quote around a failure in GEPA feedback
FEEDBACK_CONTEXT_CHARS = 400
# how many failing tasks to describe in one feedback string
FEEDBACK_MAX_TASKS = 5


def get_field(container: Any, name: str) -> Any:
    """Read one task's value from an example or prediction, whatever shape it arrives in.

    :param container: A dspy.Example, dspy.Prediction, mapping or plain object
    :param name: The task id
    :returns: The value, or None when the container does not carry that task
    """
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


@dataclass(frozen=True)
class ExampleScore:
    """Everything one example produced, kept together so nothing is recomputed later."""

    results: dict[str, MatchResult]
    aggregate: float
    all_correct: bool
    # tasks skipped because their gold class is below the support floor
    excluded: frozenset[str] = frozenset()

    @property
    def per_task(self) -> dict[str, float]:
        """Return per-task correctness as ones and zeros, in registry order."""
        return {task_id: float(result.correct) for task_id, result in self.results.items()}

    def failures(self) -> dict[str, MatchResult]:
        """Return only the tasks this example got wrong."""
        return {task_id: result for task_id, result in self.results.items() if not result.correct}


@dataclass
class Metric:
    """A registry-derived metric usable directly as a DSPy metric callable.

    :param registry: The parsed tasks.yaml
    :param excluded_classes: Per task, the gold classes below the support floor; these are
        dropped from the optimization target but still scored and reported
    :param abstention: ``scored`` treats a refusal to answer as a first-class answer;
        ``best_guess`` treats it as a wrong one
    """

    registry: Registry
    excluded_classes: Mapping[str, set[str]] = field(default_factory=dict)
    abstention: str = "scored"

    def score_task(self, task: _BaseTask, gold: Any, pred: Any) -> MatchResult:
        """Compare one task's prediction against gold using that task's matcher."""
        matcher = get_matcher(str(task.params["matcher"]))
        result = matcher(get_field(gold, task.id), get_field(pred, task.id), task.params)
        return self._apply_abstention_policy(result)

    def _apply_abstention_policy(self, result: MatchResult) -> MatchResult:
        """Charge an abstention as a wrong answer when the project wants a best guess.

        Under ``scored``, declining to answer costs recall only: it is a miss, not a
        falsehood. Under ``best_guess``, silence is not an available answer, so it costs
        precision too and reads exactly as badly as answering wrongly. That is a different
        metric, not a different prompt, which is why it is a recorded decision.
        """
        if self.abstention != "best_guess":
            return result
        abstained = result.fn > 0 and result.tp == 0 and result.fp == 0
        if not abstained:
            return result
        return replace(result, fp=result.fn, detail=f"{result.detail} (abstention charged as a wrong answer)")

    def _is_excluded(self, task: _BaseTask, result: MatchResult) -> bool:
        """Return whether this example's gold class for this task is below the support floor."""
        excluded = self.excluded_classes.get(task.id)
        if not excluded:
            return False
        gold = result.gold_normalized
        if isinstance(gold, frozenset | set):
            return bool(gold) and {str(item) for item in gold} <= excluded
        return gold is not None and str(gold) in excluded

    def score_example(self, gold: Any, pred: Any) -> ExampleScore:
        """Score every registered task for one example."""
        results: dict[str, MatchResult] = {}
        excluded: set[str] = set()
        weighted_total = 0.0
        weight_sum = 0.0
        for task in self.registry:
            result = self.score_task(task, gold, pred)
            results[task.id] = result
            if self._is_excluded(task, result):
                excluded.add(task.id)
                continue
            weighted_total += task.weight * float(result.correct)
            weight_sum += task.weight
        if weight_sum == 0.0:
            # every task on this example is a below-floor class, so it carries no signal;
            # a constant keeps it from tugging the optimizer in either direction
            logger.debug("example has no measurable task; excluded=%s", sorted(excluded))
            aggregate = 1.0
        else:
            aggregate = weighted_total / weight_sum
        all_correct = all(result.correct for result in results.values())
        return ExampleScore(results, aggregate, all_correct, frozenset(excluded))

    def __call__(self, gold: Any, pred: Any, trace: Any = None) -> float | bool:
        """Score a prediction against gold across all registered tasks.

        :param gold: The labeled example
        :param pred: The program's prediction
        :param trace: Set by DSPy during bootstrapping; when not None, return a strict
            pass/fail rather than a partial score
        :returns: Weighted aggregate in [0, 1], or bool when trace is not None
        """
        score = self.score_example(gold, pred)
        if trace is not None:
            # bootstrapping keeps demonstrations that pass; a partial score here fills the
            # demo pool with half-right examples and teaches the program to be half-right
            return score.all_correct
        return score.aggregate

    def feedback_text(self, gold: Any, score: ExampleScore) -> str:
        """Describe what went wrong in natural language, for a reflective optimizer."""
        failures = score.failures()
        if not failures:
            return "All tasks correct."
        lines = [f"{len(failures)} of {len(score.results)} tasks were wrong."]
        document = get_field(gold, "document")
        for task_id, result in list(failures.items())[:FEEDBACK_MAX_TASKS]:
            task = self.registry.by_id(task_id)
            lines.append(f"- {task_id} ({task.type}): {result.detail}")
            window = _context_window(document, result.gold_normalized)
            if window:
                lines.append(f"  document says: ...{window}...")
        if len(failures) > FEEDBACK_MAX_TASKS:
            lines.append(f"- and {len(failures) - FEEDBACK_MAX_TASKS} more.")
        return "\n".join(lines)

    def gepa_metric(
        self,
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> Any:
        """Score with natural-language feedback, in the shape GEPA expects.

        GEPA calls the metric with the predictor name and trace as well, and reads a
        ``ScoreWithFeedback`` carrying both a number and the text it should reflect on.
        """
        from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

        score = self.score_example(gold, pred)
        return ScoreWithFeedback(score=score.aggregate, feedback=self.feedback_text(gold, score))


def _context_window(document: Any, gold_value: Any) -> str:
    """Quote the document around the gold answer, so feedback shows the evidence."""
    if not isinstance(document, str) or gold_value is None:
        return ""
    needle = str(gold_value)
    if not needle:
        return ""
    position = document.casefold().find(needle.casefold())
    if position < 0:
        return ""
    half = FEEDBACK_CONTEXT_CHARS // 2
    start = max(0, position - half)
    end = min(len(document), position + len(needle) + half)
    return document[start:end].replace("\n", " ").strip()


def build_metric(
    registry: Registry,
    excluded_classes: Mapping[str, set[str]] | None = None,
    abstention: str = "scored",
) -> Metric:
    """Build the metric for a project from its registry.

    :param registry: The parsed tasks.yaml
    :param excluded_classes: Per task, gold classes below the support floor
    :param abstention: ``scored`` or ``best_guess``
    :returns: A callable metric, also exposing ``gepa_metric`` and ``score_example``
    """
    if abstention not in {"scored", "best_guess"}:
        raise ValueError(f"unknown abstention policy {abstention!r}; expected 'scored' or 'best_guess'")
    return Metric(
        registry=registry,
        excluded_classes=dict(excluded_classes or {}),
        abstention=abstention,
    )


def score_all(
    metric: Metric,
    golds: Sequence[Any],
    preds: Sequence[Any],
) -> list[ExampleScore]:
    """Score a whole split, failing loudly if the two sequences disagree in length.

    :param metric: The metric to apply
    :param golds: The labeled examples
    :param preds: The predictions, in the same order
    :returns: One score per example
    """
    if len(golds) != len(preds):
        raise ValueError(f"cannot score {len(preds)} predictions against {len(golds)} gold examples")
    return [metric.score_example(gold, pred) for gold, pred in zip(golds, preds, strict=True)]


MetricCallable = Callable[..., Any]
