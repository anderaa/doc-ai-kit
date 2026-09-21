"""Splits, stratification, enrichment strata and the support floor.

Everything downstream rests on this. Three rules drive the design:

* splits are created from a fixed seed **before any model sees any document**, so the
  holdout cannot be chosen to flatter a program that already exists;
* a class may never appear in the holdout without appearing in train, or the holdout is
  measuring something the program was never shown;
* support, not document count, is the binding constraint. At prevalence p a random sample
  needs about m/p documents to yield m examples of a class, so random labeling alone cannot
  reach rare classes and the enrichment strata have to carry their inclusion probabilities.
"""

from __future__ import annotations

import logging
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from doc_harness.dataset import LabelRecord, Splits
from doc_harness.registry import Registry, TaskType
from doc_harness.stats import HALF_WIDTH_TABLE, readable_at, wilson_half_width

logger = logging.getLogger(__name__)

# ratio tables by labeled-corpus size, from the project protocol
RATIOS: tuple[tuple[int, tuple[float, float, float], str], ...] = (
    (200, (0.50, 0.25, 0.25), "50/25/25"),
    (100, (0.45, 0.25, 0.30), "45/25/30"),
)
# below this, five-fold cross-validation on 70% with a 30% holdout
CV_MIN = 50
CV_FOLDS = 5
CV_TRAIN_SHARE = 0.70
# below this there are too few points for automated search at all
TOO_FEW = 50

STRATA = ("random", "keyword", "model_nominated")
Stratum = Literal["random", "keyword", "model_nominated"]

SupportChoice = Literal["enrich", "collapse", "binary_detection", "report_unmeasured"]

# only these task types have a class space; an extraction task's values are not classes, and
# treating them as such would make every document's contract number look like a rare class
CLASSIFICATION_TYPES = frozenset({TaskType.BINARY, TaskType.MULTICLASS, TaskType.MULTILABEL})


class SplitError(ValueError):
    """Raised when a split cannot be built without breaking one of the rules above."""


@dataclass(frozen=True)
class ClassSupport:
    """How many labeled examples one class of one task has, and what that can support."""

    task_id: str
    label: str
    count: int
    total: int

    @property
    def prevalence(self) -> float:
        """Return the share of labeled documents carrying this class."""
        return self.count / self.total if self.total else 0.0

    @property
    def half_width(self) -> float:
        """Return the approximate 95% Wilson half-width for recall near 0.8."""
        return wilson_half_width(round(0.8 * self.count), self.count) if self.count else 0.5

    @property
    def readable_as(self) -> str:
        """Return the strongest claim this many examples can carry."""
        return readable_at(self.count)

    def documents_needed(self, target: int) -> int:
        """Return roughly how many random documents would yield ``target`` examples.

        At prevalence p, a random sample needs about m/p documents. A 2% class needs around
        1,500 documents for 30 examples, which is why random labeling cannot reach it.
        """
        if self.prevalence <= 0.0:
            return 0
        return math.ceil(target / self.prevalence)


@dataclass(frozen=True)
class SupportDecision:
    """A recorded human decision about a class that falls below the support floor."""

    task_id: str
    label: str
    count: int
    choice: SupportChoice
    rationale: str = ""
    decided_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_markdown(self) -> str:
        """Render this decision as a decisions.md entry."""
        return (
            f"## {self.decided_at} - support floor: {self.task_id}/{self.label}\n\n"
            f"- Labeled examples: **{self.count}** ({readable_at(self.count)})\n"
            f"- Choice: **{self.choice}**\n"
            f"- Rationale: {self.rationale or '(none given)'}\n"
        )


SUPPORT_OPTIONS: tuple[tuple[SupportChoice, str], ...] = (
    ("enrich", "Label more documents of this class through a keyword or model-nominated stratum."),
    ("collapse", "Fold this class into a neighbouring class or into 'other' in tasks.yaml."),
    ("binary_detection", "Split it out as its own binary detection task."),
    (
        "report_unmeasured",
        "Keep scoring and reporting it, but drop it from the optimization target "
        "(config.yaml metric.excluded_classes).",
    ),
)


def class_supports(registry: Registry, records: Sequence[LabelRecord]) -> dict[str, dict[str, ClassSupport]]:
    """Count labeled examples per class, for every classification task.

    :param registry: The parsed tasks.yaml
    :param records: The labeled documents
    :returns: Task id to label to support
    """
    total = len(records)
    supports: dict[str, dict[str, ClassSupport]] = {}
    for task in registry:
        if TaskType(task.type) not in CLASSIFICATION_TYPES:
            continue
        counts: dict[str, int] = {}
        for label in _declared_labels(task):
            counts.setdefault(label, 0)
        for record in records:
            for label in _labels_of(record.labels.get(task.id)):
                counts[label] = counts.get(label, 0) + 1
        supports[task.id] = {
            label: ClassSupport(task_id=task.id, label=label, count=count, total=total)
            for label, count in counts.items()
        }
    return supports


def _declared_labels(task: Any) -> list[str]:
    if TaskType(task.type) is TaskType.BINARY:
        return ["true", "false"]
    return list(task.enum_members or [])


def _labels_of(value: Any) -> list[str]:
    """Render one label value as the list of classes it carries."""
    if value is None:
        return []
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def below_floor(
    supports: Mapping[str, Mapping[str, ClassSupport]],
    support_floor: int,
) -> list[ClassSupport]:
    """Return every class with fewer labeled examples than the floor, rarest first."""
    found = [
        support
        for task_supports in supports.values()
        for support in task_supports.values()
        if support.count < support_floor
    ]
    return sorted(found, key=lambda support: (support.count, support.task_id, support.label))


def support_floor_prompt(support: ClassSupport, support_floor: int) -> str:
    """Render the numbers and options a human needs to decide about one rare class."""
    lines = [
        f"{support.task_id}/{support.label}: {support.count} labeled example(s) "
        f"out of {support.total} ({support.prevalence:.1%}), below the floor of {support_floor}.",
        f"  95% recall half-width at this support: +/-{support.half_width * 100:.0f} points "
        f"({support.readable_as}).",
        f"  Reaching {support_floor} examples by random sampling would take about "
        f"{support.documents_needed(support_floor):,} documents.",
        "  Options:",
    ]
    lines += [f"    - {choice}: {description}" for choice, description in SUPPORT_OPTIONS]
    return "\n".join(lines)


def half_width_table() -> list[str]:
    """Render the sample-size table as markdown rows, for reports and guidance files."""
    lines = ["| examples in split | 95% half-width | usable for |", "| --- | --- | --- |"]
    for count, half_width, description in HALF_WIDTH_TABLE:
        lines.append(f"| {count} | +/-{half_width * 100:.0f} pts | {description} |")
    return lines


def _strategy_for(count: int) -> tuple[tuple[float, float, float], str, bool]:
    """Return the ratios, the strategy name, and whether cross-validation applies."""
    for threshold, ratios, name in RATIOS:
        if count >= threshold:
            return ratios, name, False
    if count >= CV_MIN:
        return (CV_TRAIN_SHARE, 0.0, 1.0 - CV_TRAIN_SHARE), f"{CV_FOLDS}-fold CV on 70% / 30% holdout", True
    return (0.45, 0.25, 0.30), "45/25/30 (below the recommended minimum)", False


def _stratification_key(
    record: LabelRecord,
    rarity: Mapping[tuple[str, str], int],
) -> str:
    """Return the key a document is stratified on: the rarest class it carries.

    Stratifying on every task at once is not solvable in general, so each document is
    placed by its scarcest signal. That is what protects the classes most at risk of
    vanishing from a split.
    """
    carried = [
        ((task_id, label), count)
        for (task_id, label), count in rarity.items()
        if label in _labels_of(record.labels.get(task_id))
    ]
    if not carried:
        return "(no class)"
    (task_id, label), _count = min(carried, key=lambda item: (item[1], item[0]))
    return f"{task_id}={label}"


def _allocate(size: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split a group of documents across train, val and holdout by largest remainder.

    Small groups fill train first, then validation, then holdout: a class with a single
    example belongs in train, because a class in the holdout but not in train is a number
    measured against something the program was never shown.
    """
    if size <= 0:
        return (0, 0, 0)
    if size < 3:
        return (1, 0, 0) if size == 1 else (1, 1, 0)
    raw = [size * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return (counts[0], counts[1], counts[2])


def make_splits(
    registry: Registry,
    records: Sequence[LabelRecord],
    seed: int,
    ratios: tuple[float, float, float] | None = None,
) -> Splits:
    """Build the train/validation/holdout assignment.

    :param registry: The parsed tasks.yaml
    :param records: Every labeled document
    :param seed: The fixed seed, committed to splits.json
    :param ratios: Optional explicit ratios, overriding the size-based table
    :returns: The split assignment, with folds populated under the cross-validation regime
    """
    if not records:
        raise SplitError("cannot build splits from zero labeled documents")
    count = len(records)
    table_ratios, strategy, use_cv = _strategy_for(count)
    ratios = ratios or table_ratios
    if count < TOO_FEW:
        logger.warning(
            "only %d labeled documents: too few points for automated search. "
            "Expect wide intervals and treat any comparison between programs as provisional",
            count,
        )

    supports = class_supports(registry, records)
    rarity = {
        (task_id, label): support.count
        for task_id, task_supports in supports.items()
        for label, support in task_supports.items()
        if support.count > 0
    }
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(_stratification_key(record, rarity), []).append(record.doc_id)

    rng = random.Random(seed)
    assignments: dict[str, list[str]] = {"train": [], "val": [], "holdout": []}
    for key in sorted(groups):
        members = sorted(groups[key])
        rng.shuffle(members)
        train_count, val_count, holdout_count = _allocate(len(members), ratios)
        assignments["train"] += members[:train_count]
        assignments["val"] += members[train_count : train_count + val_count]
        assignments["holdout"] += members[train_count + val_count : train_count + val_count + holdout_count]

    assignments = _repair_holdout_only_classes(registry, records, assignments)
    for name in assignments:
        assignments[name] = sorted(assignments[name])

    folds: list[list[str]] = []
    if use_cv:
        pool = assignments["train"] + assignments["val"]
        rng.shuffle(pool)
        folds = [sorted(pool[index::CV_FOLDS]) for index in range(CV_FOLDS)]
        # the last fold doubles as the validation split, so every downstream command works
        # unchanged while the folds remain available for cross-validation
        assignments["val"] = list(folds[-1])
        assignments["train"] = sorted(doc_id for fold in folds[:-1] for doc_id in fold)

    splits = Splits(
        seed=seed,
        strategy=strategy,
        assignments=assignments,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        folds=folds,
    )
    _verify(registry, records, splits)
    logger.info(
        "built %s splits from %d documents: %d train, %d val, %d holdout (seed %d)",
        strategy,
        count,
        len(splits.train),
        len(splits.val),
        len(splits.holdout),
        seed,
    )
    return splits


def _classes_in(registry: Registry, records: Mapping[str, LabelRecord], doc_ids: Sequence[str]) -> set[tuple[str, str]]:
    """Return every (task, class) pair present in a set of documents."""
    present: set[tuple[str, str]] = set()
    classification_tasks = [task for task in registry if TaskType(task.type) in CLASSIFICATION_TYPES]
    for doc_id in doc_ids:
        record = records[doc_id]
        for task in classification_tasks:
            for label in _labels_of(record.labels.get(task.id)):
                present.add((task.id, label))
    return present


def _repair_holdout_only_classes(
    registry: Registry,
    records: Sequence[LabelRecord],
    assignments: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Move documents so no class appears in the holdout without appearing in train."""
    by_id = {record.doc_id: record for record in records}
    repaired = {name: list(ids) for name, ids in assignments.items()}
    for _attempt in range(len(records)):
        train_classes = _classes_in(registry, by_id, repaired["train"])
        holdout_classes = _classes_in(registry, by_id, repaired["holdout"])
        orphans = holdout_classes - train_classes
        if not orphans:
            return repaired
        task_id, label = sorted(orphans)[0]
        carrier = next(
            doc_id for doc_id in repaired["holdout"] if label in _labels_of(by_id[doc_id].labels.get(task_id))
        )
        repaired["holdout"].remove(carrier)
        repaired["train"].append(carrier)
        logger.info("moved %s from holdout to train: %s=%s appeared nowhere in train", carrier, task_id, label)
    raise SplitError("could not place every class in train; the corpus has classes with too few examples")


def _verify(registry: Registry, records: Sequence[LabelRecord], splits: Splits) -> None:
    """Check the invariants that make a split trustworthy, failing loudly on any breach."""
    by_id = {record.doc_id: record for record in records}
    assigned = splits.train + splits.val + splits.holdout
    if len(assigned) != len(set(assigned)):
        raise SplitError("a document was assigned to more than one split")
    if set(assigned) != set(by_id):
        missing = sorted(set(by_id) - set(assigned))
        extra = sorted(set(assigned) - set(by_id))
        raise SplitError(f"split does not cover the corpus; missing={missing} unknown={extra}")
    orphans = _classes_in(registry, by_id, splits.holdout) - _classes_in(registry, by_id, splits.train)
    if orphans:
        readable = ", ".join(f"{task_id}={label}" for task_id, label in sorted(orphans))
        raise SplitError(f"class(es) in the holdout but not in train: {readable}")


def keyword_stratum(texts: Mapping[str, str], pattern: str) -> list[str]:
    """Return documents whose cached text matches a pattern.

    Model-independent by construction, which is the point: it can surface examples the
    model would never nominate because it never finds them.

    :param texts: Document id to extracted text
    :param pattern: A regular expression, matched case-insensitively
    :returns: The matching document ids, sorted
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    return sorted(doc_id for doc_id, text in texts.items() if compiled.search(text))


def model_nominated_stratum(
    predictions: Mapping[str, Any],
    task_id: str,
    label: str,
) -> list[str]:
    """Return documents a baseline predicted as the target class.

    Biased on purpose and labeled as such: it cannot surface examples the model misses, so
    recall estimated on this stratum alone is inflated. It is never the only stratum.

    :param predictions: Document id to prediction
    :param task_id: The task to look at
    :param label: The class being enriched for
    :returns: The nominated document ids, sorted
    """
    nominated = []
    for doc_id, prediction in predictions.items():
        value = prediction.get(task_id) if isinstance(prediction, Mapping) else getattr(prediction, task_id, None)
        if label in _labels_of(value):
            nominated.append(doc_id)
    return sorted(nominated)


def inclusion_probability(selected: int, available: int) -> float:
    """Return the probability a document in a stratum had of being selected for labeling.

    Corpus-level metrics weight by the inverse of this; per-class metrics do not, and the
    report has to say which is which.
    """
    if available <= 0:
        raise SplitError("cannot compute an inclusion probability for an empty stratum")
    if selected > available:
        raise SplitError(f"cannot select {selected} documents from a stratum of {available}")
    return selected / available


def plan_enrichment(
    candidates: Sequence[str],
    already_labeled: Sequence[str],
    target: int,
    stratum: Stratum,
    seed: int,
) -> tuple[list[str], float]:
    """Choose documents to label from an enrichment stratum.

    :param candidates: Document ids in the stratum
    :param already_labeled: Documents that are already labeled and must not be redrawn
    :param target: How many new documents to label
    :param stratum: Which stratum this is, recorded on every label
    :param seed: Fixed so the draw is reproducible
    :returns: The selected document ids and their inclusion probability
    """
    if stratum not in STRATA:
        raise SplitError(f"unknown stratum {stratum!r}; known: {', '.join(STRATA)}")
    pool = sorted(set(candidates) - set(already_labeled))
    if not pool:
        raise SplitError(f"the {stratum} stratum has no unlabeled documents left")
    take = min(target, len(pool))
    rng = random.Random(f"{seed}:{stratum}")
    chosen = sorted(rng.sample(pool, take))
    probability = inclusion_probability(take, len(pool))
    if take < target:
        logger.warning("%s stratum only had %d unlabeled document(s); wanted %d", stratum, take, target)
    return chosen, probability
