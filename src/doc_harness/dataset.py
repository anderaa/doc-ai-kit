"""Labels, splits and the DSPy examples built from them.

``labels.jsonl`` and ``splits.json`` are the project's ground truth. Everything here reads
them; nothing here writes them, and :mod:`doc_harness.guards` enforces that for the
optimization paths.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from doc_harness.registry import Registry, TaskType
from doc_harness.values import QuotedSpan

logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "holdout"]
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "holdout")

# how a document's labels were produced; the holdout must be blind
LabelingMode = Literal["corrected", "blind"]


class DatasetError(ValueError):
    """Raised when labels or splits are missing, malformed or inconsistent."""


@dataclass(frozen=True)
class LabelRecord:
    """One labeled document."""

    doc_id: str
    labels: dict[str, Any]
    stratum: str = "random"
    # the probability this document had of being drawn, used for corpus-level estimates
    inclusion_probability: float = 1.0
    labeling_mode: LabelingMode = "corrected"
    notes: str = ""

    @property
    def weight(self) -> float:
        """Return the inverse-probability weight for corpus-level estimates."""
        if self.inclusion_probability <= 0.0:
            raise DatasetError(f"{self.doc_id}: inclusion_probability must be positive")
        return 1.0 / self.inclusion_probability

    def to_dict(self) -> dict[str, Any]:
        """Return the JSONL row for this record."""
        return {
            "doc_id": self.doc_id,
            "stratum": self.stratum,
            "inclusion_probability": self.inclusion_probability,
            "labeling_mode": self.labeling_mode,
            "notes": self.notes,
            "labels": self.labels,
        }


@dataclass(frozen=True)
class Splits:
    """The fixed train/validation/holdout assignment."""

    seed: int
    strategy: str
    assignments: dict[str, list[str]]
    created_at: str = ""
    # for the cross-validation regime used on small corpora
    folds: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        missing = [name for name in SPLIT_NAMES if name not in self.assignments]
        if missing:
            raise DatasetError(f"splits.json is missing: {', '.join(missing)}")
        seen: dict[str, str] = {}
        for name, doc_ids in self.assignments.items():
            for doc_id in doc_ids:
                if doc_id in seen:
                    raise DatasetError(f"{doc_id} appears in both {seen[doc_id]} and {name}")
                seen[doc_id] = name

    @property
    def train(self) -> list[str]:
        """Return the training document ids."""
        return list(self.assignments["train"])

    @property
    def val(self) -> list[str]:
        """Return the validation document ids."""
        return list(self.assignments["val"])

    @property
    def holdout(self) -> list[str]:
        """Return the holdout document ids."""
        return list(self.assignments["holdout"])

    @property
    def optimization_pool(self) -> list[str]:
        """Return every document an optimizer may see: train plus validation, never holdout."""
        return self.train + self.val

    def to_dict(self) -> dict[str, Any]:
        """Return the splits.json payload."""
        return {
            "seed": self.seed,
            "strategy": self.strategy,
            "created_at": self.created_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "assignments": {name: list(ids) for name, ids in self.assignments.items()},
            "folds": self.folds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Splits:
        """Build splits from the contents of splits.json."""
        try:
            return cls(
                seed=int(data["seed"]),
                strategy=str(data["strategy"]),
                assignments={name: list(ids) for name, ids in data["assignments"].items()},
                created_at=str(data.get("created_at", "")),
                folds=[list(fold) for fold in data.get("folds", [])],
            )
        except KeyError as exc:
            raise DatasetError(f"splits.json is missing key {exc}") from exc


def load_labels(path: Path) -> list[LabelRecord]:
    """Read labels.jsonl, failing loudly on a malformed or duplicated row.

    :param path: The labels file
    :returns: One record per labeled document, in file order
    """
    if not path.exists():
        raise DatasetError(f"{path} does not exist; label some documents first")
    records: list[LabelRecord] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{number}: not valid JSON: {exc}") from exc
        doc_id = row.get("doc_id")
        if not doc_id:
            raise DatasetError(f"{path}:{number}: row has no doc_id")
        if doc_id in seen:
            raise DatasetError(f"{path}:{number}: duplicate doc_id {doc_id!r}")
        seen.add(doc_id)
        if not isinstance(row.get("labels"), Mapping):
            raise DatasetError(f"{path}:{number}: {doc_id} has no 'labels' mapping")
        records.append(
            LabelRecord(
                doc_id=doc_id,
                labels=dict(row["labels"]),
                stratum=str(row.get("stratum", "random")),
                inclusion_probability=float(row.get("inclusion_probability", 1.0)),
                labeling_mode=row.get("labeling_mode", "corrected"),
                notes=str(row.get("notes", "")),
            )
        )
    if not records:
        raise DatasetError(f"{path} contains no labeled documents")
    logger.info("loaded %d labeled documents from %s", len(records), path)
    return records


def write_labels(path: Path, records: Iterable[LabelRecord]) -> None:
    """Write labels.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def load_splits(path: Path) -> Splits:
    """Read splits.json."""
    if not path.exists():
        raise DatasetError(f"{path} does not exist; run `doc-harness make-splits` first")
    return Splits.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_splits(path: Path, splits: Splits) -> None:
    """Write splits.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def validate_against_registry(registry: Registry, records: Sequence[LabelRecord]) -> list[str]:
    """Check labels against the declared tasks, returning human-readable problems.

    Unknown task ids and missing labels are reported rather than raised, because
    ``audit-labels`` shows the whole list at once rather than one problem per run.
    """
    declared = set(registry.ids)
    problems: list[str] = []
    for record in records:
        unknown = sorted(set(record.labels) - declared)
        if unknown:
            problems.append(f"{record.doc_id}: labels for undeclared task(s): {', '.join(unknown)}")
        missing = sorted(declared - set(record.labels))
        if missing:
            problems.append(f"{record.doc_id}: no label for task(s): {', '.join(missing)}")
    for task in registry:
        if task.enum_members is None:
            continue
        allowed = set(task.enum_members)
        for record in records:
            value = record.labels.get(task.id)
            values = value if isinstance(value, list) else [value]
            outside = sorted(str(v) for v in values if v is not None and str(v) not in allowed)
            if outside:
                problems.append(f"{record.doc_id}: {task.id} has value(s) outside the enum: {', '.join(outside)}")
    return problems


def build_examples(
    records: Sequence[LabelRecord],
    texts: Mapping[str, str],
    input_field: str = "document",
    registry: Registry | None = None,
) -> list[Any]:
    """Build DSPy examples from labeled records and cached text.

    :param records: The labeled documents to include
    :param texts: Document id to extracted text
    :param input_field: The name of the program's input field
    :param registry: The parsed tasks.yaml. Given it, gold spans are carried as the passage
        they cover, so a labeled demonstration shows a quote -- the answer the program is
        asked for -- rather than a pair of offsets it is told never to give
    :returns: One ``dspy.Example`` per record, with only the input field marked as input
    """
    import dspy

    missing = [record.doc_id for record in records if record.doc_id not in texts]
    if missing:
        raise DatasetError(f"no cached text for {len(missing)} labeled document(s): {', '.join(sorted(missing))}")
    span_tasks = (
        {task.id for task in registry if TaskType(task.type) is TaskType.SPAN} if registry is not None else set()
    )
    examples = []
    for record in records:
        text = texts[record.doc_id]
        labels = dict(record.labels)
        for task_id in span_tasks:
            labels[task_id] = _as_quoted_span(labels.get(task_id), text, record.doc_id, task_id)
        payload = {input_field: text, "doc_id": record.doc_id, **labels}
        examples.append(dspy.Example(**payload).with_inputs(input_field))
    return examples


def _as_quoted_span(value: Any, text: str, doc_id: str, task_id: str) -> Any:
    """Carry a gold span as the passage it covers, keeping its offsets for scoring."""
    if not isinstance(value, Mapping) or "start" not in value or "end" not in value:
        return value
    start, end = int(value["start"]), int(value["end"])
    if not 0 <= start < end <= len(text):
        raise DatasetError(
            f"{doc_id}: {task_id} span [{start}:{end}] falls outside the cached text ({len(text)} characters); "
            "it was probably labeled against a different extraction"
        )
    return QuotedSpan(text[start:end], start, end)


def select(records: Sequence[LabelRecord], doc_ids: Iterable[str]) -> list[LabelRecord]:
    """Return the records for a set of document ids, failing loudly on a missing one.

    :param records: Every labeled record
    :param doc_ids: The ids to select, in the order wanted
    :returns: The matching records, in that order
    """
    by_id = {record.doc_id: record for record in records}
    wanted = list(doc_ids)
    missing = [doc_id for doc_id in wanted if doc_id not in by_id]
    if missing:
        raise DatasetError(f"{len(missing)} document(s) in the split are not labeled: {', '.join(sorted(missing))}")
    return [by_id[doc_id] for doc_id in wanted]


def weights_for(records: Sequence[LabelRecord]) -> list[float]:
    """Return inverse-probability weights, for corpus-level estimates over an enriched pool."""
    return [record.weight for record in records]
