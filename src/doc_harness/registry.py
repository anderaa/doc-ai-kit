"""Task registry: ``tasks.yaml`` parsed into typed objects.

``tasks.yaml`` is the source of truth. Signatures, output types, normalizers, matchers and
report sections are all generated from it, so adding a task means adding YAML rather than
editing the harness. Validation is strict and loud: a typo in a task type must not silently
become a free-text field.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from doc_harness.values import Granularity, PartialDate, Quantity, Span

logger = logging.getLogger(__name__)

# the starting instruction before any optimizer has rewritten it; deliberately plain, so a
# baseline measures the questions themselves rather than a hand-tuned preamble
DEFAULT_INSTRUCTIONS = (
    "Read the document and answer each question about it. " "Answer null when the document does not state the answer."
)


class TaskSpecError(ValueError):
    """Raised when tasks.yaml is malformed; the message always names the offending task."""


class TaskType(StrEnum):
    """The task types the harness knows how to score."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    EXTRACT_EXACT = "extract_exact"
    EXTRACT_FUZZY = "extract_fuzzy"
    EXTRACT_LIST = "extract_list"
    EXTRACT_NUMERIC = "extract_numeric"
    EXTRACT_DATE = "extract_date"
    SPAN = "span"


class _Strict(BaseModel):
    """Base for every spec model: unknown keys are errors, not silently ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EnumOutput(_Strict):
    """The declared value space of a classification task."""

    enum: list[str] = Field(min_length=1)
    nullable: bool | None = None
    # maps surface forms seen in gold or predictions onto canonical members
    synonyms: dict[str, str] = Field(default_factory=dict)

    @field_validator("enum")
    @classmethod
    def _unique_members(cls, value: list[str]) -> list[str]:
        duplicates = sorted({m for m in value if value.count(m) > 1})
        if duplicates:
            raise ValueError(f"duplicate enum members: {', '.join(duplicates)}")
        return value

    @model_validator(mode="after")
    def _synonyms_target_members(self) -> EnumOutput:
        unknown = sorted({t for t in self.synonyms.values() if t not in self.enum})
        if unknown:
            raise ValueError(f"synonyms map to non-members: {', '.join(unknown)}")
        return self


class ScalarOutput(_Strict):
    """The declared value space of a task with no enumerated members."""

    nullable: bool | None = None


class _BaseMatch(_Strict):
    """Shared shape of every ``match`` block."""

    matcher: str
    normalizer: str


class IdentityMatch(_BaseMatch):
    """Equality after canonicalisation, for booleans and single-label classification."""

    matcher: str = "identity"
    normalizer: str = "enum"


class BooleanMatch(_BaseMatch):
    """Equality after boolean canonicalisation."""

    matcher: str = "identity"
    normalizer: str = "boolean"


class SetMatch(_BaseMatch):
    """Set comparison, for multilabel classification."""

    matcher: str = "set"
    normalizer: str = "enum"


class ExactMatch(_BaseMatch):
    """Normalize, then require equality."""

    matcher: str = "exact"
    normalizer: str = "free_string"


class FuzzyMatch(_BaseMatch):
    """Normalize, then require similarity at or above ``theta``.

    ``theta`` has no default on purpose: a silent threshold decides what counts as correct.
    """

    matcher: str = "entity_name"
    normalizer: str = "entity_name"
    theta: float = Field(gt=0.0, le=1.0)


class ListMatch(_BaseMatch):
    """Greedy bipartite assignment between predicted and gold items."""

    matcher: str = "list"
    normalizer: str = "entity_name"
    pair_matcher: str = "entity_name"
    theta: float = Field(default=1.0, gt=0.0, le=1.0)


class NumericMatch(_BaseMatch):
    """Unit-normalize, then accept within a tolerance."""

    matcher: str = "numeric"
    normalizer: str = "numeric"
    tolerance: float = Field(ge=0.0)
    tolerance_kind: Literal["absolute", "relative"] = "relative"
    unit: str | None = None


class DateMatch(_BaseMatch):
    """Granularity-aware date equality.

    ``allow_coarser`` defaults to false: a year-only prediction does not match a full gold
    date unless the project says so, and that decision is recorded.
    """

    matcher: str = "date"
    normalizer: str = "date"
    granularity: Granularity = Granularity.DAY
    allow_coarser: bool = False


class SpanMatch(_BaseMatch):
    """Character-overlap comparison between predicted and gold offsets."""

    matcher: str = "span"
    normalizer: str = "span"
    overlap_threshold: float = Field(default=0.5, gt=0.0, le=1.0)


class _BaseTask(_Strict):
    """Fields common to every task entry."""

    id: str = Field(min_length=1)
    # narrowed to a single Literal member by each subclass, which is what drives the
    # discriminated union; declared here so every spec is statically known to carry it
    type: TaskType
    # the headline number for this task type, narrowed to a Literal by each subclass
    primary_metric: str
    question: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0.0)
    nullable: bool = True
    group: str | None = None

    @property
    def is_nullable(self) -> bool:
        """Return whether this task may legitimately answer null."""
        declared = getattr(self.output, "nullable", None) if hasattr(self, "output") else None
        return self.nullable if declared is None else bool(declared)

    @property
    def enum_members(self) -> list[str] | None:
        """Return the declared class members, or None for tasks with no value space."""
        output = getattr(self, "output", None)
        return list(output.enum) if isinstance(output, EnumOutput) else None

    @property
    def params(self) -> dict[str, Any]:
        """Return everything a normalizer or matcher needs, as one flat mapping."""
        match_block: _BaseMatch = self.match  # type: ignore[attr-defined]
        params: dict[str, Any] = dict(match_block.model_dump())
        params["task_id"] = self.id
        params["task_type"] = str(self.type)
        params["nullable"] = self.is_nullable
        output = getattr(self, "output", None)
        if isinstance(output, EnumOutput):
            params["enum"] = list(output.enum)
            params["synonyms"] = dict(output.synonyms)
        return params


class BinaryTask(_BaseTask):
    """A yes/no question about the document."""

    type: Literal[TaskType.BINARY]
    output: ScalarOutput | None = None
    match: BooleanMatch = BooleanMatch()
    primary_metric: Literal["f1_positive"] = "f1_positive"


class MulticlassTask(_BaseTask):
    """Exactly one member of a declared enum."""

    type: Literal[TaskType.MULTICLASS]
    output: EnumOutput
    match: IdentityMatch = IdentityMatch()
    primary_metric: Literal["macro_f1"] = "macro_f1"


class MultilabelTask(_BaseTask):
    """Any subset of a declared enum."""

    type: Literal[TaskType.MULTILABEL]
    output: EnumOutput
    match: SetMatch = SetMatch()
    primary_metric: Literal["macro_f1"] = "macro_f1"


class ExtractExactTask(_BaseTask):
    """A string that must match gold exactly once normalized."""

    type: Literal[TaskType.EXTRACT_EXACT]
    output: ScalarOutput | None = None
    match: ExactMatch = ExactMatch()
    primary_metric: Literal["f1"] = "f1"


class ExtractFuzzyTask(_BaseTask):
    """A string scored at a similarity threshold, always reported beside strict equality."""

    type: Literal[TaskType.EXTRACT_FUZZY]
    output: ScalarOutput | None = None
    match: FuzzyMatch
    primary_metric: Literal["f1"] = "f1"


class ExtractListTask(_BaseTask):
    """Zero or more strings, compared as a set."""

    type: Literal[TaskType.EXTRACT_LIST]
    output: ScalarOutput | None = None
    match: ListMatch = ListMatch()
    primary_metric: Literal["f1"] = "f1"


class ExtractNumericTask(_BaseTask):
    """A number with an optional unit, accepted within a tolerance."""

    type: Literal[TaskType.EXTRACT_NUMERIC]
    output: ScalarOutput | None = None
    match: NumericMatch
    primary_metric: Literal["f1"] = "f1"


class ExtractDateTask(_BaseTask):
    """An ISO 8601 date, possibly partial."""

    type: Literal[TaskType.EXTRACT_DATE]
    output: ScalarOutput | None = None
    match: DateMatch = DateMatch()
    primary_metric: Literal["f1"] = "f1"


class SpanTask(_BaseTask):
    """Character offsets into the extracted text."""

    type: Literal[TaskType.SPAN]
    output: ScalarOutput | None = None
    match: SpanMatch = SpanMatch()
    primary_metric: Literal["token_f1"] = "token_f1"


TaskSpec = Annotated[
    BinaryTask
    | MulticlassTask
    | MultilabelTask
    | ExtractExactTask
    | ExtractFuzzyTask
    | ExtractListTask
    | ExtractNumericTask
    | ExtractDateTask
    | SpanTask,
    Field(discriminator="type"),
]

_MODEL_BY_TYPE: dict[TaskType, type[_BaseTask]] = {
    TaskType.BINARY: BinaryTask,
    TaskType.MULTICLASS: MulticlassTask,
    TaskType.MULTILABEL: MultilabelTask,
    TaskType.EXTRACT_EXACT: ExtractExactTask,
    TaskType.EXTRACT_FUZZY: ExtractFuzzyTask,
    TaskType.EXTRACT_LIST: ExtractListTask,
    TaskType.EXTRACT_NUMERIC: ExtractNumericTask,
    TaskType.EXTRACT_DATE: ExtractDateTask,
    TaskType.SPAN: SpanTask,
}

# the python type each task's DSPy output field carries, before nullability is applied
_BASE_OUTPUT_TYPE: dict[TaskType, Any] = {
    TaskType.BINARY: bool,
    TaskType.EXTRACT_EXACT: str,
    TaskType.EXTRACT_FUZZY: str,
    TaskType.EXTRACT_LIST: list[str],
    TaskType.EXTRACT_NUMERIC: Quantity,
    TaskType.EXTRACT_DATE: PartialDate,
    TaskType.SPAN: Span,
}


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def _parse_task(raw: Any, index: int) -> _BaseTask:
    """Parse one raw task entry, attributing every failure to a task id."""
    if not isinstance(raw, Mapping):
        raise TaskSpecError(f"task #{index}: expected a mapping, got {type(raw).__name__}")
    task_id = raw.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise TaskSpecError(f"task #{index}: missing or empty 'id'")
    declared = raw.get("type")
    # checked before pydantic so the error names the type, not a union of nine alternatives
    if declared not in set(TaskType):
        known = ", ".join(t.value for t in TaskType)
        raise TaskSpecError(f"task {task_id!r}: unknown type {declared!r}; known types: {known}")
    model = _MODEL_BY_TYPE[TaskType(declared)]
    try:
        return model.model_validate(dict(raw))
    except ValidationError as exc:
        raise TaskSpecError(f"task {task_id!r}: {_format_validation_error(exc)}") from exc


class Registry:
    """The parsed contents of ``tasks.yaml``."""

    def __init__(self, tasks: Sequence[_BaseTask], source: Path | None = None) -> None:
        """Build a registry from already-parsed task specs.

        :param tasks: The task specs, in declaration order
        :param source: The file they were read from, recorded for run metadata
        """
        if not tasks:
            raise TaskSpecError("tasks.yaml declares no tasks")
        seen: dict[str, int] = {}
        for position, task in enumerate(tasks):
            if task.id in seen:
                raise TaskSpecError(f"task {task.id!r}: duplicate id, already declared at #{seen[task.id]}")
            seen[task.id] = position
        self.tasks: list[_BaseTask] = list(tasks)
        self.source = source
        self._by_id = {task.id: task for task in self.tasks}

    @classmethod
    def from_mapping(cls, data: Any, source: Path | None = None) -> Registry:
        """Build a registry from the raw contents of a tasks.yaml document."""
        if not isinstance(data, Mapping) or "tasks" not in data:
            raise TaskSpecError("tasks.yaml must be a mapping with a top-level 'tasks' key")
        raw_tasks = data["tasks"]
        if raw_tasks is None:
            # a freshly scaffolded tasks.yaml has every example commented out
            raise TaskSpecError("tasks.yaml declares no tasks; uncomment or add at least one task entry")
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, str | bytes):
            raise TaskSpecError("'tasks' must be a list of task entries")
        unknown_top_level = sorted(set(data) - {"tasks"})
        if unknown_top_level:
            raise TaskSpecError(f"unknown top-level keys in tasks.yaml: {', '.join(unknown_top_level)}")
        return cls([_parse_task(raw, index) for index, raw in enumerate(raw_tasks)], source=source)

    @classmethod
    def from_yaml(cls, path: Path) -> Registry:
        """Load and validate a tasks.yaml file."""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TaskSpecError(f"{path}: not valid YAML: {exc}") from exc
        registry = cls.from_mapping(data, source=path)
        logger.info("loaded %d tasks from %s", len(registry.tasks), path)
        return registry

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[_BaseTask]:
        return iter(self.tasks)

    @property
    def ids(self) -> list[str]:
        """Return every task id in declaration order."""
        return [task.id for task in self.tasks]

    def by_id(self, task_id: str) -> _BaseTask:
        """Look up one task, failing loudly on an unknown id."""
        try:
            return self._by_id[task_id]
        except KeyError:
            raise TaskSpecError(f"unknown task id {task_id!r}; declared: {', '.join(self.ids)}") from None

    def groups(self) -> dict[str, list[_BaseTask]]:
        """Return tasks bucketed by ``group``, defaulting to a single ``all`` predictor.

        DSPy tunes instructions per predictor, so a task needing its own tuned instruction
        needs its own group and therefore its own module.
        """
        buckets: dict[str, list[_BaseTask]] = {}
        for task in self.tasks:
            buckets.setdefault(task.group or "all", []).append(task)
        return buckets

    def output_type(self, task: _BaseTask) -> Any:
        """Return the python type the DSPy output field for this task should carry."""
        task_type = TaskType(task.type)
        if task_type is TaskType.MULTICLASS:
            base: Any = Literal[tuple(task.enum_members or [])]
        elif task_type is TaskType.MULTILABEL:
            base = list[Literal[tuple(task.enum_members or [])]]  # type: ignore[misc]
        else:
            base = _BASE_OUTPUT_TYPE[task_type]
        # multilabel abstains with an empty list, so nullability would only add ambiguity
        if task.is_nullable and task_type is not TaskType.MULTILABEL:
            return base | None
        return base

    def total_weight(self) -> float:
        """Return the sum of task weights, used to normalize the aggregate metric."""
        return sum(task.weight for task in self.tasks)

    def signature_for(
        self,
        group: str,
        instructions: str | None = None,
        input_field: str = "document",
        input_description: str = "The full extracted text of the document.",
    ) -> Any:
        """Build the DSPy signature for one group of tasks.

        The docstring is the tunable instruction and each field description carries that
        task's ``question``, which is what an optimizer rewrites.

        :param group: The group name, as returned by :meth:`groups`
        :param instructions: The starting instruction; a neutral default is used if omitted
        :param input_field: The name of the single input field
        :param input_description: The description attached to the input field
        :returns: A ``dspy.Signature`` subclass
        """
        # imported here so the registry can be used, and tested, without loading DSPy
        import dspy
        from dspy.signatures.signature import make_signature

        tasks = self.groups().get(group)
        if not tasks:
            raise TaskSpecError(f"unknown task group {group!r}; declared: {', '.join(sorted(self.groups()))}")
        fields: dict[str, tuple[Any, Any]] = {
            input_field: (str, dspy.InputField(desc=input_description)),
        }
        for task in tasks:
            fields[task.id] = (self.output_type(task), dspy.OutputField(desc=task.question.strip()))
        name = "".join(part.capitalize() for part in group.replace("-", "_").split("_")) + "Signature"
        return make_signature(fields, instructions or DEFAULT_INSTRUCTIONS, name)

    def signatures(self, instructions: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Build one DSPy signature per group.

        DSPy tunes instructions per predictor, so a task that needs its own tuned
        instruction needs its own group, and therefore its own signature and module.
        """
        instructions = instructions or {}
        return {group: self.signature_for(group, instructions.get(group)) for group in self.groups()}
