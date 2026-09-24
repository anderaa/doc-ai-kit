"""Value objects shared by the registry, normalizers, matchers and metric.

These live apart from :mod:`doc_ai_kit.registry` so that normalizers and matchers can be
imported and unit tested without pulling in task-spec parsing or DSPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from pydantic import BaseModel, Field


class Granularity(StrEnum):
    """How precise a date is, preserved through normalization."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"

    @property
    def parts(self) -> int:
        """Return the number of ISO 8601 components this granularity carries."""
        return {Granularity.YEAR: 1, Granularity.MONTH: 2, Granularity.DAY: 3}[self]


class PartialDate(BaseModel):
    """An ISO 8601 date that may be truncated to year or month."""

    value: str = Field(description="ISO 8601 date, possibly partial: 2024, 2024-06 or 2024-06-15")
    granularity: Granularity

    def truncated_to(self, granularity: Granularity) -> str:
        """Return this date's ISO string cut down to the given granularity."""
        return "-".join(self.value.split("-")[: granularity.parts])


class Quantity(BaseModel):
    """A number with an optional unit."""

    value: float
    unit: str | None = None


class Span(BaseModel):
    """A character offset range into the extracted text."""

    start: int
    end: int
    # the text the offsets cover, kept so a report can show the passage and not just numbers
    text: str | None = None

    def tokens(self) -> set[int]:
        """Return the set of character offsets this span covers."""
        return set(range(self.start, self.end))


class QuotedSpan(str):
    """A gold span as the model would give it: the passage's text, carrying its offsets.

    Programs are asked to quote a span rather than count characters, so a gold span shown to
    them as a labeled demonstration has to look like a quote too. Scoring still happens on the
    offsets, which the span normalizer reads from this object.
    """

    start: int
    end: int

    def __new__(cls, text: str, start: int, end: int) -> QuotedSpan:
        instance = super().__new__(cls, text)
        instance.start = start
        instance.end = end
        return instance


@dataclass(frozen=True)
class MatchResult:
    """The outcome of comparing one prediction against one gold value for one task.

    Counts rather than a single verdict, because list-valued tasks produce several
    outcomes from a single comparison, and because a wrong non-null value counts as both a
    false positive and a false negative.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    correct: bool = False
    detail: str = ""
    # the normalized values compared, kept so evaluate.py can build confusion matrices
    gold_normalized: object | None = None
    pred_normalized: object | None = None
    # how close the call was, for thresholded matchers: a similarity, an overlap ratio, or a
    # distance-to-tolerance ratio, beside the threshold it was judged against. Kept so a human
    # can adjudicate the decisions a threshold actually made rather than parse them from text.
    measure: float | None = None
    threshold: float | None = None
    # side-by-side results under other criteria, e.g. {"strict": ...} for fuzzy tasks
    alternates: dict[str, MatchResult] = field(default_factory=dict)

    @property
    def counted(self) -> int:
        """Return the total number of outcomes recorded, used to reconcile counts."""
        return self.tp + self.fp + self.fn + self.tn

    def with_values(self, gold: object | None, pred: object | None) -> MatchResult:
        """Return a copy carrying the normalized values that produced this result."""
        return replace(self, gold_normalized=gold, pred_normalized=pred)

    def with_alternates(self, **alternates: MatchResult) -> MatchResult:
        """Return a copy carrying side-by-side results under other criteria."""
        return replace(self, alternates={**self.alternates, **alternates})


def true_negative(detail: str = "") -> MatchResult:
    """Build the result for a correct abstention: predicted null when gold is null."""
    return MatchResult(tn=1, correct=True, detail=detail)


def true_positive(detail: str = "") -> MatchResult:
    """Build the result for a correct non-null answer."""
    return MatchResult(tp=1, correct=True, detail=detail)


def false_positive(detail: str = "") -> MatchResult:
    """Build the result for a value predicted where gold is null."""
    return MatchResult(fp=1, correct=False, detail=detail)


def false_negative(detail: str = "") -> MatchResult:
    """Build the result for an abstention where gold has a value."""
    return MatchResult(fn=1, correct=False, detail=detail)


def wrong_value(detail: str = "") -> MatchResult:
    """Build the result for a wrong non-null answer, which is both a miss and a spurious answer."""
    return MatchResult(fp=1, fn=1, correct=False, detail=detail)
