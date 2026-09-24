"""Matchers: compare a normalized prediction against a normalized gold value.

Every matcher returns a :class:`MatchResult` carrying outcome counts rather than a verdict,
because scoring has to be able to say things a boolean cannot:

* a wrong non-null value is both a false positive and a false negative;
* predicting null when gold is null is a true negative, and is scored, not ignored;
* a list-valued task produces several outcomes from a single comparison.

Matchers are pure and are applied identically to gold and prediction. They are the most
common source of fake results -- a bad matcher makes a good prompt look broken and sends
the optimizer chasing a bug -- so they are unit tested against adversarial fixtures before
anything downstream is trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rapidfuzz import fuzz

from doc_ai_kit.hooks import get_normalizer, register_matcher
from doc_ai_kit.values import (
    Granularity,
    MatchResult,
    PartialDate,
    Quantity,
    Span,
    false_negative,
    false_positive,
    true_negative,
    true_positive,
    wrong_value,
)


def _render(value: Any) -> str:
    """Render a normalized value for a failure line."""
    if value is None:
        return "null"
    if isinstance(value, PartialDate):
        return value.value
    if isinstance(value, Quantity):
        return f"{value.value:g}{' ' + value.unit if value.unit else ''}"
    if isinstance(value, Span):
        if value.text:
            excerpt = " ".join(value.text.split())
            return f"[{value.start}:{value.end}] \"{excerpt[:60]}{'...' if len(excerpt) > 60 else ''}\""
        return f"[{value.start}:{value.end}]"
    if isinstance(value, frozenset | set):
        return "{" + ", ".join(sorted(str(v) for v in value)) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(str(v) for v in value) + "]"
    return str(value)


def _line(gold: Any, pred: Any) -> str:
    return f"gold={_render(gold)} predicted={_render(pred)}"


def _scalar_outcome(gold: Any, pred: Any, equal: bool) -> MatchResult:
    """Turn a normalized pair and an equality verdict into counted outcomes."""
    detail = _line(gold, pred)
    if gold is None and pred is None:
        return true_negative(detail).with_values(gold, pred)
    if gold is None:
        return false_positive(detail).with_values(gold, pred)
    if pred is None:
        return false_negative(detail).with_values(gold, pred)
    result = true_positive(detail) if equal else wrong_value(detail)
    return result.with_values(gold, pred)


def _normalize_pair(gold: Any, pred: Any, params: Mapping[str, Any], name: str | None = None) -> tuple[Any, Any]:
    normalizer = get_normalizer(name or str(params.get("normalizer", "free_string")))
    return normalizer(gold, params), normalizer(pred, params)


@register_matcher("identity")
def identity(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Compare enum or boolean values for equality after canonicalisation."""
    gold_value, pred_value = _normalize_pair(gold, pred, params)
    return _scalar_outcome(gold_value, pred_value, gold_value == pred_value)


@register_matcher("exact")
def exact(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Normalize both sides, then require exact equality."""
    gold_value, pred_value = _normalize_pair(gold, pred, params)
    return _scalar_outcome(gold_value, pred_value, gold_value == pred_value)


def _similarity(gold: str, pred: str) -> float:
    """Return order-insensitive similarity in [0, 1].

    token_sort, not token_set: token_set scores a subset as a perfect match, so "Acme"
    would match "Acme Industries" at 1.0 and every truncated answer would score as correct.
    """
    return float(fuzz.token_sort_ratio(gold, pred)) / 100.0


@register_matcher("entity_name")
def entity_name(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Accept a prediction whose similarity to gold reaches ``theta``.

    The strict-equality result is always attached alongside, because a fuzzy number
    reported on its own hides how much of the credit came from the threshold.
    """
    theta = float(params["theta"])
    gold_value, pred_value = _normalize_pair(gold, pred, params)
    strict = _scalar_outcome(gold_value, pred_value, gold_value == pred_value)
    if not isinstance(gold_value, str) or not isinstance(pred_value, str):
        return strict.with_alternates(strict=strict)
    score = _similarity(gold_value, pred_value)
    fuzzy = _scalar_outcome(gold_value, pred_value, score >= theta)
    detail = f"{_line(gold_value, pred_value)} similarity={score:.3f} theta={theta:.2f}"
    return MatchResult(
        tp=fuzzy.tp,
        fp=fuzzy.fp,
        fn=fuzzy.fn,
        tn=fuzzy.tn,
        correct=fuzzy.correct,
        detail=detail,
        gold_normalized=gold_value,
        pred_normalized=pred_value,
        alternates={"strict": strict},
        measure=score,
        threshold=theta,
    )


@register_matcher("free_text")
def free_text(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Fuzzy comparison over free-string normalization rather than entity-name rules."""
    merged = {**params, "normalizer": params.get("normalizer", "free_string")}
    return entity_name(gold, pred, merged)


def _as_items(value: Any, params: Mapping[str, Any], normalizer_name: str) -> list[Any]:
    """Normalize a list-valued answer item by item, dropping only true abstentions."""
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence | set | frozenset):
        value = [value]
    normalizer = get_normalizer(normalizer_name)
    items = [normalizer(item, params) for item in value]
    return [item for item in items if item is not None]


@register_matcher("set")
def set_match(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Compare two label sets, counting each label separately."""
    normalizer_name = str(params.get("normalizer", "enum"))
    gold_items = set(_as_items(gold, params, normalizer_name))
    pred_items = set(_as_items(pred, params, normalizer_name))
    detail = _line(gold_items, pred_items)
    if not gold_items and not pred_items:
        return true_negative(detail).with_values(gold_items, pred_items)
    return MatchResult(
        tp=len(gold_items & pred_items),
        fp=len(pred_items - gold_items),
        fn=len(gold_items - pred_items),
        correct=gold_items == pred_items,
        detail=detail,
        gold_normalized=gold_items,
        pred_normalized=pred_items,
    )


def _greedy_bipartite(gold_items: list[Any], pred_items: list[Any], theta: float) -> int:
    """Pair predicted items with gold items greedily by similarity, best pairs first.

    Greedy rather than optimal: the lists are short, and an optimal assignment would move
    credit between near-identical candidates without changing the reported totals.
    """
    scored: list[tuple[float, int, int]] = []
    for gold_index, gold_item in enumerate(gold_items):
        for pred_index, pred_item in enumerate(pred_items):
            if isinstance(gold_item, str) and isinstance(pred_item, str):
                score = _similarity(gold_item, pred_item)
            else:
                score = 1.0 if gold_item == pred_item else 0.0
            if score >= theta:
                scored.append((score, gold_index, pred_index))
    scored.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matched = 0
    for _score, gold_index, pred_index in scored:
        if gold_index in used_gold or pred_index in used_pred:
            continue
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        matched += 1
    return matched


@register_matcher("list")
def list_match(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Compare two lists of strings by greedy bipartite assignment on the pair matcher."""
    theta = float(params.get("theta", 1.0))
    normalizer_name = str(params.get("normalizer", "entity_name"))
    gold_items = _as_items(gold, params, normalizer_name)
    pred_items = _as_items(pred, params, normalizer_name)
    detail = _line(gold_items, pred_items)
    if not gold_items and not pred_items:
        return true_negative(detail).with_values(gold_items, pred_items)
    matched = _greedy_bipartite(gold_items, pred_items, theta)
    return MatchResult(
        tp=matched,
        fp=len(pred_items) - matched,
        fn=len(gold_items) - matched,
        correct=matched == len(gold_items) == len(pred_items),
        detail=f"{detail} matched={matched} theta={theta:.2f}",
        gold_normalized=gold_items,
        pred_normalized=pred_items,
        measure=_closest_pair_to_threshold(gold_items, pred_items, theta),
        threshold=theta,
    )


def _closest_pair_to_threshold(gold_items: list[Any], pred_items: list[Any], theta: float) -> float | None:
    """Return the best-match similarity, per gold item, that lies nearest the threshold.

    A list decision is only as close as its closest call: one signatory at 0.86 against a
    theta of 0.85 is the pairing worth a human look, however clean the others are.
    """
    best_per_gold: list[float] = []
    for gold_item in gold_items:
        scores = [
            _similarity(gold_item, pred_item) if isinstance(gold_item, str) and isinstance(pred_item, str) else 0.0
            for pred_item in pred_items
        ]
        best_per_gold.append(max(scores, default=0.0))
    if not best_per_gold:
        return None
    return min(best_per_gold, key=lambda score: abs(score - theta))


@register_matcher("numeric")
def numeric(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Compare quantities within a tolerance, after unit normalization.

    Differing units are a mismatch rather than a conversion: the package will not guess an
    exchange rate or a scale factor it was not told about.
    """
    tolerance = float(params["tolerance"])
    kind = str(params.get("tolerance_kind", "relative"))
    gold_value, pred_value = _normalize_pair(gold, pred, params, name=str(params.get("normalizer", "numeric")))
    if not isinstance(gold_value, Quantity) or not isinstance(pred_value, Quantity):
        return _scalar_outcome(gold_value, pred_value, gold_value == pred_value)
    if gold_value.unit != pred_value.unit:
        detail = f"{_line(gold_value, pred_value)} unit mismatch"
        return wrong_value(detail).with_values(gold_value, pred_value)
    difference = abs(gold_value.value - pred_value.value)
    allowed = tolerance * abs(gold_value.value) if kind == "relative" else tolerance
    detail = f"{_line(gold_value, pred_value)} delta={difference:g} allowed={allowed:g}"
    result = _scalar_outcome(gold_value, pred_value, difference <= allowed)
    if allowed > 0:
        distance: float | None = difference / allowed
    else:
        distance = 0.0 if difference == 0 else None
    return MatchResult(
        tp=result.tp,
        fp=result.fp,
        fn=result.fn,
        tn=result.tn,
        correct=result.correct,
        detail=detail,
        gold_normalized=gold_value,
        pred_normalized=pred_value,
        measure=distance,
        threshold=1.0,
    )


@register_matcher("date")
def date_match(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Compare dates at the task's declared granularity.

    A prediction coarser than gold is wrong unless the task sets ``allow_coarser``, because
    "2024" and "2024-06-15" are different claims.
    """
    granularity = Granularity(params.get("granularity", Granularity.DAY))
    allow_coarser = bool(params.get("allow_coarser", False))
    gold_value, pred_value = _normalize_pair(gold, pred, params, name=str(params.get("normalizer", "date")))
    if not isinstance(gold_value, PartialDate) or not isinstance(pred_value, PartialDate):
        return _scalar_outcome(gold_value, pred_value, gold_value == pred_value)
    compare_at = min(granularity.parts, gold_value.granularity.parts, pred_value.granularity.parts)
    if pred_value.granularity.parts < min(granularity.parts, gold_value.granularity.parts) and not allow_coarser:
        detail = f"{_line(gold_value, pred_value)} prediction coarser than {granularity.value}"
        return wrong_value(detail).with_values(gold_value, pred_value)
    level = {1: Granularity.YEAR, 2: Granularity.MONTH, 3: Granularity.DAY}[compare_at]
    equal = gold_value.truncated_to(level) == pred_value.truncated_to(level)
    detail = f"{_line(gold_value, pred_value)} compared at {level.value}"
    result = _scalar_outcome(gold_value, pred_value, equal)
    return MatchResult(
        tp=result.tp,
        fp=result.fp,
        fn=result.fn,
        tn=result.tn,
        correct=result.correct,
        detail=detail,
        gold_normalized=gold_value,
        pred_normalized=pred_value,
    )


@register_matcher("span")
def span_match(gold: Any, pred: Any, params: Mapping[str, Any]) -> MatchResult:
    """Compare character offsets, counting matched characters for token-level P/R/F1."""
    threshold = float(params.get("overlap_threshold", 0.5))
    gold_value, pred_value = _normalize_pair(gold, pred, params, name=str(params.get("normalizer", "span")))
    if not isinstance(gold_value, Span) or not isinstance(pred_value, Span):
        return _scalar_outcome(gold_value, pred_value, gold_value == pred_value)
    gold_chars = gold_value.tokens()
    pred_chars = pred_value.tokens()
    overlap = gold_chars & pred_chars
    ratio = len(overlap) / len(gold_chars | pred_chars)
    detail = f"{_line(gold_value, pred_value)} overlap={ratio:.3f} threshold={threshold:.2f}"
    return MatchResult(
        tp=len(overlap),
        fp=len(pred_chars - gold_chars),
        fn=len(gold_chars - pred_chars),
        correct=ratio >= threshold,
        detail=detail,
        gold_normalized=gold_value,
        pred_normalized=pred_value,
        measure=ratio,
        threshold=threshold,
    )
