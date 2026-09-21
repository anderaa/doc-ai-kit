"""Confidence intervals and small-sample tables.

Wilson rather than normal-approximation intervals: at the sample sizes these projects work
with, and at accuracies near 1.0, the normal approximation produces intervals that run past
1.0 and understate the uncertainty exactly where it matters most.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# two-sided 95% normal quantile
Z95 = 1.959963984540054

# approximate 95% Wilson half-widths for recall near 0.8, used to tell a project what a
# split of a given size can and cannot measure
HALF_WIDTH_TABLE: tuple[tuple[int, float, str], ...] = (
    (5, 0.29, "presence check only"),
    (10, 0.22, "detecting total failure"),
    (30, 0.14, "coarse comparison"),
    (50, 0.11, "a number with a caveat"),
    (100, 0.08, "a number"),
)


def wilson_interval(successes: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion.

    :param successes: The number of successes observed
    :param total: The number of trials
    :param z: The normal quantile, defaulting to two-sided 95%
    :returns: The lower and upper bounds, clamped to [0, 1]
    """
    if total <= 0:
        return (0.0, 1.0)
    if successes < 0 or successes > total:
        raise ValueError(f"{successes} successes out of {total} trials is not a proportion")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def wilson_half_width(successes: int, total: int, z: float = Z95) -> float:
    """Return half the width of the Wilson interval, as a single uncertainty number."""
    low, high = wilson_interval(successes, total, z)
    return (high - low) / 2.0


def readable_at(support: int) -> str:
    """Describe what a split with this many examples of a class can support.

    :param support: The number of labeled examples of the class
    :returns: A short phrase naming the strongest claim the sample can carry
    """
    verdict = "presence check only"
    for threshold, _half_width, description in HALF_WIDTH_TABLE:
        if support >= threshold:
            verdict = description
    if support < HALF_WIDTH_TABLE[0][0]:
        return "not measurable"
    return verdict


def bootstrap_interval(
    values: Sequence[float],
    resamples: int = 2000,
    seed: int = 0,
    z: float = Z95,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for the mean of ``values``.

    Bootstrap rather than Wilson for the aggregate, because the aggregate is a weighted
    mean of several per-task metrics and is not a binomial proportion.

    :param values: The per-example scores
    :param resamples: How many bootstrap resamples to draw
    :param seed: Fixed so a report is reproducible from its inputs
    :param z: The normal quantile defining the coverage, defaulting to 95%
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    tail = (1.0 - _coverage(z)) / 2.0 * 100.0
    return (float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail)))


def _coverage(z: float) -> float:
    """Return the two-sided coverage implied by a normal quantile."""
    return math.erf(z / math.sqrt(2.0))


def precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    """Return precision, recall and F1 from outcome counts, with zero-safe denominators."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
