"""Project-specific baseline inputs.

The harness records three baselines: zero-shot, hand-written few-shot, and BootstrapFewShot.
Only the few-shot demonstrations and any starting instruction are genuinely project-specific,
so only those live here.

Write the demonstrations by hand, from documents you have read. They are the bar a compiled
program has to clear -- a weak hand-written baseline makes any later number look like progress.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def hand_written_demos(examples: Sequence[Any]) -> Mapping[str, Sequence[Any]]:
    """Return hand-picked demonstrations per task group.

    :param examples: The training examples, already built from labels and cached text
    :returns: Group name to the demonstrations for that group's predictor

    Return an empty mapping to skip the few-shot baseline; the harness will say so loudly
    rather than quietly recording two baselines instead of three.
    """
    # Pick deliberately: cover the hard cases and the rare classes, not the first few rows.
    # return {"all": [examples[3], examples[11], examples[17]]}
    return {}


def starting_instructions() -> Mapping[str, str]:
    """Return an optional starting instruction per task group.

    Leave this empty unless the project has a genuine convention the questions cannot carry.
    A hand-tuned preamble here makes the zero-shot baseline measure your preamble rather than
    the questions.
    """
    return {}
