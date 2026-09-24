"""Turning a quoted passage back into character offsets.

A span task is scored on offsets, but a model asked for offsets has to count characters, and
it miscounts: on the adversarial corpus it found the right sentence and placed it 84
characters early. Counting only gets harder as documents grow. So the program asks for the
passage verbatim and the package finds it in the text.

The search tolerates what legitimately differs between a quote and its source -- line breaks
introduced by extraction, typographic quotes and dashes, capitalisation, quotation marks and
ellipses wrapped around the excerpt -- and nothing more. A paraphrase is not a quote: it is
not found, and it scores as a wrong answer rather than being stretched to fit.
"""

from __future__ import annotations

import logging
import re

from doc_ai_kit.normalize import fold_unicode, strip_wrapping_quotes
from doc_ai_kit.values import Span

logger = logging.getLogger(__name__)

_ELLIPSIS = re.compile(r"^(?:\.\.\.|…)\s*|\s*(?:\.\.\.|…)$")


def _clean_quote(quote: str) -> str:
    """Strip what a model wraps around an excerpt without it being part of the text."""
    cleaned = strip_wrapping_quotes(quote.strip())
    previous = None
    while previous != cleaned:
        previous = cleaned
        cleaned = strip_wrapping_quotes(_ELLIPSIS.sub("", cleaned).strip())
    return cleaned


def _pattern(quote: str) -> re.Pattern[str] | None:
    """Build a pattern that matches the quote across any run of whitespace."""
    words = quote.split()
    if not words:
        return None
    return re.compile(r"\s+".join(re.escape(word) for word in words), re.IGNORECASE)


def _folded(text: str) -> tuple[str, list[int]]:
    """Fold typographic characters, keeping a map from each folded character to its source.

    Folding can change length ("…" becomes "..."), so offsets found in the folded text are
    mapped back through this index rather than used directly.
    """
    chars: list[str] = []
    origin: list[int] = []
    for position, character in enumerate(text):
        for folded_character in fold_unicode(character):
            chars.append(folded_character)
            origin.append(position)
    return "".join(chars), origin


def locate_quote(quote: str, text: str) -> Span | None:
    """Find a quoted passage in a document and return its offsets.

    :param quote: The passage as the model gave it
    :param text: The extracted text the model was shown
    :returns: The span of the first occurrence, with the document's own text attached, or
        None when the passage is not in the document
    """
    cleaned = _clean_quote(quote)
    if not cleaned:
        return None
    position = text.find(cleaned)
    if position >= 0:
        return Span(start=position, end=position + len(cleaned), text=text[position : position + len(cleaned)])

    pattern = _pattern(cleaned)
    if pattern is None:
        return None
    match = pattern.search(text)
    if match:
        return Span(start=match.start(), end=match.end(), text=match.group(0))

    folded_text, origin = _folded(text)
    folded_pattern = _pattern(fold_unicode(cleaned))
    if folded_pattern is None:
        return None
    match = folded_pattern.search(folded_text)
    if match is None or match.end() == 0:
        logger.debug("quote not found in the document: %r", cleaned[:80])
        return None
    start, end = origin[match.start()], origin[match.end() - 1] + 1
    return Span(start=start, end=end, text=text[start:end])
