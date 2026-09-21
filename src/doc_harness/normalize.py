"""Normalizers: pure functions applied identically to gold and prediction.

A normalizer maps a raw value onto a canonical form, or to ``None`` when the value is an
abstention. It never decides whether two values match -- that is :mod:`doc_harness.match`.

A value that cannot be interpreted is *not* normalized to ``None``: an uninterpretable
answer is a wrong answer, not an abstention, and collapsing the two would turn false
positives into false negatives and quietly flatter recall.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from dateutil import parser as date_parser

from doc_harness.hooks import register_normalizer
from doc_harness.values import Granularity, PartialDate, Quantity, QuotedSpan, Span

# surface forms that mean "no answer"; a project can override via params["null_tokens"]
NULL_TOKENS = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "nil",
        "none",
        "not applicable",
        "not found",
        "not specified",
        "not stated",
        "null",
        "unspecified",
    }
)

# legal-entity suffixes collapsed away before comparing entity names
LEGAL_SUFFIXES = (
    "incorporated",
    "corporation",
    "company",
    "limited",
    "inc",
    "llc",
    "llp",
    "lllp",
    "pllc",
    "plc",
    "ltd",
    "corp",
    "gmbh",
    "ag",
    "nv",
    "bv",
    "sa",
    "spa",
    "srl",
    "pty",
    "lp",
    "pc",
    "co",
)

_LEADING_ARTICLES = ("the ", "a ", "an ")

# "l.l.c." or "n.v": letters joined by dots, collapsed to "llc" before punctuation is stripped;
# otherwise the dots become spaces and "l l c" matches no legal suffix at all. Needs two or
# more letters, so a lone initial such as "j." is left alone.
_DOTTED_ABBREVIATION = re.compile(r"\b(?:[a-z]\.){1,}[a-z]\b\.?")

# titles and credentials that never distinguish one person from another. Generational
# suffixes -- jr, sr, ii, iii -- are deliberately absent: they do distinguish people.
_HONORIFICS = frozenset({"mr", "mrs", "ms", "mx", "miss", "dr", "prof", "professor", "sir", "dame", "rev", "hon"})
_CREDENTIALS = frozenset({"phd", "md", "jd", "esq", "mba", "cpa", "dds", "llm", "msc", "bsc"})

_TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1", "affirmative", "present"})
_FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0", "negative", "absent"})

# multipliers accepted as a magnitude suffix on a number
_MAGNITUDES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "mm": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

# currency names as a model or a labeler writes them. Qualified names map to their own code.
_CURRENCY_NAMES = {
    "us dollar": "USD", "us dollars": "USD", "u.s. dollars": "USD", "us$": "USD", "usd": "USD",
    "canadian dollar": "CAD", "canadian dollars": "CAD", "cad": "CAD", "c$": "CAD",
    "australian dollar": "AUD", "australian dollars": "AUD", "aud": "AUD", "a$": "AUD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound sterling": "GBP", "pounds sterling": "GBP", "british pounds": "GBP", "gbp": "GBP",
    "yen": "JPY", "japanese yen": "JPY", "jpy": "JPY",
}  # fmt: skip
# a bare "dollars" says which currency family but not which member of it
_DOLLAR_CODES = frozenset({"USD", "CAD", "AUD", "NZD", "SGD", "HKD"})

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}  # fmt: skip

_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_MONTH_RE = re.compile(r"^(\d{4})-(\d{1,2})$")
_MONTH_NAME_YEAR_RE = re.compile(r"^([a-z]{3,9})\.?,?\s+(\d{4})$")
_YEAR_MONTH_NAME_RE = re.compile(r"^(\d{4})\s+([a-z]{3,9})$")
_NUMERIC_RE = re.compile(r"-?\d[\d,_\s]*\.?\d*")


def null_tokens(params: Mapping[str, Any]) -> frozenset[str]:
    """Return the set of surface forms treated as an abstention for this task."""
    override = params.get("null_tokens")
    return frozenset(str(token).strip().casefold() for token in override) if override else NULL_TOKENS


def fold_unicode(text: str) -> str:
    """Fold typographic punctuation and compatibility characters onto ASCII equivalents.

    Smart quotes and en dashes routinely differ between a PDF and a hand-typed label, and
    that difference is never a substantive disagreement.
    """
    folded = unicodedata.normalize("NFKC", text)
    replacements = {
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "“": '"', "”": '"', "„": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
        " ": " ", " ": " ", " ": " ", "…": "...",
    }  # fmt: skip
    for source, target in replacements.items():
        folded = folded.replace(source, target)
    return folded


def strip_wrapping_quotes(text: str) -> str:
    """Remove a matched pair of quotes around the whole value.

    Models routinely return an extracted string already quoted, because the document quoted
    it or because quoting looks like the careful thing to do. The quotes are a formatting
    artifact, never part of the answer, and leaving them on turns correct extractions into
    both a false positive and a false negative.
    """
    stripped = text.strip()
    for opening, closing in (('"', '"'), ("'", "'"), ("`", "`")):
        if len(stripped) > 2 and stripped.startswith(opening) and stripped.endswith(closing):
            return stripped[1:-1].strip()
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse every run of whitespace to a single space and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def is_null(value: Any, params: Mapping[str, Any] | None = None) -> bool:
    """Return whether a raw value is an abstention: None, or a null word such as "N/A"."""
    return _is_null(value, params or {})


def _is_null(value: Any, params: Mapping[str, Any]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return collapse_whitespace(fold_unicode(value)).casefold() in null_tokens(params)
    return False


@register_normalizer("passthrough")
def passthrough(value: Any, params: Mapping[str, Any]) -> Any:
    """Return the value unchanged, for tasks that are already canonical."""
    return None if value is None else value


@register_normalizer("free_string")
def free_string(value: Any, params: Mapping[str, Any]) -> str | None:
    """Canonicalize a free-text string: unicode folded, whitespace collapsed, casefolded."""
    if _is_null(value, params):
        return None
    folded = strip_wrapping_quotes(collapse_whitespace(fold_unicode(_as_text(value))))
    return collapse_whitespace(folded).casefold()


@register_normalizer("boolean")
def boolean(value: Any, params: Mapping[str, Any]) -> bool | str | None:
    """Canonicalize a yes/no answer.

    An unrecognised value comes back as a canonicalized string rather than ``None``, so it
    scores as a wrong answer rather than an abstention.
    """
    if _is_null(value, params):
        return None
    if isinstance(value, bool):
        return value
    text = collapse_whitespace(fold_unicode(_as_text(value))).casefold()
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return text


@register_normalizer("enum")
def enum(value: Any, params: Mapping[str, Any]) -> str | None:
    """Map a value onto a declared enum member via synonyms and case folding.

    A value outside the enum is returned canonicalized but unmapped, which cannot equal any
    member and therefore scores as a wrong answer.
    """
    members: list[str] = list(params.get("enum") or [])
    synonyms: Mapping[str, str] = params.get("synonyms") or {}
    if value is None:
        return None
    text = collapse_whitespace(fold_unicode(_as_text(value)))
    # membership is checked before null tokens so an enum member named "none" survives
    for member in members:
        if text == member:
            return member
    folded = text.casefold()
    for surface, target in synonyms.items():
        if folded == collapse_whitespace(fold_unicode(surface)).casefold():
            return target
    # punctuation is squashed too, so "N.Y." reaches the member "NY" without a synonym entry
    squashed = re.sub(r"[\s_\-.]+", "", folded)
    for member in members:
        if folded == member.casefold() or squashed == re.sub(r"[\s_\-.]+", "", member.casefold()):
            return member
    if folded in null_tokens(params):
        return None
    return folded


@register_normalizer("controlled_code")
def controlled_code(value: Any, params: Mapping[str, Any]) -> str | None:
    """Map a value onto a controlled vocabulary such as USPS state codes.

    Identical in shape to :func:`enum`; kept as its own name so a project can point a task
    at a vocabulary supplied through ``params["vocabulary"]`` rather than inline members.
    """
    vocabulary: Mapping[str, str] = params.get("vocabulary") or {}
    merged = dict(params)
    merged.setdefault("enum", sorted(set(vocabulary.values())) or list(params.get("enum") or []))
    merged["synonyms"] = {**vocabulary, **(params.get("synonyms") or {})}
    return enum(value, merged)


def _strip_legal_suffixes(text: str) -> str:
    """Repeatedly drop trailing legal-entity suffixes such as "inc" or "llc"."""
    changed = True
    while changed:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            if text.endswith(" " + suffix):
                text = text[: -len(suffix) - 1].strip()
                changed = True
    return text


@register_normalizer("entity_name")
def entity_name(value: Any, params: Mapping[str, Any]) -> str | None:
    """Canonicalize an organisation or person name for comparison.

    Casing, punctuation, whitespace, leading articles and legal suffixes are all collapsed,
    because none of them distinguish two references to the same entity.
    """
    if _is_null(value, params):
        return None
    text = collapse_whitespace(fold_unicode(_as_text(value))).casefold()
    text = text.replace("&", " and ")
    text = _DOTTED_ABBREVIATION.sub(lambda match: match.group(0).replace(".", ""), text)
    text = re.sub(r"[.,;:()\[\]{}'\"`/\\]", " ", text)
    text = re.sub(r"\s*-\s*", " ", text)
    text = collapse_whitespace(text)
    for article in _LEADING_ARTICLES:
        if text.startswith(article):
            text = text[len(article) :]
            break
    text = _strip_legal_suffixes(text)
    return collapse_whitespace(text)


@register_normalizer("person_name")
def person_name(value: Any, params: Mapping[str, Any]) -> str | None:
    """Canonicalize a person's name for comparison.

    Separate from :func:`entity_name` on purpose. Legal suffixes are an organisation's, and
    stripping "co" or "group" from a person's surname damages it; titles and credentials are
    a person's, and stripping a leading "dr" from an organisation ("Dr Pepper") damages that.
    Only what never distinguishes two people is removed: honorifics and credentials.
    """
    if _is_null(value, params):
        return None
    text = collapse_whitespace(fold_unicode(_as_text(value))).casefold()
    text = _DOTTED_ABBREVIATION.sub(lambda match: match.group(0).replace(".", ""), text)
    text = re.sub(r"[.,;:()\[\]{}'\"`/\\]", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    tokens = collapse_whitespace(text).split(" ")
    while tokens and tokens[0] in _HONORIFICS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _CREDENTIALS:
        tokens = tokens[:-1]
    return " ".join(tokens) or None


def _parse_number(text: str) -> tuple[float, float] | None:
    """Return the numeric value and any magnitude multiplier found in the text."""
    match = _NUMERIC_RE.search(text)
    if match is None:
        return None
    digits = re.sub(r"[,\s_]", "", match.group(0))
    try:
        number = float(digits)
    except ValueError:
        return None
    remainder = text[match.end() :].strip()
    multiplier = 1.0
    for token, scale in sorted(_MAGNITUDES.items(), key=lambda item: -len(item[0])):
        if remainder.startswith(token) and (len(remainder) == len(token) or not remainder[len(token)].isalpha()):
            multiplier = scale
            break
    return number, multiplier


@register_normalizer("numeric")
def numeric(value: Any, params: Mapping[str, Any]) -> Quantity | str | None:
    """Parse a number with an optional unit, expanding magnitude suffixes and currency symbols.

    An unparseable value comes back as a canonicalized string so it scores as wrong rather
    than as an abstention.
    """
    if _is_null(value, params):
        return None
    if isinstance(value, Quantity):
        # a typed model output still needs its unit canonicalized: "usd" and "USD" are the
        # same currency, and returning the object untouched scored that as a unit mismatch
        return Quantity(value=value.value, unit=_canonical_unit(value.unit, params))
    if isinstance(value, bool):
        return collapse_whitespace(_as_text(value)).casefold()
    if isinstance(value, int | float):
        return Quantity(value=float(value), unit=params.get("unit"))
    if isinstance(value, Mapping):
        raw_value = value.get("value")
        if raw_value is None:
            return None
        parsed = _parse_number(str(raw_value))
        if parsed is None:
            return collapse_whitespace(_as_text(raw_value)).casefold()
        number, multiplier = parsed
        return Quantity(value=number * multiplier, unit=_canonical_unit(value.get("unit"), params))
    text = collapse_whitespace(fold_unicode(_as_text(value)))
    unit: str | None = None
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            unit = code
            text = text.replace(symbol, " ")
            break
    if "%" in text:
        unit = "percent"
        text = text.replace("%", " ")
    text = collapse_whitespace(text).casefold()
    negative = text.startswith("(") and text.endswith(")")
    parsed = _parse_number(text.strip("()"))
    if parsed is None:
        return text
    number, multiplier = parsed
    amount = number * multiplier * (-1.0 if negative else 1.0)
    trailing = re.sub(r"[\d.,\s_()%$-]", "", text)
    for token in _MAGNITUDES:
        if trailing.startswith(token):
            trailing = trailing[len(token) :]
            break
    if unit is None and trailing:
        unit = trailing
    return Quantity(value=amount, unit=_canonical_unit(unit, params))


def _canonical_unit(unit: Any, params: Mapping[str, Any]) -> str | None:
    """Fold a unit onto a canonical code: task aliases first, then currency names and symbols.

    A bare "dollars" resolves to the task's declared unit when that is a dollar currency,
    and to USD otherwise -- the same convention the "$" symbol already follows. That is an
    assumption, and a project whose documents mix dollar currencies should declare
    unit_aliases rather than rely on it.
    """
    declared = params.get("unit")
    if unit is None:
        return declared
    text = collapse_whitespace(fold_unicode(_as_text(unit))).casefold()
    if not text:
        return declared
    aliases: Mapping[str, str] = params.get("unit_aliases") or {}
    for surface, target in aliases.items():
        if text == collapse_whitespace(fold_unicode(surface)).casefold():
            return target
    if text in _CURRENCY_NAMES:
        return _CURRENCY_NAMES[text]
    if text in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[text]
    if text in {"dollar", "dollars"}:
        return declared if declared in _DOLLAR_CODES else "USD"
    return text


def _partial_date(year: int, month: int | None = None, day: int | None = None) -> PartialDate:
    if day is not None and month is not None:
        return PartialDate(value=f"{year:04d}-{month:02d}-{day:02d}", granularity=Granularity.DAY)
    if month is not None:
        return PartialDate(value=f"{year:04d}-{month:02d}", granularity=Granularity.MONTH)
    return PartialDate(value=f"{year:04d}", granularity=Granularity.YEAR)


@register_normalizer("date")
def date(value: Any, params: Mapping[str, Any]) -> PartialDate | str | None:
    """Parse a date to ISO 8601, preserving how precise the original was.

    Granularity is preserved rather than defaulted, because "2024" and "2024-06-15" are
    different claims and whether the coarser one matches the finer is a per-task decision.
    """
    if _is_null(value, params):
        return None
    if isinstance(value, PartialDate):
        return value
    if isinstance(value, datetime):
        return _partial_date(value.year, value.month, value.day)
    if isinstance(value, Mapping) and "value" in value:
        return date(value["value"], params)
    text = collapse_whitespace(fold_unicode(_as_text(value))).casefold().replace(",", " ")
    text = collapse_whitespace(text)
    if match := _YEAR_RE.match(text):
        return _partial_date(int(match.group(1)))
    if match := _YEAR_MONTH_RE.match(text):
        month = int(match.group(2))
        if 1 <= month <= 12:
            return _partial_date(int(match.group(1)), month)
    if match := _MONTH_NAME_YEAR_RE.match(text):
        month_number = _MONTHS.get(match.group(1)[:3])
        if month_number is not None:
            return _partial_date(int(match.group(2)), month_number)
    if match := _YEAR_MONTH_NAME_RE.match(text):
        month_number = _MONTHS.get(match.group(2)[:3])
        if month_number is not None:
            return _partial_date(int(match.group(1)), month_number)
    try:
        parsed = date_parser.parse(text, fuzzy=False, dayfirst=bool(params.get("dayfirst", False)))
    except (ValueError, OverflowError):
        return text
    return _partial_date(parsed.year, parsed.month, parsed.day)


@register_normalizer("span")
def span(value: Any, params: Mapping[str, Any]) -> Span | str | None:
    """Coerce character offsets into a :class:`Span`."""
    if value is None:
        return None
    if isinstance(value, Span):
        return value if value.end > value.start else None
    if isinstance(value, QuotedSpan):
        # a gold span carried as its text for demonstrations; scored on its offsets
        return Span(start=value.start, end=value.end, text=str(value)) if value.end > value.start else None
    # checked after the quoted-span case, so a gold passage whose text is "N/A" stays a passage
    if isinstance(value, str) and _is_null(value, params):
        return None
    if isinstance(value, Mapping) and "start" in value and "end" in value:
        start, end = int(value["start"]), int(value["end"])
    elif isinstance(value, list | tuple) and len(value) == 2:
        start, end = int(value[0]), int(value[1])
    else:
        return collapse_whitespace(_as_text(value)).casefold()
    if end <= start or start < 0:
        return None
    return Span(start=start, end=end)
