"""Normalizer tests driven by the adversarial fixture corpus.

Nothing downstream is trustworthy until these pass: every metric the package reports is
computed on normalized values, so a normalizer bug silently rewrites every number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from doc_ai_kit.hooks import get_normalizer, registered_names
from doc_ai_kit.values import PartialDate, Quantity, Span


def _load_fixture(fixtures_dir: Path) -> dict[str, Any]:
    return dict(yaml.safe_load((fixtures_dir / "normalizers.yaml").read_text(encoding="utf-8")))


def _cases(fixtures_dir: Path, kind: str) -> list[tuple[str, dict[str, Any], Any, Any]]:
    collected = []
    for name, block in _load_fixture(fixtures_dir).items():
        params = block.get("params", {})
        for entry in block.get(kind, []):
            collected.append((name, params, entry, None))
    return collected


def _comparable(value: Any) -> Any:
    """Reduce a normalized value to something a fixture file can express."""
    if isinstance(value, PartialDate):
        return {"value": value.value, "granularity": str(value.granularity)}
    if isinstance(value, Quantity):
        return {"value": value.value, "unit": value.unit}
    if isinstance(value, Span):
        return {"start": value.start, "end": value.end}
    return value


def _ids(cases: list[Any]) -> list[str]:
    return [f"{case[0]}:{case[2]}" for case in cases]


# these delegate to normalizers the fixture already covers, so they need no block of their own
DELEGATING_NORMALIZERS = {"passthrough", "controlled_code"}


def test_fixture_covers_every_registered_normalizer(fixtures_dir: Path) -> None:
    """A new normalizer must arrive with adversarial cases, not after them."""
    registered = set(registered_names()["normalizers"]) - DELEGATING_NORMALIZERS
    covered = set(_load_fixture(fixtures_dir))
    assert registered - covered == set(), f"normalizers with no fixture block: {sorted(registered - covered)}"


def test_single_values(fixtures_dir: Path) -> None:
    failures = []
    for name, block in _load_fixture(fixtures_dir).items():
        normalizer = get_normalizer(name)
        params = block.get("params", {})
        for case in block.get("cases", []):
            got = _comparable(normalizer(case["input"], params))
            want = case["expect"]
            if got != want:
                failures.append(f"{name}({case['input']!r}) -> {got!r}, expected {want!r}")
    assert not failures, "\n".join(failures)


def test_equivalent_forms_collapse(fixtures_dir: Path) -> None:
    """Surface variants of the same value must land on one canonical form."""
    failures = []
    for name, block in _load_fixture(fixtures_dir).items():
        normalizer = get_normalizer(name)
        params = block.get("params", {})
        for group in block.get("equivalent", []):
            results = [_comparable(normalizer(value, params)) for value in group]
            if any(result != results[0] for result in results):
                failures.append(f"{name}: {group} -> {results}")
    assert not failures, "\n".join(failures)


def test_distinct_values_stay_distinct(fixtures_dir: Path) -> None:
    """Canonicalisation must not collapse values that genuinely differ."""
    failures = []
    for name, block in _load_fixture(fixtures_dir).items():
        normalizer = get_normalizer(name)
        params = block.get("params", {})
        for group in block.get("distinct", []):
            results = [_comparable(normalizer(value, params)) for value in group]
            if len({str(result) for result in results}) != len(results):
                failures.append(f"{name}: {group} collapsed to {results}")
    assert not failures, "\n".join(failures)


def test_normalizers_are_pure(fixtures_dir: Path) -> None:
    """Calling a normalizer twice on the same input must give the same answer."""
    for name, block in _load_fixture(fixtures_dir).items():
        normalizer = get_normalizer(name)
        params = block.get("params", {})
        for case in block.get("cases", []):
            first = _comparable(normalizer(case["input"], params))
            second = _comparable(normalizer(case["input"], params))
            assert first == second, name


def test_normalization_is_idempotent(fixtures_dir: Path) -> None:
    """Normalizing an already-normalized value must be a no-op.

    Gold labels are often stored already canonical, so a normalizer that keeps changing its
    own output would score gold against a different value than it scores predictions.
    """
    failures = []
    for name, block in _load_fixture(fixtures_dir).items():
        normalizer = get_normalizer(name)
        params = block.get("params", {})
        for case in block.get("cases", []):
            once = normalizer(case["input"], params)
            twice = normalizer(once, params)
            if _comparable(once) != _comparable(twice):
                failures.append(f"{name}({case['input']!r}): {_comparable(once)!r} -> {_comparable(twice)!r}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "value,expected",
    [("unknown", "unknown"), ("Unknown", "Unknown")],
)
def test_enum_member_named_like_a_null_token_survives(value: str, expected: str) -> None:
    """An enum whose member looks like a null token must not be read as an abstention."""
    normalizer = get_normalizer("enum")
    params = {"enum": ["Unknown", "Known"]}
    assert normalizer(value, params) == "Unknown"


def test_uninterpretable_is_not_an_abstention() -> None:
    """A value the normalizer cannot read must stay non-null so it scores as wrong."""
    assert get_normalizer("boolean")("perhaps", {}) == "perhaps"
    assert get_normalizer("numeric")("a lot", {}) == "a lot"
    assert get_normalizer("date")("whenever", {}) == "whenever"
    assert get_normalizer("enum")("Ontario", {"enum": ["CA"]}) == "ontario"
