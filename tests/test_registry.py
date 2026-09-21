"""Registry tests: tasks.yaml must parse into typed objects or fail with the task id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from doc_harness.registry import (
    ExtractFuzzyTask,
    MulticlassTask,
    Registry,
    TaskSpecError,
    TaskType,
)
from doc_harness.values import PartialDate, Quantity, Span


@pytest.fixture
def all_types(fixtures_dir: Path) -> Registry:
    return Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")


def test_every_task_type_round_trips(all_types: Registry) -> None:
    assert len(all_types) == 9
    assert {str(task.type) for task in all_types} == set(TaskType)


def test_typed_objects_not_dicts(all_types: Registry) -> None:
    state = all_types.by_id("filing_state")
    assert isinstance(state, MulticlassTask)
    assert state.output.enum == ["AL", "AK", "AZ", "CA", "DE", "NY", "TX"]
    assert state.output.synonyms["California"] == "CA"
    assert state.weight == 2.0

    counterparty = all_types.by_id("counterparty")
    assert isinstance(counterparty, ExtractFuzzyTask)
    assert counterparty.match.theta == 0.90


def test_output_types(all_types: Registry) -> None:
    expected: dict[str, Any] = {
        "has_arbitration_clause": bool | None,
        "covered_products": list,
        "contract_number": str | None,
        "counterparty": str | None,
        "signatories": list[str] | None,
        "contract_value": Quantity | None,
        "effective_date": PartialDate | None,
        "governing_law_span": Span | None,
    }
    for task_id, want in expected.items():
        got = all_types.output_type(all_types.by_id(task_id))
        if want is list:
            # multilabel abstains with an empty list, so it is never optional
            assert str(got).startswith("list["), got
        else:
            assert got == want, (task_id, got)

    multiclass = all_types.output_type(all_types.by_id("filing_state"))
    assert "CA" in str(multiclass) and "Optional" in str(multiclass)


def test_groups_default_to_one_predictor(all_types: Registry) -> None:
    groups = all_types.groups()
    assert set(groups) == {"all", "spans"}
    assert [task.id for task in groups["spans"]] == ["governing_law_span"]


def test_params_flatten_match_and_output(all_types: Registry) -> None:
    params = all_types.by_id("filing_state").params
    assert params["matcher"] == "identity"
    assert params["normalizer"] == "enum"
    assert params["enum"][0] == "AL"
    assert params["synonyms"]["New York"] == "NY"
    assert params["nullable"] is True


def _load(raw: str) -> Registry:
    return Registry.from_mapping(yaml.safe_load(raw))


MALFORMED: list[tuple[str, str, str]] = [
    (
        "unknown type",
        """
        tasks:
          - id: filing_state
            type: multiclas
            question: Which state.
        """,
        "unknown type 'multiclas'",
    ),
    (
        "missing theta on fuzzy",
        """
        tasks:
          - id: counterparty
            type: extract_fuzzy
            question: Who.
            match: {matcher: entity_name}
        """,
        "counterparty",
    ),
    (
        "missing tolerance on numeric",
        """
        tasks:
          - id: value
            type: extract_numeric
            question: How much.
            match: {matcher: numeric}
        """,
        "value",
    ),
    (
        "empty enum",
        """
        tasks:
          - id: state
            type: multiclass
            question: Which state.
            output: {enum: []}
        """,
        "state",
    ),
    (
        "duplicate enum members",
        """
        tasks:
          - id: state
            type: multiclass
            question: Which state.
            output: {enum: [CA, CA]}
        """,
        "duplicate enum members",
    ),
    (
        "synonym to non-member",
        """
        tasks:
          - id: state
            type: multiclass
            question: Which state.
            output: {enum: [CA], synonyms: {Ontario: ONT}}
        """,
        "non-members",
    ),
    (
        "zero weight",
        """
        tasks:
          - id: state
            type: multiclass
            question: Which state.
            output: {enum: [CA]}
            weight: 0
        """,
        "state",
    ),
    (
        "unknown key",
        """
        tasks:
          - id: state
            type: multiclass
            question: Which state.
            output: {enum: [CA]}
            mtach: {matcher: identity}
        """,
        "state",
    ),
    (
        "missing id",
        """
        tasks:
          - type: multiclass
            question: Which state.
            output: {enum: [CA]}
        """,
        "missing or empty 'id'",
    ),
    (
        "duplicate ids",
        """
        tasks:
          - id: state
            type: binary
            question: One.
          - id: state
            type: binary
            question: Two.
        """,
        "duplicate id",
    ),
    (
        "theta out of range",
        """
        tasks:
          - id: counterparty
            type: extract_fuzzy
            question: Who.
            match: {matcher: entity_name, theta: 1.5}
        """,
        "counterparty",
    ),
    (
        "no tasks",
        "tasks: []",
        "declares no tasks",
    ),
    (
        "missing tasks key",
        "task: []",
        "top-level 'tasks' key",
    ),
]


@pytest.mark.parametrize("name,raw,expected", MALFORMED, ids=[case[0] for case in MALFORMED])
def test_malformed_fails_loudly(name: str, raw: str, expected: str) -> None:
    with pytest.raises(TaskSpecError) as excinfo:
        _load(raw)
    assert expected in str(excinfo.value), str(excinfo.value)


def test_unknown_task_id_lists_declared(all_types: Registry) -> None:
    with pytest.raises(TaskSpecError, match="unknown task id"):
        all_types.by_id("no_such_task")


def test_total_weight(all_types: Registry) -> None:
    assert all_types.total_weight() == pytest.approx(10.0)
