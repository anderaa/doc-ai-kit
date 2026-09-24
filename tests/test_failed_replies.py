"""A reply that cannot be read is a failure, never an abstention.

Before this, DSPy filled any field a reply left out with None -- every package task is
nullable -- and dspy.Evaluate replaced a program that raised with an empty prediction. Both
read as the model declining to answer. A failure scored that way earns credit wherever the
gold is null, drags recall down elsewhere, and says nothing about what happened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy
import pytest
from conftest import StubProvider, document_for, real_lm
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

from doc_ai_kit.adapters import (
    StrictChatAdapter,
    StrictJSONAdapter,
    chat_fields_present,
    json_fields_present,
    strict_version_of,
)
from doc_ai_kit.config import Config
from doc_ai_kit.dataset import LabelRecord, build_examples
from doc_ai_kit.evaluate import (
    FAILED_REPLY,
    EvaluationError,
    FailedReply,
    load_predictions,
    run_program,
    score_split,
    write_run,
)
from doc_ai_kit.metric import build_metric
from doc_ai_kit.program import build_program
from doc_ai_kit.registry import Registry
from doc_ai_kit.report import run_holdout

GOOD = {"flag": "true", "state": "CA", "number": "A-1"}
GARBLED = {"nonsense": "cut off mid-answ"}


def _examples(n: int, labels: dict[str, Any] | None = None) -> list[Any]:
    records = [
        LabelRecord(doc_id=f"d{i}", labels=labels or {"flag": True, "state": "CA", "number": "A-1"}) for i in range(n)
    ]
    return build_examples(records, {r.doc_id: document_for(r.doc_id) for r in records})


# --- the parser -----------------------------------------------------------------------------


def test_a_reply_missing_a_nullable_field_raises(toy_registry: Registry) -> None:
    """The root cause: DSPy would have returned null for all three fields here."""
    with dspy.context(lm=DummyLM({"[doc=d0]": GARBLED})), pytest.raises(AdapterParseError):
        build_program(toy_registry)(document=document_for("d0"))


def test_a_reply_cut_off_partway_raises_instead_of_nulling_the_rest(toy_registry: Registry) -> None:
    """Truncated after the first field: the other two must not quietly become null."""
    with dspy.context(lm=DummyLM({"[doc=d0]": {"flag": "true"}})), pytest.raises(AdapterParseError):
        build_program(toy_registry)(document=document_for("d0"))


def test_an_explicit_null_is_still_an_answer(toy_registry: Registry) -> None:
    """Strictness is about presence, not about null: a field present and null is fine."""
    with dspy.context(lm=DummyLM({"[doc=d0]": {"flag": "true", "state": "null", "number": "null"}})):
        prediction = build_program(toy_registry)(document=document_for("d0"))
    assert prediction.flag is True
    assert prediction.state is None


def test_presence_is_read_from_the_reply_itself() -> None:
    chat = "[[ ## flag ## ]]\ntrue\n\n[[ ## state ## ]]\nNone\n\n[[ ## completed ## ]]"
    assert chat_fields_present(chat) == {"flag", "state", "completed"}
    assert json_fields_present('{"flag": true, "state": null}') == {"flag", "state"}
    assert json_fields_present('Here you go: {"flag": true} -- done') == {"flag"}
    assert json_fields_present("not json at all") == set()


def test_the_chat_fallback_stays_strict() -> None:
    """DSPy's stock fallback to JSON would quietly bring the null-filling back."""
    assert isinstance(StrictChatAdapter()._make_json_adapter_fallback(), StrictJSONAdapter)


@pytest.mark.parametrize(
    "configured,expected",
    [
        (None, StrictChatAdapter),
        (dspy.ChatAdapter(), StrictChatAdapter),
        (dspy.JSONAdapter(), StrictJSONAdapter),
        (StrictJSONAdapter(), StrictJSONAdapter),
    ],
)
def test_strictness_follows_the_configured_adapter(configured: Any, expected: type) -> None:
    assert isinstance(strict_version_of(configured), expected)


# --- scoring --------------------------------------------------------------------------------


def test_a_failed_reply_never_earns_credit_on_any_task(fixtures_dir: Path) -> None:
    """Scored as wrong everywhere -- including where the gold is null, which an abstention gets right."""
    registry = Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")
    metric = build_metric(registry)
    all_null = {task.id: ([] if str(task.type) == "multilabel" else None) for task in registry}
    failed = FailedReply(registry.ids, "AdapterParseError: truncated", attempts=3)
    score = metric.score_example({"doc_id": "d0", **all_null}, failed)
    assert score.aggregate == 0.0
    wrongly_credited = [task_id for task_id, result in score.results.items() if result.correct or result.tn]
    assert not wrongly_credited, f"a failure earned credit on: {wrongly_credited}"


def test_the_sentinel_is_neither_null_nor_matchable(fixtures_dir: Path) -> None:
    """If any normalizer read it as null, a failure would score as an abstention again."""
    from doc_ai_kit.evaluate import is_abstention
    from doc_ai_kit.hooks import get_normalizer

    registry = Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")
    for task in registry:
        normalized = get_normalizer(str(task.params["normalizer"]))(FAILED_REPLY, task.params)
        assert not is_abstention(normalized), f"{task.id} reads the failure sentinel as null"


# --- evaluation -----------------------------------------------------------------------------


def test_a_persistent_failure_stops_the_run_by_default(toy_registry: Registry) -> None:
    """Zero tolerance by default: numbers with an unreadable reply in them are not reported."""
    examples = _examples(4)
    answers = {f"[doc=d{i}]": GOOD for i in range(3)} | {"[doc=d3]": GARBLED}
    with dspy.context(lm=DummyLM(answers)), pytest.raises(EvaluationError) as refusal:
        run_program(build_program(toy_registry), examples, build_metric(toy_registry), num_threads=1)
    assert "d3" in str(refusal.value)
    assert "max_tokens" in str(refusal.value)


def test_below_the_threshold_a_failure_is_scored_wrong_and_listed(tmp_path: Path, toy_registry: Registry) -> None:
    """The toy case that motivated this: 1.00 would have read as 0.75 with no explanation."""
    examples = _examples(4)
    answers = {f"[doc=d{i}]": GOOD for i in range(3)} | {"[doc=d3]": GARBLED}
    metric = build_metric(toy_registry)
    with dspy.context(lm=DummyLM(answers)):
        preds = run_program(build_program(toy_registry), examples, metric, num_threads=1, max_failure_rate=0.5)
    assert isinstance(preds[3], FailedReply)
    assert preds[3].attempts == 3

    result = score_split(toy_registry, metric, examples, preds, support_floor=1, measurable_floor=1)
    assert result.aggregate == pytest.approx(0.75)
    assert [f["doc_id"] for f in result.failures] == ["d3"]
    assert result.metadata["n_failed_replies"] == 1
    # the failure is not an abstention, so it cannot inflate the rate the production gate uses
    assert result.tasks["state"].predicted_null_rate == 0.0

    write_run(tmp_path, toy_registry, result, examples, preds)
    payload = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["failures"][0]["doc_id"] == "d3"
    assert "Failed replies (1)" in (tmp_path / "failures.md").read_text(encoding="utf-8")


def test_failures_survive_a_free_rescore(tmp_path: Path, toy_registry: Registry) -> None:
    examples = _examples(2)
    metric = build_metric(toy_registry)
    preds = [dict(GOOD), FailedReply(toy_registry.ids, "AdapterParseError: truncated", attempts=3)]
    result = score_split(toy_registry, metric, examples, preds, support_floor=1, measurable_floor=1)
    write_run(tmp_path, toy_registry, result, examples, preds)

    saved = load_predictions(tmp_path / "predictions.jsonl")
    assert isinstance(saved["d1"], FailedReply)
    rescored = score_split(toy_registry, metric, examples, [saved["d0"], saved["d1"]], support_floor=1)
    assert rescored.failures == result.failures
    assert rescored.aggregate == pytest.approx(result.aggregate)


def test_a_flaky_reply_recovers_on_a_fresh_retry(
    toy_registry: Registry, isolated_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through dspy.Evaluate and the real cache: the first reply is cut off, the retry is whole."""
    import litellm

    probe = StubProvider(bad=10_000)
    monkeypatch.setattr(litellm, "completion", probe)
    with dspy.context(lm=real_lm()), pytest.raises(Exception):  # noqa: B017
        build_program(toy_registry)(document=document_for("probe"))
    first_attempt_calls = probe.calls

    stub = StubProvider(bad=first_attempt_calls)
    monkeypatch.setattr(litellm, "completion", stub)
    examples = _examples(1)
    with dspy.context(lm=real_lm()):
        preds = run_program(build_program(toy_registry), examples, build_metric(toy_registry), num_threads=1)
    assert not isinstance(preds[0], FailedReply)
    assert preds[0].state == "CA"


def test_a_refused_holdout_does_not_spend_the_holdout(tmp_path: Path, toy_registry: Registry) -> None:
    """If too many replies fail, no number exists, so the one-shot lock must not be taken."""
    examples = _examples(2)
    config = Config.from_mapping({"models": {"task": "fake/scripted"}, "optimization": {"num_threads": 1}})
    with dspy.context(lm=DummyLM({"[doc=d0]": GARBLED, "[doc=d1]": GARBLED})), pytest.raises(EvaluationError):
        run_holdout(
            toy_registry,
            config,
            build_metric(toy_registry),
            build_program(toy_registry),
            examples,
            tmp_path,
            1.0,
            {},
            opened_by="test",
        )
    assert not (tmp_path / "runs" / "holdout" / ".lock").exists()
