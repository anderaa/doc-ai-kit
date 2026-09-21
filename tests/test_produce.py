"""Production run, checkpointing, QA gates and review triage."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from conftest import StubProvider, document_for, real_lm, scripted_lm

from doc_harness.config import Config
from doc_harness.produce import (
    DocumentOutcome,
    ProductionResult,
    produce,
    run_qa,
    triage,
    write_outputs,
    write_qa_report,
)
from doc_harness.program import build_program
from doc_harness.registry import Registry

CORPUS = 12


def _config(**overrides: Any) -> Config:
    payload: dict[str, Any] = {
        "models": {"task": "fake/scripted", "reflection": "fake/scripted"},
        "optimization": {"num_threads": 1},
        "splits": {"support_floor": 2, "measurable_floor": 2},
        "production": {"max_retries": 1, "spot_check_sample": 3},
    }
    for key, value in overrides.items():
        payload.setdefault(key, {})
        payload[key].update(value)
    return Config.from_mapping(payload)


@pytest.fixture
def texts() -> dict[str, str]:
    return {f"p{index:02d}": document_for(f"p{index:02d}") for index in range(CORPUS)}


@pytest.fixture
def answers(texts: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {
        doc_id: {"flag": "true", "state": "CA" if index % 2 else "NY", "number": f"A-{index}"}
        for index, doc_id in enumerate(sorted(texts))
    }


def test_production_covers_the_corpus_and_checkpoints(
    tmp_path: Path, toy_registry: Registry, texts: dict[str, str], answers: Any
) -> None:
    program = build_program(toy_registry)
    with scripted_lm(answers):
        outcomes = produce(toy_registry, _config(), program, texts, tmp_path)
    assert len(outcomes) == CORPUS
    assert all(outcome.ok for outcome in outcomes)
    raw_dir = tmp_path / "runs" / "production" / "raw"
    assert len(list(raw_dir.glob("*.json"))) == CORPUS
    stored = json.loads((raw_dir / "p00.json").read_text(encoding="utf-8"))
    assert stored["doc_id"] == "p00"
    assert stored["values"]["state"] == "NY"


def test_production_resumes_from_checkpoints(
    tmp_path: Path, toy_registry: Registry, texts: dict[str, str], answers: Any
) -> None:
    """A resumed run must not pay for inference it already paid for."""
    program = build_program(toy_registry)
    with scripted_lm(answers):
        produce(toy_registry, _config(), program, texts, tmp_path)
    marker = tmp_path / "runs" / "production" / "raw" / "p00.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["values"]["number"] = "FROM-CHECKPOINT"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    # no LM configured at all: a resumed run must not need one
    outcomes = produce(toy_registry, _config(), program, texts, tmp_path)
    assert next(o for o in outcomes if o.doc_id == "p00").values["number"] == "FROM-CHECKPOINT"

    with scripted_lm(answers):
        fresh = produce(toy_registry, _config(), program, texts, tmp_path, resume=False)
    assert next(o for o in fresh if o.doc_id == "p00").values["number"] == "A-0"


def test_failures_are_recorded_never_dropped(tmp_path: Path, toy_registry: Registry, texts: dict[str, str]) -> None:
    class Exploding:
        def __call__(self, **kwargs: Any) -> Any:
            raise RuntimeError("provider said no")

    outcomes = produce(toy_registry, _config(), Exploding(), texts, tmp_path)
    assert len(outcomes) == CORPUS
    assert all(not outcome.ok for outcome in outcomes)
    assert all("provider said no" in outcome.error for outcome in outcomes)
    # retried before giving up, and the attempt count is kept
    assert outcomes[0].attempts == 2
    # a failed document is checkpointed too, so the failure survives a crash
    assert (tmp_path / "runs" / "production" / "raw" / "p00.json").exists()


def test_failed_checkpoints_are_retried(
    tmp_path: Path, toy_registry: Registry, texts: dict[str, str], answers: Any
) -> None:
    class Exploding:
        def __call__(self, **kwargs: Any) -> Any:
            raise RuntimeError("transient")

    produce(toy_registry, _config(), Exploding(), texts, tmp_path)
    with scripted_lm(answers):
        outcomes = produce(toy_registry, _config(), build_program(toy_registry), texts, tmp_path)
    assert all(outcome.ok for outcome in outcomes)


def test_outputs_file_covers_failures_too(tmp_path: Path) -> None:
    outcomes = [
        DocumentOutcome(doc_id="a", values={"flag": True}),
        DocumentOutcome(doc_id="b", error="boom"),
    ]
    path = tmp_path / "outputs.jsonl"
    write_outputs(path, outcomes)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["doc_id"] for row in rows] == ["a", "b"]
    assert rows[1]["ok"] is False and rows[1]["error"] == "boom"


def test_coverage_gate_fails_on_a_short_run(toy_registry: Registry) -> None:
    outcomes = [DocumentOutcome(doc_id="a", values={})]
    checks = {check.name: check for check in run_qa(toy_registry, _config(), outcomes, expected_documents=10)}
    assert checks["coverage"].passed is False
    assert "1 outputs for 10 documents" in checks["coverage"].detail


def test_schema_gate_uses_the_configured_ceiling(toy_registry: Registry) -> None:
    outcomes = [DocumentOutcome(doc_id=f"d{i}", values={}) for i in range(99)]
    outcomes.append(DocumentOutcome(doc_id="bad", error="unparseable"))
    checks = {check.name: check for check in run_qa(toy_registry, _config(), outcomes, expected_documents=100)}
    # 1% failures against a 0.5% ceiling
    assert checks["schema"].passed is False

    lenient = _config(production={"max_schema_failure_rate": 0.02})
    checks = {check.name: check for check in run_qa(toy_registry, lenient, outcomes, expected_documents=100)}
    assert checks["schema"].passed is True


def test_class_distribution_gate_catches_a_shift(toy_registry: Registry) -> None:
    reference = [{"flag": True, "state": "CA"} for _ in range(10)]
    produced = [DocumentOutcome(doc_id=f"d{i}", values={"flag": True, "state": "NY"}) for i in range(10)]
    checks = {
        check.name: check
        for check in run_qa(toy_registry, _config(), produced, expected_documents=10, reference_rows=reference)
    }
    assert checks["class distribution"].passed is False
    assert "state/NY" in checks["class distribution"].detail


def test_class_distribution_gate_passes_when_stable(toy_registry: Registry) -> None:
    reference = [{"flag": True, "state": "CA"} for _ in range(10)]
    produced = [DocumentOutcome(doc_id=f"d{i}", values={"flag": True, "state": "CA"}) for i in range(10)]
    checks = {
        check.name: check
        for check in run_qa(toy_registry, _config(), produced, expected_documents=10, reference_rows=reference)
    }
    assert checks["class distribution"].passed is True


def test_null_rate_gate_catches_a_program_that_gave_up(toy_registry: Registry, texts: dict[str, str]) -> None:
    # the champion answered every task on validation, so nothing here should abstain
    reference_null_rates = {"flag": 0.0, "state": 0.0, "number": 0.0}
    abstaining = [
        DocumentOutcome(doc_id=doc_id, values={"flag": None, "state": None, "number": None}) for doc_id in texts
    ]
    checks = {
        check.name: check
        for check in run_qa(
            toy_registry,
            _config(),
            abstaining,
            expected_documents=CORPUS,
            reference_null_rates=reference_null_rates,
        )
    }
    assert checks["null rates"].passed is False
    assert "state" in checks["null rates"].detail


def test_triage_keeps_the_random_slice(tmp_path: Path, toy_registry: Registry) -> None:
    manifest = tmp_path / "extraction_manifest.csv"
    manifest.write_text(
        "doc_id,ocr,ocr_needed,truncated\n" "d0,True,True,False\n" "d1,False,False,True\n" "d2,False,False,False\n",
        encoding="utf-8",
    )
    outcomes = [
        DocumentOutcome(doc_id="d0", values={"flag": True, "state": "CA", "number": "A"}),
        DocumentOutcome(doc_id="d1", values={"flag": True, "state": "CA", "number": "A"}),
        DocumentOutcome(doc_id="d2", values={"flag": None, "state": None, "number": None}),
        DocumentOutcome(doc_id="d3", error="boom"),
    ]
    routes = triage(toy_registry, _config(), outcomes, manifest_path=manifest, seed=1)
    assert routes["ocr"] == ["d0"]
    assert routes["truncated"] == ["d1"]
    assert routes["failed"] == ["d3"]
    assert "d2" in routes["nulls_on_answered_tasks"]
    assert "d2" in routes["low_confidence"]
    assert routes["random_slice"], "the random slice is what keeps the quality estimate unbiased"


def test_triage_is_reproducible(toy_registry: Registry) -> None:
    outcomes = [
        DocumentOutcome(doc_id=f"d{i:02d}", values={"flag": True, "state": "CA", "number": "A"}) for i in range(20)
    ]
    first = triage(toy_registry, _config(), outcomes, seed=5)
    second = triage(toy_registry, _config(), outcomes, seed=5)
    assert first == second


def test_qa_report_says_when_gates_failed(tmp_path: Path, toy_registry: Registry) -> None:
    outcomes = [DocumentOutcome(doc_id="a", values={}), DocumentOutcome(doc_id="b", error="boom")]
    checks = run_qa(toy_registry, _config(), outcomes, expected_documents=10)
    result = ProductionResult(outcomes=outcomes, checks=checks, triage=triage(toy_registry, _config(), outcomes))
    path = tmp_path / "qa_report.md"
    write_qa_report(path, toy_registry, result, expected_documents=10)
    text = path.read_text(encoding="utf-8")
    assert "**FAIL**" in text
    assert "do not hand these results over yet" in text
    assert "random slice is not optional" in text
    assert "| b | 1 | boom |" in text


def test_production_runs_documents_concurrently(
    tmp_path: Path, toy_registry: Registry, texts: dict[str, str], answers: Any
) -> None:
    """The configured LM lives in a context variable; a bare worker thread would not see it."""
    seen_threads: set[int] = set()

    class ThreadRecording:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def __call__(self, **kwargs: Any) -> Any:
            seen_threads.add(threading.get_ident())
            return self.inner(**kwargs)

    program = ThreadRecording(build_program(toy_registry))
    config = _config(production={"num_threads": 4})
    with scripted_lm(answers):
        outcomes = produce(toy_registry, config, program, texts, tmp_path)

    assert len(outcomes) == CORPUS
    assert all(outcome.ok for outcome in outcomes), [o.error for o in outcomes if not o.ok]
    assert len(seen_threads) > 1, "documents ran serially"
    # order follows the corpus, not whichever thread happened to finish first
    assert [outcome.doc_id for outcome in outcomes] == sorted(texts)


def test_null_rate_gate_refuses_to_pass_without_a_reference(toy_registry: Registry) -> None:
    """A stale reference run cannot make the gate look satisfied."""
    outcomes = [DocumentOutcome(doc_id="a", values={"flag": True, "state": "CA", "number": "A"})]
    checks = {
        check.name: check
        for check in run_qa(toy_registry, _config(), outcomes, expected_documents=1, reference_null_rates={})
    }
    assert checks["null rates"].passed is False
    assert "could not run" in checks["null rates"].detail
    assert "rescore" in checks["null rates"].detail


def test_failure_errors_stay_on_one_table_row(tmp_path: Path, toy_registry: Registry) -> None:
    """A provider traceback with newlines must not break the failures table apart."""
    outcomes = [DocumentOutcome(doc_id="a", error="AdapterParseError: bad\n\nLM Response: {'text': None}\n\n")]
    result = ProductionResult(
        outcomes=outcomes,
        checks=run_qa(toy_registry, _config(), outcomes, expected_documents=1),
        triage=triage(toy_registry, _config(), outcomes),
    )
    path = tmp_path / "qa_report.md"
    write_qa_report(path, toy_registry, result, expected_documents=1)
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("| a |")]
    assert len(rows) == 1
    assert rows[0].endswith("|")
    assert "LM Response: {'text': None}" in rows[0]


def test_a_plain_retry_replays_the_cached_truncation(
    toy_registry: Registry, isolated_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug, reproduced: an identical retry is answered from the cache, never the model."""
    import dspy
    import litellm

    stub = StubProvider(bad=10_000)
    monkeypatch.setattr(litellm, "completion", stub)
    program = build_program(toy_registry)
    with dspy.context(lm=real_lm()):
        with pytest.raises(Exception):  # noqa: B017 - any parse failure will do
            program(document=document_for("p00"))
        calls_after_first = stub.calls
        with pytest.raises(Exception):  # noqa: B017
            program(document=document_for("p00"))
    assert calls_after_first > 0
    assert stub.calls == calls_after_first, "the second identical call should never have reached the provider"


def test_a_retry_draws_a_fresh_answer_past_the_cache(
    tmp_path: Path, toy_registry: Registry, isolated_cache: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix: the retry reaches the provider, gets a complete answer, and the document succeeds."""
    import dspy
    import litellm

    # measure how many calls one failing attempt makes (the adapter may fall back and retry itself)
    probe = StubProvider(bad=10_000)
    monkeypatch.setattr(litellm, "completion", probe)
    with dspy.context(lm=real_lm()), pytest.raises(Exception):  # noqa: B017
        build_program(toy_registry)(document=document_for("probe"))
    first_attempt_calls = probe.calls
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=str(tmp_path / "c2"))

    stub = StubProvider(bad=first_attempt_calls)
    monkeypatch.setattr(litellm, "completion", stub)
    with dspy.context(lm=real_lm()):
        outcomes = produce(
            toy_registry,
            _config(production={"max_retries": 2, "num_threads": 1}),
            build_program(toy_registry),
            {"p00": document_for("p00")},
            tmp_path,
        )
    assert outcomes[0].ok, outcomes[0].error
    assert outcomes[0].attempts == 2
    assert outcomes[0].values["state"] == "CA"
    assert stub.calls > first_attempt_calls, "the retry never reached the provider"


def test_each_retry_uses_its_own_rollout(tmp_path: Path, toy_registry: Registry, texts: dict[str, str]) -> None:
    """Every attempt after the first carries a distinct rollout id, so a distinct cache key."""
    import dspy

    seen: dict[str, list[Any]] = {}
    lock = threading.Lock()

    class AlwaysFails:
        def __call__(self, **kwargs: Any) -> Any:
            rollout = dspy.settings.lm.kwargs.get("rollout_id")
            with lock:
                seen.setdefault(kwargs["document"], []).append(rollout)
            raise RuntimeError("truncated")

    with dspy.context(lm=real_lm()):
        produce(toy_registry, _config(production={"max_retries": 2, "num_threads": 4}), AlwaysFails(), texts, tmp_path)
    assert len(seen) == CORPUS
    # the first attempt shares the cache; each retry gets its own key, in every worker thread
    assert all(rollouts == [None, 2, 3] for rollouts in seen.values()), seen


def test_typed_values_reach_the_deliverable_as_structured_json(tmp_path: Path) -> None:
    """outputs.jsonl is what the client gets: no Python repr strings in it, ever."""
    from doc_harness.values import PartialDate, Quantity, Span

    values = {
        "contract_value": Quantity(value=375000.0, unit="USD"),
        "effective_date": PartialDate(value="2024-11-15", granularity="day"),
        "governing_law_span": Span(start=726, end=877),
    }
    outcome = DocumentOutcome(doc_id="d1", values=values)
    write_outputs(tmp_path / "outputs.jsonl", [outcome])
    row = json.loads((tmp_path / "outputs.jsonl").read_text(encoding="utf-8"))
    assert row["values"]["contract_value"] == {"value": 375000.0, "unit": "USD"}
    assert row["values"]["effective_date"] == {"value": "2024-11-15", "granularity": "day"}
    assert row["values"]["governing_law_span"]["start"] == 726


def test_a_resumed_run_reads_structured_values_back(tmp_path: Path) -> None:
    """A checkpoint written as repr text would come back as a string on resume."""
    from doc_harness.produce import _load_checkpoint, _write_checkpoint
    from doc_harness.values import Quantity

    raw_dir = tmp_path / "raw"
    _write_checkpoint(raw_dir, DocumentOutcome(doc_id="d1", values={"contract_value": Quantity(value=5.0, unit="USD")}))
    resumed = _load_checkpoint(raw_dir, "d1")
    assert resumed is not None
    assert resumed.values["contract_value"] == {"value": 5.0, "unit": "USD"}
