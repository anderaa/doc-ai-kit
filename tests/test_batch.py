"""Production through the Message Batches API.

The contract: batching changes the price and nothing else. Requests are the live path's own,
replies are parsed by the live path's own code, and anything the batch cannot deliver falls
back to the live path. Every test here runs offline -- the batch service and the model's HTTP
endpoint are both faked.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspy
import httpx
import pytest
from conftest import document_for
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from doc_harness import batch as batch_module
from doc_harness.batch import STATE_FILE, render_requests
from doc_harness.config import Config
from doc_harness.produce import produce
from doc_harness.program import build_program
from doc_harness.registry import Registry

GOOD = "[[ ## flag ## ]]\ntrue\n\n[[ ## state ## ]]\n{state}\n\n[[ ## number ## ]]\n{number}\n\n[[ ## completed ## ]]"
GARBLED = "[[ ## flag ## ]]\ntr"


def reply_for(doc_id: str) -> str:
    """The right answer for a synthetic document, keyed on its number."""
    index = int(doc_id[1:])
    return GOOD.format(state="CA" if index % 2 else "NY", number=f"A-{index}")


def doc_in(params: dict[str, Any]) -> str:
    """Find which document a request is about, from the text it carries."""
    match = re.search(r"\[doc=(\w+)\]", json.dumps(params))
    assert match, "request does not carry its document"
    return match.group(1)


class FakeBatches:
    """Stands in for client.messages.batches, answering each request through ``respond``."""

    def __init__(self, respond: Callable[[str], tuple[str, str | None]], polls_until_ended: int = 1) -> None:
        self.respond = respond
        self.polls_until_ended = polls_until_ended
        self.created: dict[str, list[dict[str, Any]]] = {}
        self.polls: dict[str, int] = {}

    def create(self, requests: list[dict[str, Any]]) -> Any:
        batch_id = f"msgbatch_{len(self.created)}"
        self.created[batch_id] = list(requests)
        self.polls[batch_id] = 0
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id: str) -> Any:
        self.polls[batch_id] += 1
        ended = self.polls[batch_id] >= self.polls_until_ended
        counts = SimpleNamespace(processing=0 if ended else 1, succeeded=0, errored=0)
        return SimpleNamespace(
            id=batch_id, processing_status="ended" if ended else "in_progress", request_counts=counts
        )

    def results(self, batch_id: str) -> Iterator[Any]:
        for request in self.created[batch_id]:
            kind, text = self.respond(doc_in(request["params"]))
            if kind == "succeeded":
                # a thinking block ahead of the answer, as the model returns with thinking on
                content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
                result = SimpleNamespace(type="succeeded", message=SimpleNamespace(content=content))
            else:
                result = SimpleNamespace(type=kind)
            yield SimpleNamespace(custom_id=request["custom_id"], result=result)


def fake_client(respond: Callable[[str], tuple[str, str | None]], **kwargs: Any) -> Any:
    return SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(respond, **kwargs)))


def always_right(doc_id: str) -> tuple[str, str | None]:
    return "succeeded", reply_for(doc_id)


@pytest.fixture
def live_http(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer live model calls at the HTTP layer, recording which documents went live."""
    went_live: list[str] = []

    def post(self: Any, url: str, data: Any = None, json: Any = None, **kwargs: Any) -> httpx.Response:
        body = json if json is not None else __import__("json").loads(data)
        doc_id = doc_in(body)
        went_live.append(doc_id)
        message = {
            "id": "msg_live", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": reply_for(doc_id)}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 10},
        }  # fmt: skip
        return httpx.Response(200, json=message, request=httpx.Request("POST", url))

    monkeypatch.setattr(HTTPHandler, "post", post)
    return went_live


@pytest.fixture
def lm() -> Iterator[Any]:
    model = dspy.LM("anthropic/claude-sonnet-5", temperature=1.0, max_tokens=512, cache=False, num_retries=0)
    with dspy.context(lm=model):
        yield model


def _config(**production: Any) -> Config:
    return Config.from_mapping(
        {
            "models": {"task": "anthropic/claude-sonnet-5"},
            "production": {"max_retries": 1, "num_threads": 2, "batch_poll_seconds": 1, **production},
        }
    )


def _texts(n: int) -> dict[str, str]:
    return {f"p{i:02d}": document_for(f"p{i:02d}") for i in range(n)}


def _no_wait(_seconds: float) -> None:
    return None


# --- fidelity -------------------------------------------------------------------------------


def test_a_batch_request_is_exactly_the_live_request(toy_registry: Registry, lm: Any) -> None:
    """Captured independently from a normal program call, the live body must equal the batched one."""
    program = build_program(toy_registry)
    text = document_for("p07")
    live_bodies: list[dict[str, Any]] = []

    def capture(self: Any, url: str, data: Any = None, json: Any = None, **kwargs: Any) -> Any:
        live_bodies.append(json if json is not None else __import__("json").loads(data))
        raise RuntimeError("stop")

    original = HTTPHandler.post
    HTTPHandler.post = capture  # type: ignore[method-assign,assignment]
    try:
        with pytest.raises(Exception):  # noqa: B017 - the capture stops the call
            program(document=text)
    finally:
        HTTPHandler.post = original  # type: ignore[method-assign]

    batched = render_requests(program, {"p07": text}, lm)
    assert len(batched) == 1
    assert batched[0].params == live_bodies[0]


def test_thinking_settings_travel_into_the_batch(toy_registry: Registry) -> None:
    """The batch carries the task model's effort exactly as a live call would."""
    model = dspy.LM("anthropic/claude-sonnet-5", temperature=1.0, max_tokens=512, cache=False, reasoning_effort="low")
    with dspy.context(lm=model):
        params = render_requests(build_program(toy_registry), {"p00": document_for("p00")}, model)[0].params
    assert params["output_config"] == {"effort": "low"}
    assert params["thinking"]["type"] == "adaptive"
    assert params["model"] == "claude-sonnet-5"
    assert "stream" not in params


def test_each_predictor_group_is_its_own_request(fixtures_dir: Path, lm: Any) -> None:
    registry = Registry.from_yaml(fixtures_dir / "tasks_all_types.yaml")
    requests = render_requests(build_program(registry), _texts(3), lm)
    assert len(requests) == 3 * len(registry.groups())
    assert len({request.custom_id for request in requests}) == len(requests)
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", request.custom_id) for request in requests)


# --- the production path --------------------------------------------------------------------


def test_the_batch_delivers_the_same_values_as_the_live_path(
    tmp_path: Path, toy_registry: Registry, lm: Any, live_http: list[str]
) -> None:
    """Same reply text, either route, same answers: batching changes the price and nothing else."""
    texts = _texts(4)
    batched = produce(
        toy_registry, _config(), build_program(toy_registry), texts, tmp_path / "b",
        batch_client=fake_client(always_right), sleep=_no_wait,
    )  # fmt: skip
    live = produce(
        toy_registry, _config(use_batch_api=False), build_program(toy_registry), texts, tmp_path / "l",
    )  # fmt: skip
    assert all(outcome.via == "batch" for outcome in batched)
    assert all(outcome.via == "live" for outcome in live)
    assert [o.values for o in batched] == [o.values for o in live]
    assert batched[1].values == {"flag": True, "state": "CA", "number": "A-1"}


def test_the_raw_reply_is_kept_before_post_processing(tmp_path: Path, toy_registry: Registry, lm: Any) -> None:
    produce(
        toy_registry, _config(), build_program(toy_registry), _texts(2), tmp_path,
        batch_client=fake_client(always_right), sleep=_no_wait,
    )  # fmt: skip
    checkpoint = json.loads((tmp_path / "runs" / "production" / "raw" / "p01.json").read_text(encoding="utf-8"))
    assert checkpoint["via"] == "batch"
    assert checkpoint["raw"]["all"] == reply_for("p01")


def test_a_failed_batch_request_falls_back_to_live(
    tmp_path: Path, toy_registry: Registry, lm: Any, live_http: list[str]
) -> None:
    def respond(doc_id: str) -> tuple[str, str | None]:
        return ("errored", None) if doc_id == "p01" else always_right(doc_id)

    outcomes = produce(
        toy_registry, _config(), build_program(toy_registry), _texts(3), tmp_path,
        batch_client=fake_client(respond), sleep=_no_wait,
    )  # fmt: skip
    via = {outcome.doc_id: outcome.via for outcome in outcomes}
    assert via == {"p00": "batch", "p01": "live", "p02": "batch"}
    assert all(outcome.ok for outcome in outcomes)
    assert live_http == ["p01"]


def test_an_unreadable_batch_reply_falls_back_to_live(
    tmp_path: Path, toy_registry: Registry, lm: Any, live_http: list[str]
) -> None:
    """Strict parsing holds on the batch path too: a cut-off reply is never taken as nulls."""

    def respond(doc_id: str) -> tuple[str, str | None]:
        return ("succeeded", GARBLED) if doc_id == "p02" else always_right(doc_id)

    outcomes = produce(
        toy_registry, _config(), build_program(toy_registry), _texts(3), tmp_path,
        batch_client=fake_client(respond), sleep=_no_wait,
    )  # fmt: skip
    p02 = next(outcome for outcome in outcomes if outcome.doc_id == "p02")
    assert p02.via == "live"
    assert p02.values["number"] == "A-2"


def test_an_interrupted_run_collects_its_batch_instead_of_paying_again(
    tmp_path: Path, toy_registry: Registry, lm: Any
) -> None:
    client = fake_client(always_right, polls_until_ended=2)

    def interrupted(_seconds: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        produce(toy_registry, _config(), build_program(toy_registry), _texts(3), tmp_path,
                batch_client=client, sleep=interrupted)  # fmt: skip
    state = json.loads((tmp_path / "runs" / "production" / STATE_FILE).read_text(encoding="utf-8"))
    assert [entry["batch_id"] for entry in state["batches"]] == ["msgbatch_0"]

    outcomes = produce(toy_registry, _config(), build_program(toy_registry), _texts(3), tmp_path,
                       batch_client=client, sleep=_no_wait)  # fmt: skip
    assert len(client.messages.batches.created) == 1, "the batch was submitted, and paid for, twice"
    assert all(outcome.via == "batch" for outcome in outcomes)


def test_no_resume_submits_afresh(tmp_path: Path, toy_registry: Registry, lm: Any) -> None:
    client = fake_client(always_right)
    produce(toy_registry, _config(), build_program(toy_registry), _texts(2), tmp_path,
            batch_client=client, sleep=_no_wait)  # fmt: skip
    produce(toy_registry, _config(), build_program(toy_registry), _texts(2), tmp_path,
            resume=False, batch_client=client, sleep=_no_wait)  # fmt: skip
    assert len(client.messages.batches.created) == 2


def test_large_runs_split_into_several_batches(
    tmp_path: Path, toy_registry: Registry, lm: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch_module, "MAX_REQUESTS_PER_BATCH", 2)
    client = fake_client(always_right)
    outcomes = produce(toy_registry, _config(), build_program(toy_registry), _texts(5), tmp_path,
                       batch_client=client, sleep=_no_wait)  # fmt: skip
    assert len(client.messages.batches.created) == 3
    assert len(outcomes) == 5 and all(outcome.via == "batch" for outcome in outcomes)


@pytest.mark.parametrize(
    "config",
    [
        Config.from_mapping({"models": {"task": "openai/gpt-x"}, "production": {"num_threads": 1}}),
        Config.from_mapping(
            {"models": {"task": "anthropic/claude-sonnet-5"}, "production": {"use_batch_api": False, "num_threads": 1}}
        ),
    ],
    ids=["other provider", "batch switched off"],
)
def test_runs_live_when_the_batch_does_not_apply(
    tmp_path: Path, toy_registry: Registry, lm: Any, live_http: list[str], config: Config
) -> None:
    client = fake_client(always_right)
    outcomes = produce(toy_registry, config, build_program(toy_registry), _texts(2), tmp_path,
                       batch_client=client, sleep=_no_wait)  # fmt: skip
    assert client.messages.batches.created == {}
    assert all(outcome.via == "live" for outcome in outcomes)
