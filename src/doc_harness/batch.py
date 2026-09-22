"""Production through the Message Batches API, at half the price of live requests.

Batching must not change what the program does, or the validation numbers stop describing
production. So nothing here builds a request or parses a reply by hand:

* **requests** are the live path's own. Each predictor is run exactly as production would run
  it, and the request body is captured at the point LiteLLM would send it -- the batch body is
  the live body by construction, not by imitation;
* **replies** go through the adapter's own post-processing, the same call the live path makes,
  and then through the program's own merge.

A document is only taken from the batch when every one of its requests succeeded and parsed.
Anything else -- an errored or expired request, an unreadable reply -- is handed back to the
live path, with its retries and fresh generations, so failure handling is identical too.

Batch ids are written to disk the moment a batch is created, before any waiting, so a run
that is interrupted resumes by collecting the batch it already paid for rather than
submitting and paying again.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from doc_harness.adapters import strict_version_of
from doc_harness.program import INPUT_FIELD, attribute_for

logger = logging.getLogger(__name__)

STATE_FILE = "batches.json"
# well under the API's own ceiling of 100,000 requests or 256 MB per batch
MAX_REQUESTS_PER_BATCH = 10_000


class BatchError(RuntimeError):
    """Raised when a batch cannot be built or collected faithfully."""


@dataclass(frozen=True)
class BatchRequest:
    """One predictor's request for one document, with the id it travels under."""

    custom_id: str
    doc_id: str
    group: str
    params: dict[str, Any]


@contextmanager
def _capture_http(sink: list[dict[str, Any]]) -> Iterator[None]:
    """Record the request body LiteLLM is about to send, and send nothing.

    The live path runs unchanged up to the moment of sending, so the body recorded here is
    byte for byte what a live request would have been. The call is then stopped with an
    error, which DSPy treats as a model error and does not retry through a fallback adapter.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    original = HTTPHandler.post

    def capture(self: Any, url: str, data: Any = None, json: Any = None, **kwargs: Any) -> Any:
        import json as json_module

        body = json if json is not None else json_module.loads(data)
        sink.append(dict(body))
        raise BatchError("request captured for batch submission")

    HTTPHandler.post = capture  # type: ignore[method-assign,assignment]
    try:
        yield
    finally:
        HTTPHandler.post = original  # type: ignore[method-assign]


def render_requests(program: Any, texts: Mapping[str, str], lm: Any) -> list[BatchRequest]:
    """Build every document's requests by running the live path up to the point of sending.

    :param program: The compiled program, whose predictors and demonstrations are used as is
    :param texts: Document id to extracted text
    :param lm: The configured task model
    :returns: One request per document per predictor group
    """
    import dspy

    # the cache would answer instead of sending, and retries would send twice
    capture_lm = lm.copy(cache=False, num_retries=0)
    adapter = strict_version_of(dspy.settings.adapter)
    requests: list[BatchRequest] = []
    for doc_index, doc_id in enumerate(sorted(texts)):
        for group_index, group in enumerate(program.group_tasks):
            predictor = getattr(program, attribute_for(group))
            sink: list[dict[str, Any]] = []
            # the capture itself stops the call with an error, so the error is expected
            with _capture_http(sink), dspy.context(lm=capture_lm, adapter=adapter), suppress(Exception):
                predictor(**{INPUT_FIELD: texts[doc_id]})
            if len(sink) != 1:
                raise BatchError(f"expected one request for {doc_id}/{group}, captured {len(sink)}")
            params = sink[0]
            params.pop("stream", None)
            requests.append(BatchRequest(f"d{doc_index}-g{group_index}", doc_id, group, params))
    return requests


def _state_path(production_dir: Path) -> Path:
    return production_dir / STATE_FILE


def load_state(production_dir: Path) -> dict[str, Any]:
    """Read the record of submitted batches, so an interrupted run can collect them."""
    path = _state_path(production_dir)
    if not path.exists():
        return {"batches": []}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _save_state(production_dir: Path, state: Mapping[str, Any]) -> None:
    production_dir.mkdir(parents=True, exist_ok=True)
    _state_path(production_dir).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def submit(client: Any, requests: list[BatchRequest], production_dir: Path) -> None:
    """Create batches for the requests, recording each id before anything else happens."""
    state = load_state(production_dir)
    for start in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
        chunk = requests[start : start + MAX_REQUESTS_PER_BATCH]
        batch = client.messages.batches.create(
            requests=[{"custom_id": request.custom_id, "params": request.params} for request in chunk]
        )
        state["batches"].append(
            {
                "batch_id": batch.id,
                "collected": False,
                "requests": {request.custom_id: [request.doc_id, request.group] for request in chunk},
            }
        )
        # written before waiting: a crash from here on still knows what it has paid for
        _save_state(production_dir, state)
        logger.info("submitted batch %s with %d request(s)", batch.id, len(chunk))


def wait_for(client: Any, batch_id: str, poll_seconds: int, sleep: Callable[[float], None] = time.sleep) -> None:
    """Poll a batch until it has ended."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return
        counts = batch.request_counts
        logger.info(
            "batch %s: %s -- %d processing, %d succeeded, %d errored",
            batch_id,
            batch.processing_status,
            counts.processing,
            counts.succeeded,
            counts.errored,
        )
        sleep(poll_seconds)


def _reply_text(result: Any) -> str | None:
    """Return a succeeded result's answer text, leaving out any thinking blocks."""
    if result.result.type != "succeeded":
        return None
    return "".join(block.text for block in result.result.message.content if block.type == "text")


def collect(
    client: Any,
    program: Any,
    texts: Mapping[str, str],
    lm: Any,
    production_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, str]]:
    """Turn every ended, uncollected batch's replies into predictions.

    :returns: Predictions for documents whose every request succeeded and parsed, the raw
        reply text behind them, and the reason each remaining document fell back to live
    """
    import dspy

    adapter = strict_version_of(dspy.settings.adapter)
    state = load_state(production_dir)
    replies: dict[str, dict[str, str]] = {}
    refused: dict[str, str] = {}
    for entry in state["batches"]:
        if entry["collected"]:
            continue
        for result in client.messages.batches.results(entry["batch_id"]):
            doc_id, group = entry["requests"][result.custom_id]
            text = _reply_text(result)
            if text is None:
                refused[doc_id] = f"batch request {result.result.type}"
            else:
                replies.setdefault(doc_id, {})[group] = text

    predictions: dict[str, Any] = {}
    for doc_id, by_group in replies.items():
        if doc_id in refused or doc_id not in texts:
            continue
        if set(by_group) != set(program.group_tasks):
            refused[doc_id] = "not every request in the batch returned"
            continue
        try:
            parsed = {group: _parse(program, adapter, lm, group, texts[doc_id], by_group[group]) for group in by_group}
        except Exception as exc:  # noqa: BLE001 - the live path retries it
            refused[doc_id] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            continue
        predictions[doc_id] = program.assemble(texts[doc_id], parsed)
    return predictions, {doc_id: replies[doc_id] for doc_id in predictions}, refused


def _parse(program: Any, adapter: Any, lm: Any, group: str, document: str, text: str) -> dict[str, Any]:
    """Parse one reply through the adapter's own post-processing, as the live path does."""
    predictor = getattr(program, attribute_for(group))
    lm_kwargs = dict(getattr(predictor, "config", {}) or {})
    inputs = {INPUT_FIELD: document}
    processed = adapter._call_preprocess(lm, lm_kwargs, predictor.signature, inputs)
    values: list[dict[str, Any]] = adapter._call_postprocess(processed, predictor.signature, [text], lm, lm_kwargs)
    return values[0]


def mark_collected(production_dir: Path) -> None:
    """Record that every submitted batch has been collected, so a rerun does not reread them."""
    state = load_state(production_dir)
    for entry in state["batches"]:
        entry["collected"] = True
    _save_state(production_dir, state)


def pending_in_batches(production_dir: Path) -> set[str]:
    """Return documents already submitted in a batch that has not been collected."""
    covered: set[str] = set()
    for entry in load_state(production_dir)["batches"]:
        if not entry["collected"]:
            covered.update(doc_id for doc_id, _group in entry["requests"].values())
    return covered


def run_batches(
    program: Any,
    texts: Mapping[str, str],
    lm: Any,
    production_dir: Path,
    client: Any,
    poll_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, str]]:
    """Submit what has not been submitted, wait for every open batch, and collect it.

    :param program: The compiled program
    :param texts: Document id to text, for the documents still to produce
    :param lm: The configured task model
    :param production_dir: runs/production, where the batch record lives
    :param client: An Anthropic client
    :param poll_seconds: Seconds between status checks
    :param sleep: Injected so tests do not wait
    :returns: Predictions, the raw replies behind them, and why any document fell back to live
    """
    already = pending_in_batches(production_dir)
    to_submit = {doc_id: text for doc_id, text in texts.items() if doc_id not in already}
    if to_submit:
        submit(client, render_requests(program, to_submit, lm), production_dir)
    else:
        logger.info("resuming: every document is already in a submitted batch")
    for entry in load_state(production_dir)["batches"]:
        if not entry["collected"]:
            wait_for(client, entry["batch_id"], poll_seconds, sleep)
    predictions, raw, refused = collect(client, program, texts, lm, production_dir)
    mark_collected(production_dir)
    return predictions, raw, refused
