"""The full-corpus run and the QA gates that stand between it and a handover.

Three properties matter more than throughput:

* **checkpointed and resumable** -- raw responses land on disk before any post-processing,
  so a run that dies halfway resumes instead of starting over and paying twice;
* **no silent drops** -- a document that fails every retry is recorded as a failure and
  counted; the run reconciles its counts and refuses to finish quietly if they disagree;
* **gated** -- coverage, schema, class distribution and null rates are checked against the
  labeled sample before anyone reads the outputs as results.

Note on the Batch API: this runs concurrent requests rather than a provider's asynchronous
batch endpoint. Going through DSPy's adapters is what keeps the typed parsing and the
normalizers identical to validation, which matters more than the batch discount; the
checkpoint format is designed so a batch backend can be dropped in behind it.
"""

from __future__ import annotations

import contextvars
import csv
import json
import logging
import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from doc_harness.config import Config
from doc_harness.evaluate import is_abstention, to_json_value
from doc_harness.metric import get_field
from doc_harness.program import fresh_generation
from doc_harness.registry import Registry, TaskType
from doc_harness.splits import CLASSIFICATION_TYPES, _labels_of

logger = logging.getLogger(__name__)

# a class share that moves by more than this against the labeled sample is worth a look
DISTRIBUTION_SHIFT = 0.15
# a per-task null rate this far above the validation rate suggests the program gave up
NULL_RATE_TOLERANCE = 0.15
# documents in this share of the highest null counts are routed as low confidence
LOW_CONFIDENCE_DECILE = 0.10


class ProductionError(RuntimeError):
    """Raised when the full-corpus run cannot be trusted to have covered the corpus."""


@dataclass
class DocumentOutcome:
    """One document's production result, successful or not."""

    doc_id: str
    values: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    attempts: int = 1

    @property
    def ok(self) -> bool:
        """Return whether this document produced usable output."""
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        """Return the raw checkpoint payload, with typed values as structured JSON.

        Not ``default=str``: that writes a number as ``"value=375000.0 unit='USD'"``, which is
        useless to anyone consuming outputs.jsonl and is read back as a string on resume.
        """
        return {
            "doc_id": self.doc_id,
            "values": {task_id: to_json_value(value) for task_id, value in self.values.items()},
            "error": self.error,
            "attempts": self.attempts,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }


@dataclass
class QACheck:
    """One gate and whether it passed."""

    name: str
    passed: bool
    detail: str

    def to_markdown(self) -> str:
        """Render this check as a report line."""
        return f"| {self.name} | {'pass' if self.passed else '**FAIL**'} | {self.detail} |"


@dataclass
class ProductionResult:
    """Everything the full-corpus run produced."""

    outcomes: list[DocumentOutcome]
    checks: list[QACheck]
    triage: dict[str, list[str]]

    @property
    def succeeded(self) -> list[DocumentOutcome]:
        """Return the documents that produced output."""
        return [outcome for outcome in self.outcomes if outcome.ok]

    @property
    def failures(self) -> list[DocumentOutcome]:
        """Return the documents that failed every retry."""
        return [outcome for outcome in self.outcomes if not outcome.ok]

    @property
    def all_passed(self) -> bool:
        """Return whether every gate passed."""
        return all(check.passed for check in self.checks)


def _raw_path(raw_dir: Path, doc_id: str) -> Path:
    return raw_dir / f"{doc_id}.json"


def _load_checkpoint(raw_dir: Path, doc_id: str) -> DocumentOutcome | None:
    """Read one document's checkpointed response, if the run already produced it."""
    path = _raw_path(raw_dir, doc_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    outcome = DocumentOutcome(
        doc_id=data["doc_id"],
        values=dict(data.get("values", {})),
        error=str(data.get("error", "")),
        attempts=int(data.get("attempts", 1)),
    )
    # a checkpointed failure is retried; a checkpointed success is never paid for twice
    return outcome if outcome.ok else None


def _write_checkpoint(raw_dir: Path, outcome: DocumentOutcome) -> None:
    """Write one document's raw response before any post-processing touches it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    _raw_path(raw_dir, outcome.doc_id).write_text(json.dumps(outcome.to_dict(), indent=2) + "\n", encoding="utf-8")


def _run_one(program: Any, registry: Registry, doc_id: str, text: str, max_retries: int) -> DocumentOutcome:
    """Run one document, retrying with a fresh generation before recording the failure."""
    last_error = ""
    for attempt in range(1, max_retries + 2):
        try:
            with fresh_generation(attempt):
                prediction = program(document=text)
            values = {task.id: get_field(prediction, task.id) for task in registry}
            return DocumentOutcome(doc_id=doc_id, values=values, attempts=attempt)
        except Exception as exc:  # noqa: BLE001 - the failure is recorded, never swallowed
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s failed on attempt %d: %s", doc_id, attempt, last_error)
    return DocumentOutcome(doc_id=doc_id, error=last_error, attempts=max_retries + 1)


def produce(
    registry: Registry,
    config: Config,
    program: Any,
    texts: Mapping[str, str],
    project_dir: Path,
    resume: bool = True,
) -> list[DocumentOutcome]:
    """Run the compiled program over the whole corpus, checkpointing as it goes.

    :param registry: The parsed tasks.yaml
    :param config: The project configuration
    :param program: The compiled champion, loaded rather than recompiled
    :param texts: Document id to extracted text, for every document in the corpus
    :param project_dir: The project root
    :param resume: Reuse checkpointed successes rather than re-running them
    :returns: One outcome per document, in corpus order regardless of completion order
    """
    raw_dir = project_dir / "runs" / "production" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    doc_ids = sorted(texts)
    by_doc: dict[str, DocumentOutcome] = {}
    pending: list[str] = []
    for doc_id in doc_ids:
        cached = _load_checkpoint(raw_dir, doc_id) if resume else None
        if cached is not None:
            by_doc[doc_id] = cached
        else:
            pending.append(doc_id)
    reused = len(by_doc)

    if pending:
        with ThreadPoolExecutor(max_workers=config.production.num_threads) as pool:
            # each task runs in a copy of this thread's context, because DSPy keeps the
            # configured LM in a context variable a bare worker thread would not see
            futures = {
                pool.submit(
                    contextvars.copy_context().run,
                    _run_one,
                    program,
                    registry,
                    doc_id,
                    texts[doc_id],
                    config.production.max_retries,
                ): doc_id
                for doc_id in pending
            }
            for future in as_completed(futures):
                doc_id = futures[future]
                # a checkpoint is written as each document lands, so a run that dies keeps
                # everything it has already paid for
                outcome = future.result()
                _write_checkpoint(raw_dir, outcome)
                by_doc[doc_id] = outcome

    outcomes = [by_doc[doc_id] for doc_id in doc_ids]
    if len(outcomes) != len(doc_ids):
        raise ProductionError(f"produced {len(outcomes)} outcomes for {len(doc_ids)} documents: documents were dropped")
    failures = [outcome.doc_id for outcome in outcomes if not outcome.ok]
    logger.info(
        "production run covered %d documents (%d resumed from checkpoint, %d failed)",
        len(outcomes),
        reused,
        len(failures),
    )
    if failures:
        logger.warning("%d document(s) failed every retry and are recorded as failures: %s", len(failures), failures)
    return outcomes


def write_outputs(path: Path, outcomes: Sequence[DocumentOutcome]) -> None:
    """Write outputs.jsonl, including the failures so the file covers the corpus."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for outcome in outcomes:
        row: dict[str, Any] = {"doc_id": outcome.doc_id, "ok": outcome.ok}
        if outcome.ok:
            row["values"] = {task_id: to_json_value(value) for task_id, value in outcome.values.items()}
        else:
            row["error"] = outcome.error
        lines.append(json.dumps(row, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", path)


def _class_shares(registry: Registry, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Return each classification task's class shares over a set of answers."""
    shares: dict[str, dict[str, float]] = {}
    total = len(rows) or 1
    for task in registry:
        if TaskType(task.type) not in CLASSIFICATION_TYPES:
            continue
        counts: dict[str, int] = {}
        for row in rows:
            for label in _labels_of(row.get(task.id)):
                counts[label] = counts.get(label, 0) + 1
        shares[task.id] = {label: count / total for label, count in counts.items()}
    return shares


def _null_rates(registry: Registry, rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Return each task's abstention rate over a set of answers."""
    total = len(rows) or 1
    return {task.id: sum(1 for row in rows if is_abstention(row.get(task.id))) / total for task in registry}


def run_qa(
    registry: Registry,
    config: Config,
    outcomes: Sequence[DocumentOutcome],
    expected_documents: int,
    reference_null_rates: Mapping[str, float] | None = None,
    reference_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[QACheck]:
    """Run every QA gate against the production output.

    :param registry: The parsed tasks.yaml
    :param config: The project configuration
    :param outcomes: The production outcomes
    :param expected_documents: How many documents the corpus contains
    :param reference_null_rates: Each task's validation abstention rate, from its recorded run
    :param reference_rows: The labeled sample's gold values, for the distribution comparison
    :returns: One check per gate
    """
    checks: list[QACheck] = []
    produced = len(outcomes)
    checks.append(
        QACheck(
            name="coverage",
            passed=produced == expected_documents,
            detail=f"{produced} outputs for {expected_documents} documents",
        )
    )

    successful = [outcome for outcome in outcomes if outcome.ok]
    failure_rate = 1.0 - (len(successful) / produced) if produced else 1.0
    checks.append(
        QACheck(
            name="schema",
            passed=failure_rate <= config.production.max_schema_failure_rate,
            detail=(
                f"{len(outcomes) - len(successful)} parse or call failure(s), "
                f"{failure_rate:.2%} against a ceiling of {config.production.max_schema_failure_rate:.2%}"
            ),
        )
    )

    rows = [outcome.values for outcome in successful]
    if reference_rows:
        shifts: list[str] = []
        produced_shares = _class_shares(registry, rows)
        reference_shares = _class_shares(registry, reference_rows)
        for task_id, labels in produced_shares.items():
            for label in set(labels) | set(reference_shares.get(task_id, {})):
                delta = labels.get(label, 0.0) - reference_shares.get(task_id, {}).get(label, 0.0)
                if abs(delta) > DISTRIBUTION_SHIFT:
                    shifts.append(f"{task_id}/{label} {delta:+.1%}")
        checks.append(
            QACheck(
                name="class distribution",
                passed=not shifts,
                detail=(
                    "no class share moved by more than " f"{DISTRIBUTION_SHIFT:.0%} against the labeled sample"
                    if not shifts
                    else "shifted: " + ", ".join(sorted(shifts))
                ),
            )
        )

    if reference_null_rates is not None and not reference_null_rates:
        # a gate that cannot run must not read as a gate that passed
        checks.append(
            QACheck(
                name="null rates",
                passed=False,
                detail=(
                    "the reference run records no abstention rates, so this check could not run. "
                    "Re-score the champion's run (doc-harness rescore <exp_id>) and try again"
                ),
            )
        )
    elif reference_null_rates is not None:
        produced_nulls = _null_rates(registry, rows)
        risen: list[str] = []
        for task_id, rate in produced_nulls.items():
            if task_id not in reference_null_rates:
                continue
            validation_rate = reference_null_rates[task_id]
            if rate - validation_rate > NULL_RATE_TOLERANCE:
                risen.append(f"{task_id} {rate:.1%} against {validation_rate:.1%}")
        checks.append(
            QACheck(
                name="null rates",
                passed=not risen,
                detail=(
                    "abstention rates are in line with validation"
                    if not risen
                    else "well above validation: " + ", ".join(sorted(risen))
                ),
            )
        )

    checks.append(
        QACheck(
            name="manual spot-check",
            passed=True,
            detail=f"{min(config.production.spot_check_sample, len(successful))} random document(s) flagged for review",
        )
    )
    return checks


def triage(
    registry: Registry,
    config: Config,
    outcomes: Sequence[DocumentOutcome],
    manifest_path: Path | None = None,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Route documents to human review.

    Four targeted routes plus a random slice. The random slice is not optional: the first
    four select documents that are already suspect, so a quality estimate built on them
    alone is biased and will read worse than the corpus really is.

    :param registry: The parsed tasks.yaml
    :param config: The project configuration
    :param outcomes: The production outcomes
    :param manifest_path: extraction_manifest.csv, for the OCR and truncation routes
    :param seed: Fixed so the random slice is reproducible
    :returns: Route name to document ids
    """
    routes: dict[str, list[str]] = {
        "ocr": [],
        "truncated": [],
        "nulls_on_answered_tasks": [],
        "low_confidence": [],
        "random_slice": [],
        "failed": sorted(outcome.doc_id for outcome in outcomes if not outcome.ok),
    }
    produced_ids = {outcome.doc_id for outcome in outcomes}

    if manifest_path and manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["doc_id"] not in produced_ids:
                    continue
                if row.get("ocr") == "True" or row.get("ocr_needed") == "True":
                    routes["ocr"].append(row["doc_id"])
                if row.get("truncated") == "True":
                    routes["truncated"].append(row["doc_id"])

    successful = [outcome for outcome in outcomes if outcome.ok]
    rows = [outcome.values for outcome in successful]
    null_rates = _null_rates(registry, rows)
    usually_answered = {task_id for task_id, rate in null_rates.items() if rate < 0.5}
    null_counts: list[tuple[int, str]] = []
    for outcome in successful:
        nulls = {task.id for task in registry if not _labels_of(outcome.values.get(task.id))}
        if nulls & usually_answered:
            routes["nulls_on_answered_tasks"].append(outcome.doc_id)
        null_counts.append((len(nulls), outcome.doc_id))

    # without logprobs there is no direct confidence signal, so the proxy is how much of a
    # document the program declined to answer; it is a proxy and the report says so
    null_counts.sort(key=lambda item: (-item[0], item[1]))
    cutoff = max(1, int(len(null_counts) * LOW_CONFIDENCE_DECILE))
    routes["low_confidence"] = sorted(doc_id for count, doc_id in null_counts[:cutoff] if count > 0)

    rng = random.Random(seed)
    pool = sorted(outcome.doc_id for outcome in successful)
    take = min(config.production.spot_check_sample, len(pool))
    routes["random_slice"] = sorted(rng.sample(pool, take)) if take else []

    for name in routes:
        routes[name] = sorted(set(routes[name]))
    return routes


def write_qa_report(
    path: Path,
    registry: Registry,
    result: ProductionResult,
    expected_documents: int,
) -> None:
    """Write qa_report.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Production QA",
        "",
        f"{len(result.succeeded)} of {expected_documents} documents produced output; "
        f"{len(result.failures)} failed every retry and are recorded as failures.",
        "",
        "| check | result | detail |",
        "| --- | --- | --- |",
    ]
    lines += [check.to_markdown() for check in result.checks]
    lines += [
        "",
        "**Gates "
        + ("all passed." if result.all_passed else "did not all pass; do not hand these results over yet.")
        + "**",
        "",
        "## Human review routes",
        "",
        "The random slice is not optional. The four targeted routes select documents that are",
        "already suspect, so a quality estimate built on them alone reads worse than the corpus is.",
        "",
        "| route | documents |",
        "| --- | --- |",
    ]
    for name, doc_ids in result.triage.items():
        sample = ", ".join(doc_ids[:8]) + (f", and {len(doc_ids) - 8} more" if len(doc_ids) > 8 else "")
        lines.append(f"| {name} | {len(doc_ids)}{': ' + sample if doc_ids else ''} |")
    lines.append("")
    if result.failures:
        lines += ["## Failures", "", "| doc_id | attempts | error |", "| --- | --- | --- |"]
        for outcome in result.failures:
            # collapsed to one line: a provider traceback with newlines in it would otherwise
            # break the table apart and bury the other failures
            error = " ".join(outcome.error.split())[:300]
            lines.append(f"| {outcome.doc_id} | {outcome.attempts} | {error} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", path)
