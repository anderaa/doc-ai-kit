"""Run the zero-shot program over every adversarial document and save the predictions.

This is a matcher check, not an accuracy measurement. It deliberately scores all 24
documents rather than a validation split, because the point is to collect as many real
threshold decisions as possible for `doc-ai-kit adjudicate`. The number it prints is not an
estimate of anything and should not be quoted as one.

Predictions are saved with the run, so fixing a matcher afterwards and re-scoring costs no
inference: `doc-ai-kit rescore adversarial_zero_shot`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import dspy

from doc_ai_kit.cli import Project
from doc_ai_kit.dataset import load_labels
from doc_ai_kit.evaluate import run_program, score_split, write_run
from doc_ai_kit.program import build_program

DEFAULT_RUN_ID = "adversarial_zero_shot"

logger = logging.getLogger(__name__)


def _get(container: Any, name: str) -> Any:
    """Read a field from a usage record, which arrives as an object or as a dict."""
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def usage_of(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise what a run's calls consumed.

    Counts only replies generated in this run: DSPy reports no usage for a reply served from its
    cache. To measure what a configuration costs, run it with ``--fresh``. Thinking is billed as
    output and counts against max_tokens, which is why it is broken out.
    """
    output = reasoning = input_tokens = truncated = 0
    cost = 0.0
    for entry in history:
        usage = entry.get("usage") or _get(entry.get("response"), "usage")
        if not usage:
            continue
        input_tokens += _get(usage, "prompt_tokens") or 0
        output += _get(usage, "completion_tokens") or 0
        reasoning += _get(_get(usage, "completion_tokens_details"), "reasoning_tokens") or 0
        choices = _get(entry.get("response"), "choices") or []
        if choices and _get(choices[0], "finish_reason") == "length":
            truncated += 1
        cost += entry.get("cost") or 0.0
    return {
        "calls": len(history),
        "input_tokens": input_tokens,
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "truncated_replies": truncated,
        "cost_usd": round(cost, 4),
    }


def main() -> None:
    """Predict every document zero-shot and write the run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="The adversarial project directory")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="Name of the run directory to write")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Bypass DSPy's cache so every reply is generated, and measured, in this run",
    )
    args = parser.parse_args()
    run_id = args.run_id
    if args.fresh:
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

    project = Project(directory=args.project.resolve())
    project.load_customizations()
    project.configure_lm()
    doc_ids = [record.doc_id for record in load_labels(project.data / "labels.jsonl")]
    examples = project.examples_for(doc_ids)
    metric = project.metric()

    lm = dspy.settings.lm
    history_start = len(lm.history)
    program = build_program(project.registry, module_type=project.config.optimization.module)
    predictions = run_program(
        program,
        examples,
        metric,
        num_threads=project.config.optimization.num_threads,
        max_retries=project.config.evaluation.max_retries,
        max_failure_rate=project.config.evaluation.max_failure_rate,
    )
    usage = usage_of(lm.history[history_start:])
    result = score_split(
        project.registry,
        metric,
        examples,
        predictions,
        metadata={
            "run": run_id,
            "usage": usage,
            "split": "all documents -- a matcher check, not an accuracy estimate",
            **project.config.models.describe(),
            "config_hash": project.config.fingerprint(),
        },
        support_floor=project.config.splits.support_floor,
        measurable_floor=project.config.splits.measurable_floor,
    )
    write_run(project.runs / run_id, project.registry, result, examples, predictions)
    print(
        f"{run_id}: aggregate {result.aggregate:.3f} over {len(examples)} documents; "
        f"{usage['output_tokens']:,} output tokens ({usage['reasoning_tokens']:,} thinking), "
        f"{usage['truncated_replies']} truncated, ${usage['cost_usd']:.2f}"
    )
    print(f"Wrote {project.runs / run_id}. Next: doc-ai-kit --project {args.project} adjudicate {run_id}")


if __name__ == "__main__":
    main()
