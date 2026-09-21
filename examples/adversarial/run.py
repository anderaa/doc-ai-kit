"""Run the zero-shot program over every adversarial document and save the predictions.

This is a matcher check, not an accuracy measurement. It deliberately scores all 24
documents rather than a validation split, because the point is to collect as many real
threshold decisions as possible for `doc-harness adjudicate`. The number it prints is not an
estimate of anything and should not be quoted as one.

Predictions are saved with the run, so fixing a matcher afterwards and re-scoring costs no
inference: `doc-harness rescore adversarial_zero_shot`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from doc_harness.cli import Project
from doc_harness.dataset import load_labels
from doc_harness.evaluate import run_program, score_split, write_run
from doc_harness.program import build_program

RUN_ID = "adversarial_zero_shot"

logger = logging.getLogger(__name__)


def main() -> None:
    """Predict every document zero-shot and write the run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="The adversarial project directory")
    args = parser.parse_args()

    project = Project(directory=args.project.resolve())
    project.load_customizations()
    project.configure_lm()
    doc_ids = [record.doc_id for record in load_labels(project.data / "labels.jsonl")]
    examples = project.examples_for(doc_ids)
    metric = project.metric()

    program = build_program(project.registry, module_type=project.config.optimization.module)
    predictions = run_program(program, examples, metric, num_threads=project.config.optimization.num_threads)
    result = score_split(
        project.registry,
        metric,
        examples,
        predictions,
        metadata={
            "run": RUN_ID,
            "split": "all documents -- a matcher check, not an accuracy estimate",
            "task_model": project.config.models.task,
            "config_hash": project.config.fingerprint(),
        },
        support_floor=project.config.splits.support_floor,
        measurable_floor=project.config.splits.measurable_floor,
    )
    write_run(project.runs / RUN_ID, project.registry, result, examples, predictions)
    print(f"Wrote {project.runs / RUN_ID}. Next: doc-harness --project {args.project} adjudicate {RUN_ID}")


if __name__ == "__main__":
    main()
