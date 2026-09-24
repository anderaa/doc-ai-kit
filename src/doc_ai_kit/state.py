"""Project phase, derived from disk rather than declared.

Static guidance drifts out of sync with a half-finished project: a command file that says
"next, make the splits" is wrong the moment someone makes them, and wrong in the other
direction if they made them and then changed the tasks. So ``status`` never reads a stored
phase. It looks at what is actually on disk -- splits.json, annotation_rules.md, runs/, the
leaderboard, the holdout lock -- and reports what is done, what is next, and what is
blocking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from doc_ai_kit.config import Config, ConfigError
from doc_ai_kit.dataset import DatasetError, load_labels, load_splits
from doc_ai_kit.extract import is_pdf
from doc_ai_kit.guards import read_holdout_lock
from doc_ai_kit.labeling import PLAN_FILE, SHEET_FILE, LabelPlan, load_plan
from doc_ai_kit.registry import Registry, TaskSpecError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    """One phase of a project, and whether it is finished."""

    name: str
    command: str
    done: bool
    detail: str


@dataclass
class ProjectState:
    """What a project has done, what comes next, and what is in the way."""

    project_dir: Path
    steps: list[Step]
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def phase(self) -> str:
        """Return the furthest step completed."""
        done = [step for step in self.steps if step.done]
        return done[-1].name if done else "scaffolded"

    @property
    def next_step(self) -> Step | None:
        """Return the first unfinished step."""
        return next((step for step in self.steps if not step.done), None)

    def to_text(self) -> str:
        """Render the status report for the terminal."""
        lines = [f"Project: {self.project_dir}", f"Phase:   {self.phase}", ""]
        for step in self.steps:
            mark = "x" if step.done else " "
            lines.append(f"  [{mark}] {step.name:<22} {step.detail}")
        lines.append("")
        following = self.next_step
        if following is None:
            lines.append("Next: nothing; the project is closed.")
        else:
            lines.append(f"Next: {following.name} -- run `{following.command}`")
        if self.blockers:
            lines += ["", "Blocking:"]
            lines += [f"  - {blocker}" for blocker in self.blockers]
        if self.warnings:
            lines += ["", "Worth knowing:"]
            lines += [f"  - {warning}" for warning in self.warnings]
        return "\n".join(lines)


def _count(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.is_dir() else 0


def _champion_detail(experiments: list[str], champion: dict[str, object] | None) -> str:
    """Describe the optimization phase: how many experiments ran and what is pinned."""
    if champion is None:
        return f"{len(experiments)} experiment(s); no champion pinned"
    return f"{len(experiments)} experiment(s); champion {champion['exp_id']} at {float(str(champion['aggregate'])):.3f}"


def _sample_detail(plan: LabelPlan | None, has_labels: bool) -> str:
    """Describe the labeling sample."""
    if plan is not None:
        return f"{len(plan.sampled)} of {plan.corpus_size} to label, {len(plan.holdout)} of them holdout (blind)"
    return "labels brought in directly, without a sample" if has_labels else "not drawn"


def _labeled_detail(plan: LabelPlan | None, labeled: set[str]) -> str:
    """Describe labeling progress against the sample."""
    if plan is None:
        return f"{len(labeled)} labeled document(s)"
    holdout_done = len(set(plan.holdout) & labeled)
    detail = (
        f"{len(set(plan.sampled) & labeled)} of {len(plan.sampled)} labeled "
        f"({holdout_done} of {len(plan.holdout)} holdout)"
    )
    return f"{detail}, {len(plan.skipped)} skipped" if plan.skipped else detail


def derive(project_dir: Path) -> ProjectState:
    """Work out where a project stands by looking at its files.

    :param project_dir: The project root
    :returns: The derived state
    """
    data = project_dir / "data"
    runs = project_dir / "runs"
    blockers: list[str] = []
    warnings: list[str] = []

    registry: Registry | None = None
    try:
        registry = Registry.from_yaml(project_dir / "tasks.yaml")
        tasks_detail = f"{len(registry)} task(s) declared"
    except (TaskSpecError, FileNotFoundError) as exc:
        tasks_detail = "not readable"
        blockers.append(f"tasks.yaml: {exc}")

    try:
        config: Config | None = Config.from_yaml(project_dir / "config.yaml")
    except ConfigError as exc:
        config = None
        blockers.append(f"config.yaml: {exc}")
    if config is not None and not config.models.task:
        blockers.append("config.yaml: models.task is not set; the package will not choose a model for you")

    pdf_dir = data / "pdfs"
    pdf_count = sum(1 for path in pdf_dir.iterdir() if is_pdf(path)) if pdf_dir.is_dir() else 0
    text_count = _count(data / "text", "*.md")
    manifest = data / "extraction_manifest.csv"

    labels_path = data / "labels.jsonl"
    label_count = 0
    blind_count = 0
    labeled: set[str] = set()
    if labels_path.exists():
        try:
            records = load_labels(labels_path)
            label_count = len(records)
            labeled = {record.doc_id for record in records}
            blind_count = sum(1 for record in records if record.labeling_mode == "blind")
        except DatasetError as exc:
            blockers.append(f"labels.jsonl: {exc}")

    plan: LabelPlan | None = None
    if (data / PLAN_FILE).exists():
        try:
            plan = load_plan(data / PLAN_FILE)
        except (DatasetError, ValueError) as exc:
            blockers.append(f"{PLAN_FILE}: {exc}")
    sheet = data / SHEET_FILE
    if sheet.exists() and (not labels_path.exists() or sheet.stat().st_mtime > labels_path.stat().st_mtime):
        warnings.append(f"{SHEET_FILE} has changed since the last import; run `doc-ai-kit import-labels`")

    rules = data / "annotation_rules.md"
    splits_path = data / "splits.json"
    splits_detail = "not made"
    if splits_path.exists():
        try:
            splits = load_splits(splits_path)
            splits_detail = (
                f"{splits.strategy}: {len(splits.train)} train, {len(splits.val)} val, "
                f"{len(splits.holdout)} holdout (seed {splits.seed})"
            )
            unlabeled = len(splits.train + splits.val + splits.holdout) - label_count
            if label_count and unlabeled > 0:
                warnings.append(f"{unlabeled} document(s) in the splits are not labeled")
            blind_holdout = blind_count >= len(splits.holdout) if splits.holdout else True
            if label_count and not blind_holdout:
                blockers.append(
                    f"only {blind_count} of {len(splits.holdout)} holdout document(s) are labeled blind. "
                    "Correcting baseline output on the holdout correlates the labels with what is being "
                    "measured and inflates every number"
                )
        except DatasetError as exc:
            blockers.append(f"splits.json: {exc}")

    # directories only: baseline_report.md sits alongside them and is not a run
    baseline_runs = sorted(path.name for path in runs.glob("baseline_*") if path.is_dir()) if runs.is_dir() else []
    experiments = sorted(path.name for path in runs.glob("exp_*")) if runs.is_dir() else []
    champion_path = runs / "champion.json"
    champion = json.loads(champion_path.read_text(encoding="utf-8")) if champion_path.exists() else None
    lock = read_holdout_lock(runs / "holdout")
    outputs = runs / "production" / "outputs.jsonl"
    qa_report = runs / "production" / "qa_report.md"
    report = project_dir / "REPORT.md"

    if config is not None and experiments and len(experiments) >= config.optimization.max_experiments:
        warnings.append(
            f"the experiment budget of {config.optimization.max_experiments} is spent; "
            "raising it is a deliberate, recorded decision"
        )
    if lock is not None and lock.was_overridden:
        warnings.append(f"the holdout has been re-opened {len(lock.overrides)} time(s); this is recorded in the report")

    steps = [
        Step("tasks declared", "edit tasks.yaml", registry is not None, tasks_detail),
        Step(
            "text extracted",
            "doc-ai-kit extract",
            text_count > 0 and manifest.exists(),
            f"{text_count} of {pdf_count} PDF(s) cached" if pdf_count else "no PDFs in data/pdfs",
        ),
        Step(
            "documents sampled",
            "doc-ai-kit sample-labels --count N",
            plan is not None or labels_path.exists(),
            _sample_detail(plan, labels_path.exists()),
        ),
        Step(
            "labels imported",
            "doc-ai-kit label-sheet, then doc-ai-kit import-labels",
            labels_path.exists(),
            _labeled_detail(plan, labeled),
        ),
        Step(
            "labels audited",
            "doc-ai-kit audit-labels",
            rules.exists(),
            f"{label_count} labeled document(s); annotation_rules.md " f"{'written' if rules.exists() else 'missing'}",
        ),
        Step("splits made", "doc-ai-kit make-splits", splits_path.exists(), splits_detail),
        Step(
            "baselines recorded",
            "doc-ai-kit run-baseline",
            len(baseline_runs) >= 2,
            f"{len(baseline_runs)} baseline run(s): {', '.join(baseline_runs) or 'none'}",
        ),
        Step("compiled", "doc-ai-kit compile", champion is not None, _champion_detail(experiments, champion)),
        Step(
            "holdout measured",
            "doc-ai-kit holdout",
            lock is not None,
            f"opened {lock.opened_at} by {lock.opened_by}" if lock else "not opened",
        ),
        Step(
            "production run",
            "doc-ai-kit production",
            outputs.exists() and qa_report.exists(),
            "outputs.jsonl and qa_report.md written" if outputs.exists() else "not run",
        ),
        Step(
            "closed",
            "doc-ai-kit close",
            report.exists(),
            "REPORT.md written" if report.exists() else "REPORT.md not written",
        ),
    ]
    return ProjectState(project_dir=project_dir, steps=steps, blockers=blockers, warnings=warnings)
