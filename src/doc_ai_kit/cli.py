"""Command-line entry points.

``newproject`` is its own console script because it must run outside any project: the thing
it writes is the exact package pin the project will install. Every other command runs from
inside a project, against that pinned environment.

Commands refuse rather than guess. Every refusal names the gate and why it exists, so the
answer can be a reason rather than a shrug.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from doc_ai_kit import __version__
from doc_ai_kit.adjudicate import adjudicate as run_adjudication
from doc_ai_kit.adjudicate import write_adjudication
from doc_ai_kit.baseline import run_baselines
from doc_ai_kit.config import Config, add_excluded_classes
from doc_ai_kit.dataset import (
    build_examples,
    load_labels,
    load_splits,
    select,
    validate_against_registry,
    write_labels,
    write_splits,
)
from doc_ai_kit.evaluate import (
    load_metrics,
    load_predictions,
    null_rates_from_metrics,
    per_task_from_metrics,
    score_split,
    write_run,
)
from doc_ai_kit.extract import extract_corpus, load_texts, source_names
from doc_ai_kit.guards import GuardError, require_files
from doc_ai_kit.hooks import load_project_customizations
from doc_ai_kit.labeling import (
    PLAN_FILE,
    SHEET_FILE,
    draw_sample,
    import_rows,
    load_plan,
    read_sheet,
    write_plan,
    write_sheet,
)
from doc_ai_kit.metric import build_metric
from doc_ai_kit.optimize import read_champion, require_champion, run_experiment
from doc_ai_kit.produce import ProductionResult, produce, run_qa, triage, write_outputs, write_qa_report
from doc_ai_kit.program import build_program, load_program, task_lm
from doc_ai_kit.registry import Registry
from doc_ai_kit.report import (
    NOTES_FILE,
    append_ledger,
    best_baseline,
    holdout_aggregate,
    run_holdout,
    write_prompt,
    write_report,
)
from doc_ai_kit.scaffold_writer import PACKAGE_REPO, PIN_MODES, ScaffoldOptions, create_project, setup_steps
from doc_ai_kit.spend import check_budget, record_spend, totals
from doc_ai_kit.splits import (
    SUPPORT_OPTIONS,
    class_supports,
    holdout_share_for,
    make_splits,
    support_floor_prompt,
)
from doc_ai_kit.splits import below_floor as classes_below_floor
from doc_ai_kit.state import derive

logger = logging.getLogger(__name__)

BASELINE_SECTION = "baseline_report.md"
HOLDOUT_SECTION = "REPORT_SECTION.md"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


@dataclass
class Project:
    """A loaded project: its paths, its registry and its configuration."""

    directory: Path
    _registry: Registry | None = field(default=None, repr=False)
    _config: Config | None = field(default=None, repr=False)

    @property
    def data(self) -> Path:
        """Return the project's data directory."""
        return self.directory / "data"

    @property
    def runs(self) -> Path:
        """Return the project's runs directory."""
        return self.directory / "runs"

    @property
    def registry(self) -> Registry:
        """Return the parsed tasks.yaml, loading it on first use."""
        if self._registry is None:
            self._registry = Registry.from_yaml(self.directory / "tasks.yaml")
        return self._registry

    @property
    def config(self) -> Config:
        """Return the parsed config.yaml, loading it on first use."""
        if self._config is None:
            self._config = Config.from_yaml(self.directory / "config.yaml")
        return self._config

    def load_customizations(self) -> None:
        """Import the project's custom/ package so its hooks register."""
        load_project_customizations(self.directory)

    def metric(self) -> Any:
        """Return the metric, with any report-unmeasured classes excluded from the target."""
        excluded = {task_id: set(labels) for task_id, labels in self.config.metric.excluded_classes.items()}
        return build_metric(
            self.registry,
            excluded_classes=excluded,
            abstention=self.config.metric.abstention,
        )

    def configure_lm(self) -> None:
        """Point DSPy at the task model named in config.yaml."""
        import dspy

        models = self.config.models
        dspy.configure(lm=task_lm(models))
        logger.info(
            "using task model %s (effort %s, thinking %s)",
            models.task,
            models.task_effort or "default",
            models.task_thinking or "default",
        )

    def examples_for(self, doc_ids: Sequence[str]) -> list[Any]:
        """Build DSPy examples for a set of documents."""
        records = select(load_labels(self.data / "labels.jsonl"), doc_ids)
        texts = load_texts(self.data / "text", doc_ids)
        return build_examples(records, texts, registry=self.registry)

    @contextmanager
    def paying(self, label: str, models: Sequence[str] | None = None) -> Iterator[None]:
        """Refuse a paid step the budget cannot cover, and record what it spent.

        The ceiling is checked before the step, because a run under way cannot be unspent.

        :param label: What is running, recorded in runs/spend.json
        :param models: The models it will call; the task model unless given
        """
        from dspy.utils.usage_tracker import track_usage

        called = list(models or [self.config.models.task or ""])
        check_budget(self.runs, self.config.budget, called, label)
        with track_usage() as tracker:
            try:
                yield
            finally:
                spent = record_spend(self.runs, self.config.budget, label, tracker.get_total_tokens())
                if spent and self.config.budget.max_usd is not None:
                    click.echo(f"{label} spent ${spent:.2f}; ${totals(self.runs)['usd']:.2f} of the budget so far.")

    def record_decision(self, heading: str, body: str) -> None:
        """Append a decision to decisions.md, where human choices are kept."""
        path = self.directory / "decisions.md"
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        entry = f"\n## {stamp} - {heading}\n\n{body.rstrip()}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        logger.info("recorded decision: %s", heading)


pass_project = click.make_pass_decorator(Project)


@click.group()
@click.version_option(__version__, prog_name="doc-ai-kit")
@click.option("--project", "project_dir", type=click.Path(path_type=Path), default=".", help="Project directory.")
@click.option("-v", "--verbose", is_flag=True, help="Log at debug level.")
@click.pass_context
def cli(ctx: click.Context, project_dir: Path, verbose: bool) -> None:
    """Run a document classification and extraction project."""
    _configure_logging(verbose)
    ctx.obj = Project(directory=project_dir.resolve())


@cli.command()
@pass_project
def status(project: Project) -> None:
    """Report what is done, what is next, and what is blocking."""
    click.echo(derive(project.directory).to_text())


@cli.command()
@click.option("--force", is_flag=True, help="Re-extract documents that are already cached.")
@pass_project
def extract(project: Project, force: bool) -> None:
    """Extract text from data/pdfs into the cache, and write the manifest."""
    project.load_customizations()
    settings = project.config.extraction.transcription
    if settings.mode != "off" and settings.model:
        check_budget(project.runs, project.config.budget, [settings.model], "transcribing scanned pages")
    documents = extract_corpus(
        project.data / "pdfs",
        project.data / "text",
        project.data / "extraction_manifest.csv",
        project.config.extraction,
        force=force,
    )
    click.echo(f"Extracted {len(documents)} document(s) to {project.data / 'text'}.")
    transcribed = sum(document.transcribed_pages for document in documents)
    if transcribed:
        spent_in = sum(document.transcription_tokens[0] for document in documents)
        spent_out = sum(document.transcription_tokens[1] for document in documents)
        record_spend(
            project.runs,
            project.config.budget,
            "transcription",
            {settings.model or "": {"input_tokens": spent_in, "output_tokens": spent_out}},
        )
        click.echo(
            f"{transcribed} page(s) with no usable text layer were transcribed by "
            f"{project.config.extraction.transcription.model}; this run spent {spent_in:,} input and "
            f"{spent_out:,} output tokens on it (cached pages cost nothing)."
        )
    flagged = [document for document in documents if document.unread_pages]
    if flagged:
        shown = ", ".join(f"{d.doc_id} ({d.unread_pages})" for d in flagged[:10])
        more = f", and {len(flagged) - 10} more" if len(flagged) > 10 else ""
        click.echo(
            f"{len(flagged)} document(s) have scanned pages with no text yet, {sum(d.unread_pages for d in flagged)} "
            f"page(s) in all: {shown}{more}. The unread_pages column of extraction_manifest.csv lists them all.\n"
            "A task that scores zero on these is an extraction problem, not a prompt problem."
        )
        if project.config.extraction.transcription.model is None:
            click.echo(
                "Set extraction.transcription.model in config.yaml (e.g. anthropic/claude-sonnet-5) and run "
                "extract again: only these documents are redone."
            )


@cli.command("sample-labels")
@click.option("--count", type=int, required=True, help="How many documents to label.")
@click.option(
    "--holdout-share",
    type=float,
    default=None,
    help="Fraction of them to hold out. Defaults to the protocol's share for that many documents.",
)
@click.option("--force", is_flag=True, help="Redraw the sample. Refused once any labels exist.")
@pass_project
def sample_labels(project: Project, count: int, holdout_share: float | None, force: bool) -> None:
    """Choose which documents to label, and which of them are the holdout.

    Runs before any model sees a document. The holdout is fixed here so its rows can go out
    empty and be labeled blind, while the rest are labeled by correcting model answers.
    """
    plan_path = project.data / PLAN_FILE
    labels_path = project.data / "labels.jsonl"
    if plan_path.exists() and not force:
        raise click.ClickException(
            f"{plan_path} already exists. The sample is drawn once; redrawing it after looking at "
            "documents lets the choice drift toward easy ones. Pass --force only if nothing has been labeled."
        )
    if plan_path.exists() and labels_path.exists():
        raise click.ClickException(
            f"{labels_path} already holds labels made against the current sample, so it cannot be redrawn. "
            "Move labels.jsonl aside deliberately first, and say why in decisions.md."
        )
    doc_ids = sorted(path.stem for path in (project.data / "text").glob("*.md"))
    share = holdout_share if holdout_share is not None else holdout_share_for(count)
    plan = draw_sample(doc_ids, count, share, seed=project.config.splits.seed)
    write_plan(plan_path, plan)
    project.record_decision(
        "labeling sample",
        f"- Documents to label: **{len(plan.sampled)}** of {plan.corpus_size}, drawn at random (seed {plan.seed})\n"
        f"- Holdout: **{len(plan.holdout)}**, drawn at random from the sample, labeled blind\n"
        f"- To correct from model answers: **{len(plan.to_correct)}**",
    )
    click.echo(
        f"Sampled {len(plan.sampled)} of {plan.corpus_size} document(s): {len(plan.holdout)} holdout "
        f"(labeled blind), {len(plan.to_correct)} to correct from model answers.\nWrote {plan_path}.\n\n"
        "Next: doc-ai-kit label-sheet"
    )


@cli.command("label-sheet")
@click.option(
    "--prefill/--no-prefill",
    default=None,
    help=(
        "Required choice. --prefill fills rows outside the holdout with the model's answers, to correct: "
        "faster, and it costs model calls. --no-prefill leaves every row empty, to label from scratch."
    ),
)
@click.option("--live", is_flag=True, help="Prefill with live calls, not the Batch API: faster, twice the price.")
@click.option("--force", is_flag=True, help="Rewrite an existing sheet, keeping only what has been imported.")
@pass_project
def label_sheet(project: Project, prefill: bool | None, live: bool, force: bool) -> None:
    """Write data/labels.xlsx: one tab, one row per sampled document, one column per task.

    With --prefill, rows outside the holdout hold the model's answers, to be corrected. Holdout
    rows are always empty, and the model is never run on them. Rows already imported keep
    their labels.
    """
    project.load_customizations()
    plan_path = project.data / PLAN_FILE
    plan = load_plan(plan_path)
    sheet_path = project.data / SHEET_FILE
    if sheet_path.exists() and not force:
        raise click.ClickException(
            f"{sheet_path} already exists and may hold work that is not imported yet. Run "
            "`doc-ai-kit import-labels` first; then --force rewrites it from the imported labels."
        )

    labels_path = project.data / "labels.jsonl"
    records = load_labels(labels_path) if labels_path.exists() else []
    values: dict[str, dict[str, Any]] = {record.doc_id: dict(record.labels) for record in records}
    notes = {record.doc_id: record.notes for record in records}
    for doc_id, reason in plan.skipped.items():
        notes[doc_id] = f"skip: {reason}"

    to_prefill = [doc_id for doc_id in plan.to_correct if doc_id not in values and doc_id not in plan.skipped]
    if to_prefill and prefill is None:
        raise click.ClickException(
            "choose --prefill or --no-prefill. --prefill runs the model on the rows outside the holdout and "
            "fills them in for you to correct: two to three times faster to label, and it costs model calls. "
            "--no-prefill leaves every row empty, so every label is your own judgment from the document. "
            "Holdout rows are empty either way."
        )
    if to_prefill and prefill:
        values.update(_prefill(project, plan.holdout, to_prefill, live))
        # recorded before the sheet exists: once a row shows model answers, it is labeled by correction
        plan.prefilled = sorted(set(plan.prefilled) | {doc_id for doc_id in to_prefill if doc_id in values})
        write_plan(plan_path, plan)

    write_sheet(
        sheet_path,
        project.registry,
        plan,
        values,
        notes,
        project.data / "text",
        file_names=source_names(project.data / "extraction_manifest.csv"),
    )
    blank = sum(1 for doc_id in plan.to_correct if doc_id not in values and doc_id not in plan.skipped)
    shaded = len(plan.holdout) + blank
    click.echo(
        f"Wrote {sheet_path}: {len(plan.sampled)} row(s), one per document to label. {shaded} shaded row(s) start "
        "empty, to label from the document alone"
        + (
            f"; the other {len(plan.sampled) - shaded} hold the model's answers to correct"
            if shaded < len(plan.sampled)
            else ""
        )
        + ".\n\nFill in every row in Excel or Google Sheets: a blank cell means the document gives no answer, "
        "and a note of 'skip: <reason>' sets a document aside. Then run: doc-ai-kit import-labels"
    )


def _prefill(project: Project, holdout: Sequence[str], doc_ids: Sequence[str], live: bool) -> dict[str, dict[str, Any]]:
    """Run the zero-shot program over documents to be corrected, and return its answers."""
    from doc_ai_kit.produce import produce as run_program_over

    shown = sorted(set(doc_ids) & set(holdout))
    if shown:
        raise GuardError(f"refusing to run the model on holdout document(s): {', '.join(shown)}")
    project.configure_lm()
    config = project.config
    if live:
        config = config.model_copy(update={"production": config.production.model_copy(update={"use_batch_api": False})})
    module = _load_baseline_module(project)
    instructions = module.starting_instructions() if module and hasattr(module, "starting_instructions") else None
    program = build_program(project.registry, config.optimization.module, instructions=instructions or None)
    click.echo(f"Prefilling {len(doc_ids)} document(s) with {config.models.task}; the holdout is not sent.")
    with project.paying(f"label-sheet prefill ({len(doc_ids)} documents)"):
        outcomes = run_program_over(
            project.registry,
            config,
            program,
            load_texts(project.data / "text", doc_ids),
            project.directory,
            run_dir=project.runs / "prelabel",
        )
    failed = [outcome.doc_id for outcome in outcomes if not outcome.ok]
    if failed:
        click.echo(f"{len(failed)} document(s) got no usable answer and are left empty: {', '.join(failed)}")
    return {outcome.doc_id: outcome.values for outcome in outcomes if outcome.ok}


@cli.command("import-labels")
@click.option(
    "--sheet",
    type=click.Path(path_type=Path),
    default=None,
    help="The sheet to read: .xlsx, or a .csv export of it. Defaults to data/labels.xlsx.",
)
@click.option("--force", is_flag=True, help="Write even though documents labeled before would lose their labels.")
@pass_project
def import_labels(project: Project, sheet: Path | None, force: bool) -> None:
    """Check the finished sheet and write data/labels.jsonl.

    The sheet is taken whole: every sampled document needs a finished row. Every cell is
    checked first and every problem is listed at once; nothing is written unless all pass.
    """
    project.load_customizations()
    plan_path = project.data / PLAN_FILE
    plan = load_plan(plan_path)
    sheet_path = sheet or project.data / SHEET_FILE
    rows = read_sheet(sheet_path, project.registry)
    sampled = set(plan.sampled)
    texts = load_texts(project.data / "text", sorted({row.doc_id for row in rows} & sampled))
    result = import_rows(project.registry, plan, rows, texts)

    if result.warnings:
        click.echo(f"{len(result.warnings)} warning(s):")
        for warning in result.warnings:
            click.echo(f"  - {warning}")
    if not result.ok:
        click.echo(f"{len(result.problems)} problem(s); nothing was written:")
        for problem in result.problems:
            click.echo(f"  - {problem}")
        raise SystemExit(1)
    if not result.records:
        raise click.ClickException(f"every row of {sheet_path.name} is skipped; there is nothing to import")

    labels_path = project.data / "labels.jsonl"
    imported = {record.doc_id for record in result.records}
    splits_path = project.data / "splits.json"
    if splits_path.exists():
        splits = load_splits(splits_path)
        unlabeled = sorted(set(splits.train + splits.val + splits.holdout) - imported)
        if unlabeled:
            raise click.ClickException(
                f"{len(unlabeled)} document(s) in splits.json would have no label: {', '.join(unlabeled[:10])}. "
                "Label them rather than skipping them."
            )
    if labels_path.exists() and not force:
        lost = sorted({record.doc_id for record in load_labels(labels_path)} - imported)
        if lost:
            raise click.ClickException(
                f"{len(lost)} document(s) labeled before are not labeled in this sheet, and would lose their "
                f"labels: {', '.join(lost[:10])}. Pass --force if that is intended."
            )

    order = {doc_id: index for index, doc_id in enumerate(plan.sampled)}
    write_labels(labels_path, sorted(result.records, key=lambda record: order[record.doc_id]))
    plan.skipped = result.skipped
    write_plan(plan_path, plan)

    blind = sum(1 for record in result.records if record.labeling_mode == "blind")
    holdout_done = len(set(plan.holdout) & imported)
    click.echo(
        f"Wrote {labels_path}: {len(result.records)} document(s), {len(result.records) - blind} corrected and "
        f"{blind} blind.\nHoldout: {holdout_done} of {len(plan.holdout)} labeled."
        + (f"\n{len(result.skipped)} skipped." if result.skipped else "")
        + "\n\nNext: doc-ai-kit audit-labels"
    )


@cli.command("audit-labels")
@pass_project
def audit_labels(project: Project) -> None:
    """Check labels against the declared tasks and report per-class support."""
    project.load_customizations()
    records = load_labels(project.data / "labels.jsonl")
    problems = validate_against_registry(project.registry, records)
    if problems:
        click.echo(f"{len(problems)} problem(s):")
        for problem in problems:
            click.echo(f"  - {problem}")
    else:
        click.echo(f"{len(records)} labeled document(s); no undeclared tasks and no out-of-enum values.")

    blind = sum(1 for record in records if record.labeling_mode == "blind")
    click.echo(f"\nLabeling mode: {blind} blind, {len(records) - blind} corrected from baseline output.")

    click.echo("\nPer-class support:")
    floor = project.config.splits.support_floor
    for task_id, supports in class_supports(project.registry, records).items():
        click.echo(f"  {task_id}")
        for label, support in sorted(supports.items()):
            mark = " " if support.count >= floor else "!"
            click.echo(f"   {mark} {label:<20} {support.count:>4}  {support.readable_as}")

    rules = project.data / "annotation_rules.md"
    if not rules.exists():
        click.echo(
            f"\n{rules} is not written. compile will refuse without it: a task whose labeling rule "
            "was never written down cannot be scored consistently."
        )
    if problems:
        raise SystemExit(1)


@cli.command("make-splits")
@click.option(
    "--below-floor",
    type=click.Choice([name for name, _ in SUPPORT_OPTIONS]),
    default=None,
    help="Answer every class below the support floor the same way, instead of one prompt each.",
)
@click.option("--rationale", default="", help="The reason recorded with --below-floor.")
@click.option("--non-interactive", is_flag=True, help="Refuse rather than prompt for support-floor decisions.")
@click.option("--force", is_flag=True, help="Overwrite an existing splits.json.")
@pass_project
def make_splits_command(
    project: Project, below_floor: str | None, rationale: str, non_interactive: bool, force: bool
) -> None:
    """Create the fixed train/validation/holdout assignment."""
    project.load_customizations()
    splits_path = project.data / "splits.json"
    if splits_path.exists() and not force:
        raise click.ClickException(
            f"{splits_path} already exists. Splits are created once; re-rolling them after the "
            "fact means the holdout is no longer a holdout. "
            "Pass --force only if nothing has been run against them yet."
        )
    records = load_labels(project.data / "labels.jsonl")
    floor = project.config.splits.support_floor
    holdout = _planned_holdout(project, records)

    rare = classes_below_floor(class_supports(project.registry, records), floor)
    unmeasured: dict[str, list[str]] = {}
    if rare:
        click.echo(f"{len(rare)} class(es) fall below the support floor of {floor}.\n")
        if below_floor:
            # one answer for all of them: sixty prompts produce sixty blank rationales
            for support in rare:
                click.echo(f"  {support.task_id}/{support.label}: {support.count} example(s), {support.readable_as}")
            project.record_decision(
                f"support floor: {len(rare)} class(es), answered together",
                f"- Choice for every class below the floor of {floor}: **{below_floor}**\n"
                f"- Rationale: {rationale or '(none given)'}\n"
                + "\n".join(f"  - {s.task_id}/{s.label}: {s.count} example(s)" for s in rare),
            )
            if below_floor == "report_unmeasured":
                for support in rare:
                    unmeasured.setdefault(support.task_id, []).append(support.label)
        else:
            for support in rare:
                click.echo(support_floor_prompt(support, floor))
                if non_interactive:
                    raise click.ClickException(
                        "a class below the support floor needs a human decision; re-run without "
                        "--non-interactive, answer them together with --below-floor, or record the "
                        "decision in config.yaml first"
                    )
                choice = click.prompt(
                    "  choice",
                    type=click.Choice([name for name, _ in SUPPORT_OPTIONS]),
                    show_choices=True,
                )
                reason = click.prompt("  rationale", default="", show_default=False)
                project.record_decision(
                    f"support floor: {support.task_id}/{support.label}",
                    f"- Labeled examples: **{support.count}** ({support.readable_as})\n"
                    f"- Choice: **{choice}**\n"
                    f"- Rationale: {reason or '(none given)'}",
                )
                if choice == "report_unmeasured":
                    unmeasured.setdefault(support.task_id, []).append(support.label)
                click.echo("")
    if unmeasured:
        # written into config.yaml, not printed for a human to paste: a decision that never
        # reaches the file is a decision that did not happen
        merged = add_excluded_classes(project.directory / "config.yaml", unmeasured)
        counted = sum(len(labels) for labels in unmeasured.values())
        click.echo(
            f"Recorded {counted} class(es) in metric.excluded_classes in config.yaml: they leave the "
            f"optimization target and stay in the report. Now excluded: "
            + "; ".join(f"{task_id} [{', '.join(labels)}]" for task_id, labels in sorted(merged.items()))
            + "\n"
        )

    splits = make_splits(project.registry, records, seed=project.config.splits.seed, holdout=holdout)
    write_splits(splits_path, splits)
    moved = sorted(set(holdout or []) - set(splits.holdout))
    if moved:
        click.echo(
            f"{len(moved)} holdout document(s) moved to train, because they carried a class no training "
            f"document had: {', '.join(moved)}."
        )
    click.echo(
        f"{splits.strategy}: {len(splits.train)} train, {len(splits.val)} val, "
        f"{len(splits.holdout)} holdout, seed {splits.seed}.\nWrote {splits_path}."
    )


def _planned_holdout(project: Project, records: Sequence[Any]) -> list[str] | None:
    """Return the holdout drawn by sample-labels, refusing if part of it went unlabeled.

    Projects that brought their own labels have no plan, and are split as before.
    """
    plan_path = project.data / PLAN_FILE
    if not plan_path.exists():
        return None
    plan = load_plan(plan_path)
    labeled = {record.doc_id for record in records}
    unlabeled = [doc_id for doc_id in plan.holdout if doc_id not in labeled and doc_id not in plan.skipped]
    if unlabeled:
        raise click.ClickException(
            f"{len(unlabeled)} holdout document(s) are neither labeled nor skipped: {', '.join(unlabeled)}. "
            "Leaving the hard ones out quietly makes the holdout easier than the corpus. Label them, or "
            "skip them in the sheet with a reason."
        )
    skipped = sorted(set(plan.holdout) & set(plan.skipped))
    if skipped:
        click.echo(f"{len(skipped)} holdout document(s) were skipped and are left out: {', '.join(skipped)}.")
        project.record_decision(
            "holdout documents skipped",
            "\n".join(f"- {doc_id}: {plan.skipped[doc_id]}" for doc_id in skipped),
        )
    waiting = [doc_id for doc_id in plan.to_correct if doc_id not in labeled and doc_id not in plan.skipped]
    if waiting:
        click.echo(f"{len(waiting)} sampled document(s) outside the holdout are not labeled and are left out.")
    return [doc_id for doc_id in plan.holdout if doc_id in labeled]


@cli.command("run-baseline")
@pass_project
def run_baseline_command(project: Project) -> None:
    """Record the three baselines a compiled program has to beat."""
    project.load_customizations()
    project.configure_lm()
    splits = load_splits(project.data / "splits.json")
    trainset = project.examples_for(splits.train)
    valset = project.examples_for(splits.val)

    demos, instructions = _project_baseline_inputs(project, trainset)
    with project.paying("run-baseline"):
        report = run_baselines(
            project.registry,
            project.config,
            project.metric(),
            trainset=trainset,
            valset=valset,
            runs_dir=project.runs,
            demos=demos,
            instructions=instructions,
        )
    project.runs.mkdir(parents=True, exist_ok=True)
    (project.runs / BASELINE_SECTION).write_text(report.to_markdown(), encoding="utf-8")
    click.echo(report.to_markdown())


def _load_baseline_module(project: Project) -> Any:
    """Import programs/baseline.py, or return None if the project has none."""
    import importlib.util

    path = project.directory / "programs" / "baseline.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("project_baseline", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project_baseline_inputs(project: Project, trainset: Sequence[Any]) -> tuple[Any, Any]:
    """Load hand-written demonstrations and instructions from programs/baseline.py."""
    module = _load_baseline_module(project)
    if module is None:
        return None, None
    demos = module.hand_written_demos(trainset) if hasattr(module, "hand_written_demos") else None
    instructions = module.starting_instructions() if hasattr(module, "starting_instructions") else None
    return (demos or None), (instructions or None)


@cli.command()
@click.option("--variable", required=True, help="The single thing this experiment changes.")
@click.option("--no-branch", is_flag=True, help="Start from scratch rather than from the pinned champion.")
@pass_project
def compile(project: Project, variable: str, no_branch: bool) -> None:
    """Run one optimization experiment against the validation split."""
    project.load_customizations()
    require_files(
        [project.data / "annotation_rules.md"],
        because=(
            "a task whose labeling rule was never written down cannot be scored consistently, "
            "so optimizing against it fits noise"
        ),
    )
    require_files(
        [project.data / "splits.json"],
        because="without fixed splits there is no validation split to optimize against, only the data",
    )
    project.configure_lm()
    splits = load_splits(project.data / "splits.json")
    models = [project.config.models.task or "", project.config.models.reflection or ""]
    with project.paying(f"compile ({project.config.optimization.optimizer})", models=models):
        record, _program, result = run_experiment(
            project.registry,
            project.config,
            project.metric(),
            trainset=project.examples_for(splits.train),
            valset=project.examples_for(splits.val),
            project_dir=project.directory,
            variable=variable,
            branch_from_champion=not no_branch,
        )
    click.echo(f"{record.exp_id}: aggregate {result.aggregate:.3f} on validation ({record.rollouts} rollouts)")
    click.echo("Per task:")
    for task_id, value in result.per_task_primary.items():
        click.echo(f"  {task_id:<24} {value:.3f}")
    for note in record.notes:
        click.echo(f"  note: {note}")


@cli.command()
@click.option("--override", is_flag=True, help="Deliberately evaluate the holdout a second time.")
@click.option("--reason", default="", help="Why the override was taken; recorded in the report.")
@pass_project
def holdout(project: Project, override: bool, reason: str) -> None:
    """Spend the one-shot holdout measurement."""
    project.load_customizations()
    project.configure_lm()
    champion = require_champion(project.runs)
    splits = load_splits(project.data / "splits.json")

    records = select(load_labels(project.data / "labels.jsonl"), splits.holdout)
    corrected = [record.doc_id for record in records if record.labeling_mode != "blind"]
    if corrected:
        raise click.ClickException(
            f"{len(corrected)} holdout document(s) are not labeled blind: {', '.join(corrected[:5])}. "
            "Correcting baseline output on the holdout correlates the labels with what is being "
            "measured, and inflates every number computed against them."
        )

    program = load_program(project.registry, Path(champion["program"]), module_type=project.config.optimization.module)
    metric = project.metric()
    recorded = load_metrics(project.runs / str(champion["exp_id"]) / "metrics.json")
    with project.paying("holdout"):
        report = run_holdout(
            project.registry,
            project.config,
            metric,
            program,
            project.examples_for(splits.holdout),
            project.directory,
            validation_aggregate=float(recorded["aggregate"]["score"]),
            validation_per_task=per_task_from_metrics(recorded),
            opened_by=f"holdout command ({champion['exp_id']})",
            override=override,
            reason=reason,
        )
    (project.runs / "holdout" / HOLDOUT_SECTION).write_text(report.to_markdown(), encoding="utf-8")
    project.record_decision(
        "holdout reading",
        f"- Champion: **{champion['exp_id']}**\n"
        f"- Holdout aggregate: **{report.result.aggregate:.3f}** against "
        f"{report.validation_aggregate:.3f} on validation\n"
        f"- Reading: **{report.reading}**\n"
        f"- Action: **{report.action}**",
    )
    click.echo(report.to_markdown())


@cli.command()
@click.option("--no-resume", is_flag=True, help="Ignore checkpoints and re-run every document.")
@pass_project
def production(project: Project, no_resume: bool) -> None:
    """Run the compiled program over the full corpus and gate the results."""
    project.load_customizations()
    project.configure_lm()
    champion = require_champion(project.runs)
    require_files(
        [project.runs / "holdout" / ".lock"],
        because="production comes after the holdout has been measured and read",
    )
    program = load_program(project.registry, Path(champion["program"]), module_type=project.config.optimization.module)

    text_dir = project.data / "text"
    doc_ids = sorted(path.stem for path in text_dir.glob("*.md"))
    if not doc_ids:
        raise click.ClickException(f"no cached text in {text_dir}; run `doc-ai-kit extract` first")
    texts = load_texts(text_dir, doc_ids)

    with project.paying("production"):
        outcomes = produce(project.registry, project.config, program, texts, project.directory, resume=not no_resume)

    records = load_labels(project.data / "labels.jsonl")
    recorded = load_metrics(project.runs / str(champion["exp_id"]) / "metrics.json")
    checks = run_qa(
        project.registry,
        project.config,
        outcomes,
        expected_documents=len(doc_ids),
        reference_null_rates=null_rates_from_metrics(recorded),
        reference_rows=[record.labels for record in records],
    )
    routes = triage(
        project.registry,
        project.config,
        outcomes,
        manifest_path=project.data / "extraction_manifest.csv",
        seed=project.config.splits.seed,
    )
    result = ProductionResult(outcomes=outcomes, checks=checks, triage=routes)
    production_dir = project.runs / "production"
    write_outputs(production_dir / "outputs.jsonl", outcomes)
    write_qa_report(production_dir / "qa_report.md", project.registry, result, expected_documents=len(doc_ids))

    click.echo((production_dir / "qa_report.md").read_text(encoding="utf-8"))
    if not result.all_passed:
        raise SystemExit(1)


@cli.command()
@click.argument("run_id")
@pass_project
def rescore(project: Project, run_id: str) -> None:
    """Re-score a finished run from its saved predictions, without paying for inference.

    Use this after fixing a normalizer or matcher: the numbers move, the model is never
    called, and the corrected run replaces the wrong one in place.
    """
    project.load_customizations()
    run_dir = project.runs / run_id
    predictions = load_predictions(run_dir / "predictions.jsonl")
    previous = load_metrics(run_dir / "metrics.json")
    doc_ids = list(predictions)

    golds = project.examples_for(doc_ids)
    metric = project.metric()
    result = score_split(
        project.registry,
        metric,
        golds,
        [predictions[doc_id] for doc_id in doc_ids],
        metadata={**previous.get("metadata", {}), "rescored_at": datetime.now(UTC).isoformat(timespec="seconds")},
        support_floor=project.config.splits.support_floor,
        measurable_floor=project.config.splits.measurable_floor,
    )
    write_run(run_dir, project.registry, result, golds)
    before = float(previous["aggregate"]["score"])
    click.echo(f"{run_id}: aggregate {before:.3f} -> {result.aggregate:.3f} (no inference)")
    for task_id, value in result.per_task_primary.items():
        was = float(previous["aggregate"]["per_task_primary"].get(task_id, 0.0))
        marker = "" if abs(was - value) < 1e-9 else f"  (was {was:.3f})"
        click.echo(f"  {task_id:<24} {value:.3f}{marker}")


@cli.command()
@click.argument("run_id")
@pass_project
def adjudicate(project: Project, run_id: str) -> None:
    """List every decision a threshold made in a run, for a human to check.

    Reads the run's saved predictions, so it costs no inference. Writes adjudication.md into
    the run directory.
    """
    project.load_customizations()
    run_dir = project.runs / run_id
    predictions = load_predictions(run_dir / "predictions.jsonl")
    doc_ids = list(predictions)
    golds = project.examples_for(doc_ids)
    report = run_adjudication(
        project.registry,
        project.metric(),
        golds,
        [predictions[doc_id] for doc_id in doc_ids],
        run_id=run_id,
    )
    path = run_dir / "adjudication.md"
    write_adjudication(path, report)
    click.echo(
        f"{run_id}: {len(report.decisions)} close call(s) and {len(report.errors)} other error(s) "
        f"across {report.n_examples} documents. Wrote {path}."
    )


@cli.command()
@click.option("--labeling-hours", type=float, default=None, help="Hours spent labeling, for the ledger.")
@click.option("--cost-usd", type=float, default=None, help="Total spend; taken from runs/spend.json if omitted.")
@click.option(
    "--ledger",
    type=click.Path(path_type=Path),
    default=lambda: os.environ.get("DOC_HARNESS_LEDGER"),
    help="Path to the shared ledger.csv; defaults to $DOC_HARNESS_LEDGER. Without it, no row is written.",
)
@pass_project
def close(project: Project, labeling_hours: float | None, cost_usd: float | None, ledger: Path | None) -> None:
    """Write REPORT.md and append this project to the shared ledger."""
    project.load_customizations()
    sections: list[str] = []
    baseline_section = project.runs / BASELINE_SECTION
    holdout_section = project.runs / "holdout" / HOLDOUT_SECTION
    qa_section = project.runs / "production" / "qa_report.md"
    for path in (baseline_section, holdout_section, qa_section):
        if path.exists():
            sections.append(path.read_text(encoding="utf-8"))
        else:
            logger.warning("%s does not exist; the report will not include that section", path)
    if not sections:
        raise click.ClickException("nothing to report yet: run the baselines, the holdout and production first")

    champion = read_champion(project.runs)
    if champion is not None and Path(champion["program"]).exists():
        program = load_program(
            project.registry,
            Path(champion["program"]),
            module_type=str(champion.get("module") or project.config.optimization.module),
        )
        click.echo(f"Wrote {write_prompt(project.directory, project.registry, program)}.")
    else:
        click.echo("No champion is pinned, so PROMPT.md was not written.")

    report_path = write_report(project.directory, sections, title=project.directory.name)
    click.echo(f"Wrote {report_path}, with a link to every artifact this project produced.")
    if not (project.directory / NOTES_FILE).exists():
        click.echo(
            f"REPORT.md is regenerated on every close, so anything written into it by hand is lost. "
            f"Put such notes in {NOTES_FILE} and close appends them."
        )

    if ledger is None:
        click.echo(
            "No row was appended to any ledger: pass --ledger path/to/ledger.csv to record this project "
            "alongside the others, or set DOC_HARNESS_LEDGER."
        )
        return
    records = load_labels(project.data / "labels.jsonl")
    baseline = best_baseline(project.runs)
    notes = f"champion {champion['exp_id']}" if champion else ""
    if baseline:
        # validation, because the baselines are never measured on the holdout
        notes = f"{notes}; best baseline {baseline[0]} {baseline[1]:.3f} on validation".lstrip("; ")
    append_ledger(
        ledger,
        {
            "closed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "project": project.directory.name,
            "package_version": __version__,
            "task_types": ",".join(sorted({str(task.type) for task in project.registry})),
            "n_documents": len(list((project.data / "text").glob("*.md"))),
            "k_labeled": len(records),
            # measured, not remembered: the holdout reading and the spend ledger
            "compiled_holdout": _blank_if_none(holdout_aggregate(project.runs)),
            "baseline_holdout": "",
            "labeling_hours": labeling_hours if labeling_hours is not None else "",
            "cost_usd": cost_usd if cost_usd is not None else (totals(project.runs)["usd"] or ""),
            "notes": notes,
        },
    )
    click.echo(f"Appended to {ledger}.")


def _blank_if_none(value: float | None) -> Any:
    """Return a number for the ledger, or an empty cell when it was never measured."""
    return "" if value is None else round(value, 4)


@click.command()
@click.argument("name")
@click.option("--directory", type=click.Path(path_type=Path), default=None, help="Where to create it.")
@click.option("--package-version", default=__version__, help="Exact package version to pin.")
@click.option("--python-version", default="3.12.11", help="Python version for the project's pyenv virtualenv.")
@click.option(
    "--pin-mode",
    type=click.Choice(PIN_MODES),
    default="git",
    help=(
        "How the project pins the package. git: a tag in the package repo. "
        "pypi: a package index. path: a local checkout."
    ),
)
@click.option("--repo", default=PACKAGE_REPO, help="Package repository, for --pin-mode git.")
@click.option(
    "--package-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Local package checkout, for --pin-mode path.",
)
@click.option("--force", is_flag=True, help="Write into a directory that already has contents.")
@click.version_option(__version__, prog_name="newproject")
def newproject(
    name: str,
    directory: Path | None,
    package_version: str,
    python_version: str,
    pin_mode: str,
    repo: str,
    package_path: Path | None,
    force: bool,
) -> None:
    """Create a new doc-ai-kit project pinned to an exact package version."""
    _configure_logging(verbose=False)
    options = ScaffoldOptions(
        project_name=name,
        package_version=package_version,
        python_version=python_version,
        python_requires=".".join(python_version.split(".")[:2]),
        pin_mode=pin_mode,
        repo=repo,
        package_path=package_path,
    )
    target = (directory or Path(options.project_slug)).resolve()
    create_project(target, options, force=force)
    steps = "\n".join(f"  {step}" for step in setup_steps(target, options.project_slug, python_version))
    click.echo(
        f"Created {target}, pinned to {options.pin}.\n\n"
        f"Next:\n{steps}\n\n"
        "The last command starts Claude Code in the project. It reads the project's guidance, "
        "checks where the project stands, and tells you what to do first.\n"
        "docs/protocol.md explains the whole sequence, and CLAUDE.md lists what must not happen."
    )


def main() -> None:
    """Run the project CLI, turning gate refusals into clean errors."""
    try:
        cli(standalone_mode=False)
    except (GuardError, click.ClickException) as exc:
        click.echo(f"Refused: {exc}", err=True)
        raise SystemExit(2) from exc
    except click.Abort:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
