"""Command-line entry points.

``newproject`` is its own console script because it must run outside any project: the thing
it writes is the exact harness pin the project will install. Every other command runs from
inside a project, against that pinned environment.

Commands refuse rather than guess. Every refusal names the gate and why it exists, so the
answer can be a reason rather than a shrug.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from doc_harness import __version__
from doc_harness.adjudicate import adjudicate as run_adjudication
from doc_harness.adjudicate import write_adjudication
from doc_harness.baseline import run_baselines
from doc_harness.config import Config
from doc_harness.dataset import (
    build_examples,
    load_labels,
    load_splits,
    select,
    validate_against_registry,
    write_splits,
)
from doc_harness.evaluate import (
    load_metrics,
    load_predictions,
    null_rates_from_metrics,
    per_task_from_metrics,
    score_split,
    write_run,
)
from doc_harness.extract import extract_corpus, load_texts
from doc_harness.guards import GuardError, require_files
from doc_harness.hooks import load_project_customizations
from doc_harness.metric import build_metric
from doc_harness.optimize import require_champion, run_experiment
from doc_harness.produce import ProductionResult, produce, run_qa, triage, write_outputs, write_qa_report
from doc_harness.program import load_program
from doc_harness.registry import Registry
from doc_harness.report import append_ledger, run_holdout, write_report
from doc_harness.scaffold_writer import HARNESS_REPO, PIN_MODES, ScaffoldOptions, create_project
from doc_harness.splits import (
    SUPPORT_OPTIONS,
    below_floor,
    class_supports,
    make_splits,
    support_floor_prompt,
)
from doc_harness.state import derive

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
        dspy.configure(
            lm=dspy.LM(
                models.require_task(),
                temperature=models.temperature,
                max_tokens=models.max_tokens,
            )
        )
        logger.info("using task model %s", models.task)

    def examples_for(self, doc_ids: Sequence[str]) -> list[Any]:
        """Build DSPy examples for a set of documents."""
        records = select(load_labels(self.data / "labels.jsonl"), doc_ids)
        texts = load_texts(self.data / "text", doc_ids)
        return build_examples(records, texts, registry=self.registry)

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
@click.version_option(__version__, prog_name="doc-harness")
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
    documents = extract_corpus(
        project.data / "pdfs",
        project.data / "text",
        project.data / "extraction_manifest.csv",
        project.config.extraction,
        force=force,
    )
    flagged = [document.doc_id for document in documents if document.ocr_needed]
    click.echo(f"Extracted {len(documents)} document(s) to {project.data / 'text'}.")
    if flagged:
        click.echo(
            f"{len(flagged)} document(s) had a thin or missing text layer: {', '.join(flagged)}.\n"
            "A task that scores zero on these is an extraction problem, not a prompt problem."
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
@click.option("--non-interactive", is_flag=True, help="Refuse rather than prompt for support-floor decisions.")
@click.option("--force", is_flag=True, help="Overwrite an existing splits.json.")
@pass_project
def make_splits_command(project: Project, non_interactive: bool, force: bool) -> None:
    """Create the fixed train/validation/holdout assignment."""
    project.load_customizations()
    splits_path = project.data / "splits.json"
    if splits_path.exists() and not force:
        raise click.ClickException(
            f"{splits_path} already exists. Splits are created once, before any model sees any "
            "document; re-rolling them after the fact means the holdout is no longer a holdout. "
            "Pass --force only if nothing has been run against them yet."
        )
    records = load_labels(project.data / "labels.jsonl")
    floor = project.config.splits.support_floor

    rare = below_floor(class_supports(project.registry, records), floor)
    if rare:
        click.echo(f"{len(rare)} class(es) fall below the support floor of {floor}.\n")
        for support in rare:
            click.echo(support_floor_prompt(support, floor))
            if non_interactive:
                raise click.ClickException(
                    "a class below the support floor needs a human decision; re-run without "
                    "--non-interactive, or record the decision in config.yaml first"
                )
            choice = click.prompt(
                "  choice",
                type=click.Choice([name for name, _ in SUPPORT_OPTIONS]),
                show_choices=True,
            )
            rationale = click.prompt("  rationale", default="", show_default=False)
            project.record_decision(
                f"support floor: {support.task_id}/{support.label}",
                f"- Labeled examples: **{support.count}** ({support.readable_as})\n"
                f"- Choice: **{choice}**\n"
                f"- Rationale: {rationale or '(none given)'}",
            )
            if choice == "report_unmeasured":
                click.echo(
                    f"  Add {support.label!r} to metric.excluded_classes['{support.task_id}'] in config.yaml "
                    "so it leaves the optimization target while staying in the report."
                )
            click.echo("")

    splits = make_splits(project.registry, records, seed=project.config.splits.seed)
    write_splits(splits_path, splits)
    click.echo(
        f"{splits.strategy}: {len(splits.train)} train, {len(splits.val)} val, "
        f"{len(splits.holdout)} holdout, seed {splits.seed}.\nWrote {splits_path}."
    )


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


def _project_baseline_inputs(project: Project, trainset: Sequence[Any]) -> tuple[Any, Any]:
    """Load hand-written demonstrations and instructions from programs/baseline.py."""
    import importlib.util

    path = project.directory / "programs" / "baseline.py"
    if not path.exists():
        return None, None
    spec = importlib.util.spec_from_file_location("project_baseline", path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
        raise click.ClickException(f"no cached text in {text_dir}; run `doc-harness extract` first")
    texts = load_texts(text_dir, doc_ids)

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
@click.option("--cost-usd", type=float, default=None, help="Total spend, for the ledger.")
@click.option("--ledger", type=click.Path(path_type=Path), default=None, help="Path to the shared ledger.csv.")
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

    report_path = write_report(project.directory, sections, title=project.directory.name)
    click.echo(f"Wrote {report_path}.")

    if ledger is not None:
        records = load_labels(project.data / "labels.jsonl")
        append_ledger(
            ledger,
            {
                "closed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "project": project.directory.name,
                "harness_version": __version__,
                "task_types": ",".join(sorted({str(task.type) for task in project.registry})),
                "n_documents": len(list((project.data / "text").glob("*.md"))),
                "k_labeled": len(records),
                "labeling_hours": labeling_hours if labeling_hours is not None else "",
                "cost_usd": cost_usd if cost_usd is not None else "",
            },
        )
        click.echo(f"Appended to {ledger}.")


@click.command()
@click.argument("name")
@click.option("--directory", type=click.Path(path_type=Path), default=None, help="Where to create it.")
@click.option("--harness-version", default=__version__, help="Exact harness version to pin.")
@click.option("--python-version", default="3.12.11", help="Python version for the project's pyenv virtualenv.")
@click.option(
    "--pin-mode",
    type=click.Choice(PIN_MODES),
    default="git",
    help=(
        "How the project pins the harness. git: a tag in the harness repo. "
        "pypi: a package index. path: a local checkout."
    ),
)
@click.option("--repo", default=HARNESS_REPO, help="Harness repository, for --pin-mode git.")
@click.option(
    "--harness-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Local harness checkout, for --pin-mode path.",
)
@click.option("--force", is_flag=True, help="Write into a directory that already has contents.")
@click.version_option(__version__, prog_name="newproject")
def newproject(
    name: str,
    directory: Path | None,
    harness_version: str,
    python_version: str,
    pin_mode: str,
    repo: str,
    harness_path: Path | None,
    force: bool,
) -> None:
    """Create a new doc-harness project pinned to an exact harness version."""
    _configure_logging(verbose=False)
    options = ScaffoldOptions(
        project_name=name,
        harness_version=harness_version,
        python_version=python_version,
        python_requires=".".join(python_version.split(".")[:2]),
        pin_mode=pin_mode,
        repo=repo,
        harness_path=harness_path,
    )
    target = (directory or Path(options.project_slug)).resolve()
    create_project(target, options, force=force)
    click.echo(
        f"Created {target}, pinned to {options.pin}.\n\n"
        "Next:\n"
        f"  cd {target}\n"
        f"  pyenv virtualenv {python_version} {options.project_slug}\n"
        f"  pyenv local {options.project_slug}\n"
        "  pip install pip-tools && make lock && make sync\n"
        "  doc-harness status\n\n"
        "Then read docs/protocol.md once, and CLAUDE.md for the invariants."
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
