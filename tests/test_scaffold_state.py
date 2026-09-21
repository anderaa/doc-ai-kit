"""Scaffold creation, CLI wiring and phase derivation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from doc_harness import __version__
from doc_harness.cli import cli, newproject
from doc_harness.scaffold_writer import ScaffoldError, ScaffoldOptions, create_project
from doc_harness.state import derive


@pytest.fixture
def project(tmp_path: Path) -> Path:
    target = tmp_path / "acme"
    create_project(target, ScaffoldOptions(project_name="Acme Contracts"))
    return target


def test_scaffold_writes_the_guidance_layer(project: Path) -> None:
    for relative in (
        "CLAUDE.md",
        "config.yaml",
        "tasks.yaml",
        "decisions.md",
        "docs/protocol.md",
        "custom/__init__.py",
        "programs/baseline.py",
        "README.md",
        "Makefile",
        "pyproject.toml",
        ".python-version",
        ".gitignore",
    ):
        assert (project / relative).exists(), relative
    commands = sorted(path.stem for path in (project / ".claude" / "commands").glob("*.md"))
    assert commands == [
        "audit-labels",
        "close",
        "compile",
        "extract",
        "holdout",
        "make-splits",
        "production",
        "run-baseline",
        "status",
    ]
    skills = sorted(path.name for path in (project / ".claude" / "skills").iterdir())
    assert skills == ["matcher-semantics", "reading-failures", "sample-sizes", "task-types"]


def test_scaffold_pins_an_exact_harness_version(project: Path) -> None:
    """The pin is why newproject has to run outside a project -- and it has to resolve.

    An unpublishable `doc-harness==X.Y.Z` looks like a correct exact pin and fails at the
    project's first `make sync`, which is the worst moment to find out.
    """
    text = (project / "pyproject.toml").read_text(encoding="utf-8")
    assert f"git+https://github.com/anderaa/doc-harness.git@v{__version__}" in text
    assert "doc-harness>=" not in text


@pytest.mark.parametrize(
    "options,expected",
    [
        (ScaffoldOptions(project_name="p", pin_mode="pypi"), f"doc-harness=={__version__}"),
        (
            ScaffoldOptions(project_name="p", pin_mode="git", repo="https://example.invalid/h.git"),
            f"doc-harness @ git+https://example.invalid/h.git@v{__version__}",
        ),
    ],
)
def test_pin_modes(options: ScaffoldOptions, expected: str) -> None:
    assert options.pin == expected


def test_path_pin_is_absolute(tmp_path: Path) -> None:
    options = ScaffoldOptions(project_name="p", pin_mode="path", harness_path=tmp_path)
    assert options.pin == f"doc-harness @ file://{tmp_path.resolve()}"


def test_unknown_pin_mode_fails_loudly() -> None:
    with pytest.raises(ScaffoldError, match="unknown pin mode"):
        ScaffoldOptions(project_name="p", pin_mode="telepathy")


def test_path_pin_needs_a_path() -> None:
    with pytest.raises(ScaffoldError, match="needs --harness-path"):
        ScaffoldOptions(project_name="p", pin_mode="path")


def test_hashes_are_dropped_for_vcs_pins(tmp_path: Path) -> None:
    """pip cannot hash a git checkout, so a git-pinned project must not ask it to."""
    git_project = tmp_path / "git"
    create_project(git_project, ScaffoldOptions(project_name="g", pin_mode="git"))
    assert "--generate-hashes" not in (git_project / "Makefile").read_text(encoding="utf-8")

    pypi_project = tmp_path / "pypi"
    create_project(pypi_project, ScaffoldOptions(project_name="p", pin_mode="pypi"))
    assert "--generate-hashes" in (pypi_project / "Makefile").read_text(encoding="utf-8")


def test_scaffold_substitutes_the_project_name(project: Path) -> None:
    assert (project / "CLAUDE.md").read_text(encoding="utf-8").startswith("# Acme Contracts")
    assert "acme-contracts" in (project / "pyproject.toml").read_text(encoding="utf-8")
    assert (project / ".python-version").read_text(encoding="utf-8").strip() == "acme-contracts"


def test_no_template_suffixes_survive(project: Path) -> None:
    assert not list(project.rglob("*.template"))


def test_no_placeholders_survive(project: Path) -> None:
    for path in project.rglob("*"):
        if path.is_file():
            assert "{{" not in path.read_text(encoding="utf-8"), path


def test_scaffold_creates_the_working_directories(project: Path) -> None:
    for relative in ("data/pdfs", "data/text", "programs/compiled", "runs"):
        assert (project / relative).is_dir(), relative


def test_config_has_no_model_default(project: Path) -> None:
    """A harness that silently picks a model is a harness that silently spends money."""
    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    assert config["models"]["task"] is None
    assert config["models"]["reflection"] is None
    assert config["budget"]["max_usd"] is None


def test_scaffold_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    target = tmp_path / "taken"
    target.mkdir()
    (target / "something.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ScaffoldError, match="not empty"):
        create_project(target, ScaffoldOptions(project_name="x"))
    create_project(target, ScaffoldOptions(project_name="x"), force=True)
    assert (target / "CLAUDE.md").exists()


def test_newproject_command(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(newproject, ["Beta Filings", "--directory", str(tmp_path / "beta")])
    assert result.exit_code == 0, result.output
    assert "pinned to doc-harness" in result.output
    assert (tmp_path / "beta" / "tasks.yaml").exists()


def test_status_on_a_fresh_project(project: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--project", str(project), "status"])
    assert result.exit_code == 0, result.output
    assert "Phase:   scaffolded" in result.output
    assert "models.task is not set" in result.output
    # the scaffold ships every task commented out, and says so plainly
    assert "declares no tasks" in result.output


def test_phase_advances_with_the_files_on_disk(project: Path, fixtures_dir: Path) -> None:
    """Phase is derived from disk, so it cannot drift out of sync with the project."""
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    config = (project / "config.yaml").read_text(encoding="utf-8").replace("task: null", "task: fake/scripted")
    (project / "config.yaml").write_text(config, encoding="utf-8")
    assert derive(project).phase == "tasks declared"

    (project / "data" / "text" / "d01.md").write_text("text", encoding="utf-8")
    (project / "data" / "extraction_manifest.csv").write_text("doc_id\nd01\n", encoding="utf-8")
    assert derive(project).phase == "text extracted"

    (project / "data" / "labels.jsonl").write_text(
        json.dumps({"doc_id": "d01", "labels": {"flag": True, "state": "CA", "number": "A-1"}}) + "\n",
        encoding="utf-8",
    )
    (project / "data" / "annotation_rules.md").write_text("rules", encoding="utf-8")
    assert derive(project).phase == "labels audited"

    (project / "data" / "splits.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "strategy": "toy",
                "assignments": {"train": ["d01"], "val": [], "holdout": []},
            }
        ),
        encoding="utf-8",
    )
    assert derive(project).phase == "splits made"

    (project / "runs" / "baseline_zero_shot").mkdir(parents=True)
    (project / "runs" / "baseline_bootstrap_few_shot").mkdir(parents=True)
    assert derive(project).phase == "baselines recorded"

    (project / "runs" / "exp_001").mkdir(parents=True)
    (project / "runs" / "champion.json").write_text(
        json.dumps({"exp_id": "exp_001", "aggregate": 0.81, "program": "p.json"}), encoding="utf-8"
    )
    state = derive(project)
    assert state.phase == "compiled"
    assert "exp_001 at 0.810" in next(step.detail for step in state.steps if step.name == "compiled")

    (project / "runs" / "holdout").mkdir(parents=True)
    (project / "runs" / "holdout" / ".lock").write_text(
        json.dumps({"opened_at": "now", "opened_by": "test", "overrides": []}), encoding="utf-8"
    )
    assert derive(project).phase == "holdout measured"

    production = project / "runs" / "production"
    production.mkdir(parents=True)
    (production / "outputs.jsonl").write_text("{}\n", encoding="utf-8")
    (production / "qa_report.md").write_text("qa", encoding="utf-8")
    assert derive(project).phase == "production run"

    (project / "REPORT.md").write_text("report", encoding="utf-8")
    state = derive(project)
    assert state.phase == "closed"
    assert state.next_step is None


def test_blind_holdout_is_a_blocker(project: Path, fixtures_dir: Path) -> None:
    """Correcting baseline output on the holdout inflates every number computed against it."""
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (project / "data" / "labels.jsonl").write_text(
        "\n".join(
            json.dumps({"doc_id": f"d{i}", "labels": {"flag": True}, "labeling_mode": "corrected"}) for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    (project / "data" / "splits.json").write_text(
        json.dumps({"seed": 1, "strategy": "toy", "assignments": {"train": ["d0"], "val": [], "holdout": ["d1"]}}),
        encoding="utf-8",
    )
    blockers = "\n".join(derive(project).blockers)
    assert "labeled blind" in blockers
    assert "inflates every number" in blockers


def test_overridden_holdout_is_surfaced(project: Path) -> None:
    holdout = project / "runs" / "holdout"
    holdout.mkdir(parents=True)
    (holdout / ".lock").write_text(
        json.dumps(
            {"opened_at": "then", "opened_by": "first", "overrides": [{"at": "now", "by": "second", "reason": "x"}]}
        ),
        encoding="utf-8",
    )
    assert any("re-opened" in warning for warning in derive(project).warnings)


def test_compile_refuses_without_annotation_rules(project: Path, fixtures_dir: Path) -> None:
    """The gate is code, and the refusal explains itself."""
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--project", str(project), "compile", "--variable", "x"], standalone_mode=False)
    assert result.exit_code != 0
    assert "annotation_rules.md" in str(result.exception)
    assert "fits noise" in str(result.exception)


def test_make_splits_refuses_to_reroll(project: Path, fixtures_dir: Path) -> None:
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (project / "data" / "splits.json").write_text("{}", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--project", str(project), "make-splits"], standalone_mode=False)
    assert result.exit_code != 0
    assert "no longer a holdout" in str(result.exception)


def test_close_assembles_the_report_and_ledger(project: Path, fixtures_dir: Path) -> None:
    """Close writes REPORT.md from the sections each phase left behind."""
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (project / "data" / "labels.jsonl").write_text(
        json.dumps({"doc_id": "d01", "labels": {"flag": True, "state": "CA", "number": "A-1"}}) + "\n",
        encoding="utf-8",
    )
    (project / "data" / "text" / "d01.md").write_text("text", encoding="utf-8")
    (project / "runs").mkdir(exist_ok=True)
    (project / "runs" / "baseline_report.md").write_text("# Baselines\n\nbest 0.5", encoding="utf-8")
    (project / "runs" / "holdout").mkdir(parents=True)
    (project / "runs" / "holdout" / "REPORT_SECTION.md").write_text("# Holdout\n\n0.48", encoding="utf-8")
    (project / "runs" / "production").mkdir(parents=True)
    (project / "runs" / "production" / "qa_report.md").write_text("# Production QA\n\nall passed", encoding="utf-8")

    ledger = project / "ledger.csv"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--project", str(project), "close", "--ledger", str(ledger), "--labeling-hours", "12.5"],
        standalone_mode=False,
    )
    assert result.exit_code == 0, result.output

    report = (project / "REPORT.md").read_text(encoding="utf-8")
    assert report.index("# Baselines") < report.index("# Holdout") < report.index("# Production QA")

    rows = list(csv.DictReader(ledger.open(encoding="utf-8", newline="")))
    assert len(rows) == 1
    assert rows[0]["project"] == project.name
    assert rows[0]["k_labeled"] == "1"
    assert rows[0]["labeling_hours"] == "12.5"
    assert "binary" in rows[0]["task_types"]


def test_close_refuses_with_nothing_to_report(project: Path, fixtures_dir: Path) -> None:
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--project", str(project), "close"], standalone_mode=False)
    assert result.exit_code != 0
    assert "nothing to report yet" in str(result.exception)


def test_baseline_report_is_not_counted_as_a_run(project: Path, fixtures_dir: Path) -> None:
    """A file sitting beside the baseline runs is not itself a baseline run."""
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (project / "runs").mkdir(exist_ok=True)
    (project / "runs" / "baseline_report.md").write_text("# Baselines", encoding="utf-8")
    (project / "runs" / "baseline_zero_shot").mkdir()
    detail = next(step.detail for step in derive(project).steps if step.name == "baselines recorded")
    assert "1 baseline run(s)" in detail
    assert "baseline_report.md" not in detail
