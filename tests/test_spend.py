"""The spend ledger and the budget ceiling.

Reported from the first real project: budget.max_usd was recorded with every run and
enforced nowhere, and the overspend was worked out afterwards from token counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from doc_ai_kit.config import Config
from doc_ai_kit.guards import GuardError
from doc_ai_kit.spend import check_budget, load, record_spend, spent_usd, totals

SONNET = "anthropic/claude-sonnet-5"
USAGE = {SONNET: {"prompt_tokens": 1_000_000, "completion_tokens": 100_000}}


def _budget(**overrides: Any) -> Any:
    payload: dict[str, Any] = {
        "models": {"task": SONNET, "reflection": "anthropic/claude-opus-5"},
        "budget": {"prices": {SONNET: {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}, **overrides},
    }
    return Config.from_mapping(payload).budget


def test_spend_is_priced_from_the_project_s_own_prices(tmp_path: Path) -> None:
    added = record_spend(tmp_path, _budget(), "compile exp_001", USAGE)
    assert added == pytest.approx(3.0 + 1.5)
    assert spent_usd(tmp_path) == pytest.approx(4.5)
    entry = load(tmp_path)[0]
    assert (entry.label, entry.model, entry.route) == ("compile exp_001", SONNET, "live")
    assert (entry.input_tokens, entry.output_tokens) == (1_000_000, 100_000)


def test_the_batch_route_is_billed_at_its_multiplier(tmp_path: Path) -> None:
    record_spend(tmp_path, _budget(), "production (batch)", USAGE, route="batch")
    assert spent_usd(tmp_path) == pytest.approx(2.25)


def test_tokens_are_recorded_even_without_a_price(tmp_path: Path) -> None:
    """A project with no ceiling still gets the token counts its report quotes."""
    budget = Config.from_mapping({"models": {"task": SONNET}}).budget
    assert record_spend(tmp_path, budget, "compile", USAGE) == 0.0
    assert totals(tmp_path)["input_tokens"] == 1_000_000
    assert totals(tmp_path)["models_without_a_declared_price"] == [SONNET]


def test_a_spent_budget_refuses_the_next_step(tmp_path: Path) -> None:
    budget = _budget(max_usd=5.0)
    check_budget(tmp_path, budget, [SONNET], "compile")
    record_spend(tmp_path, budget, "compile exp_001", USAGE)
    check_budget(tmp_path, budget, [SONNET], "compile")  # $4.50 of $5.00: still allowed
    record_spend(tmp_path, budget, "compile exp_002", USAGE)
    with pytest.raises(GuardError, match=r"budget of \$5.00 is spent"):
        check_budget(tmp_path, budget, [SONNET], "compile")


def test_a_ceiling_without_a_price_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """Enforcing a dollar ceiling needs prices, and guessing them would be worse than refusing."""
    budget = _budget(max_usd=5.0)
    with pytest.raises(GuardError, match="no price is declared for anthropic/claude-opus-5"):
        check_budget(tmp_path, budget, [SONNET, "anthropic/claude-opus-5"], "compile")


def test_no_ceiling_means_no_refusal(tmp_path: Path) -> None:
    budget = Config.from_mapping({"models": {"task": SONNET}}).budget
    record_spend(tmp_path, budget, "compile", USAGE)
    check_budget(tmp_path, budget, [SONNET], "compile")


def test_a_price_matches_with_or_without_the_provider_prefix(tmp_path: Path) -> None:
    budget = Config.from_mapping(
        {
            "models": {"task": SONNET},
            "budget": {"prices": {"claude-sonnet-5": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}},
        }
    ).budget
    assert record_spend(tmp_path, budget, "compile", USAGE) == pytest.approx(4.5)


def test_the_cli_refuses_a_paid_command_over_budget(tmp_path: Path, fixtures_dir: Path) -> None:
    """The gate is code, not advice: `compile` stops before spending anything more."""
    import yaml

    from doc_ai_kit.cli import cli
    from doc_ai_kit.scaffold_writer import ScaffoldOptions, create_project

    project = tmp_path / "acme"
    create_project(project, ScaffoldOptions(project_name="Acme"))
    (project / "tasks.yaml").write_text((fixtures_dir / "toy_tasks.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
    config["models"]["task"] = config["models"]["reflection"] = SONNET
    config["budget"] = {
        "max_usd": 1.0,
        "prices": {SONNET: {"input_per_mtok": 3.0, "output_per_mtok": 15.0}},
    }
    (project / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (project / "data").mkdir(exist_ok=True)
    (project / "data" / "annotation_rules.md").write_text("rules", encoding="utf-8")
    (project / "data" / "splits.json").write_text(
        '{"seed": 1, "strategy": "x", "assignments": {"train": [], "val": [], "holdout": []}}', encoding="utf-8"
    )
    record_spend(project / "runs", _budget(max_usd=1.0), "compile exp_001", USAGE)

    result = CliRunner().invoke(cli, ["--project", str(project), "compile", "--variable", "x"])
    assert result.exit_code != 0
    assert "budget of $1.00 is spent" in f"{result.output}{result.exception}"
    assert "spent before it" not in result.output
