"""What the project has spent, and the ceiling it may not pass.

``budget.max_usd`` was recorded with every run and enforced nowhere, which is the same as
having no budget: the first real project found it after the fact, from token counts.

Enforcement needs prices, and prices change and differ per account, so the package has none
built in. A project that sets a ceiling declares its prices beside it; a project that sets no
ceiling still gets its tokens recorded, which is what the ledger and the report quote.

The ceiling is checked before each paid step, not during one: a run already under way cannot
be unspent, and killing it halfway would waste what it has already paid for. So a step that
would start over the ceiling is refused, and a step that ends over it says so loudly.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from doc_ai_kit.config import BudgetConfig, ModelPrice
from doc_ai_kit.guards import GuardError

logger = logging.getLogger(__name__)

SPEND_FILE = "spend.json"
PER_MILLION = 1_000_000


@dataclass(frozen=True)
class SpendEntry:
    """One paid step, as it was billed."""

    at: str
    label: str
    model: str
    route: str
    input_tokens: int
    output_tokens: int
    # None when the project declared no price for this model, so tokens are known and cost is not
    usd: float | None


def price_for(budget: BudgetConfig, model: str) -> ModelPrice | None:
    """Return the declared price for a model, by exact name or without its provider prefix."""
    prices = budget.prices
    if model in prices:
        return prices[model]
    bare = model.split("/", 1)[-1]
    return prices.get(bare)


def usd_for(budget: BudgetConfig, model: str, input_tokens: int, output_tokens: int, route: str) -> float | None:
    """Return what a call cost, or None when the project declared no price for that model."""
    price = price_for(budget, model)
    if price is None:
        return None
    cost = (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / PER_MILLION
    return float(cost * (price.batch_multiplier if route == "batch" else 1.0))


def load(runs_dir: Path) -> list[SpendEntry]:
    """Read the spend ledger."""
    path = runs_dir / SPEND_FILE
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
    return [SpendEntry(**row) for row in rows]


def spent_usd(runs_dir: Path) -> float:
    """Return what has been spent so far, counting only steps whose price was known."""
    return sum(entry.usd or 0.0 for entry in load(runs_dir))


def totals(runs_dir: Path) -> dict[str, Any]:
    """Return totals for the report and the ledger: tokens always, dollars where priced."""
    entries = load(runs_dir)
    unpriced = sorted({entry.model for entry in entries if entry.usd is None})
    return {
        "input_tokens": sum(entry.input_tokens for entry in entries),
        "output_tokens": sum(entry.output_tokens for entry in entries),
        "usd": round(sum(entry.usd or 0.0 for entry in entries), 4),
        "models_without_a_declared_price": unpriced,
        "steps": len(entries),
    }


def record_spend(
    runs_dir: Path,
    budget: BudgetConfig,
    label: str,
    usage: Mapping[str, Mapping[str, Any]],
    route: str = "live",
) -> float:
    """Append what one step spent, and warn if that took the project past its ceiling.

    :param runs_dir: The project's runs directory
    :param budget: The project's budget settings
    :param label: What was run, e.g. ``compile exp_003``
    :param usage: Per model, a mapping carrying ``prompt_tokens``/``input_tokens`` and
        ``completion_tokens``/``output_tokens``, as DSPy's usage tracker returns it
    :param route: ``live`` or ``batch``; batch is billed at the declared multiplier
    :returns: What this step cost, as far as prices are known
    """
    entries = load(runs_dir)
    added = 0.0
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for model, counts in sorted(usage.items()):
        input_tokens = int(counts.get("prompt_tokens") or counts.get("input_tokens") or 0)
        output_tokens = int(counts.get("completion_tokens") or counts.get("output_tokens") or 0)
        if not input_tokens and not output_tokens:
            continue
        usd = usd_for(budget, model, input_tokens, output_tokens, route)
        entries.append(
            SpendEntry(
                at=stamp,
                label=label,
                model=model,
                route=route,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usd=None if usd is None else round(usd, 6),
            )
        )
        added += usd or 0.0
    if not entries:
        return 0.0
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {"entries": [asdict(entry) for entry in entries], "totals": None}
    (runs_dir / SPEND_FILE).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    total = sum(entry.usd or 0.0 for entry in entries)
    logger.info("%s spent $%.2f; $%.2f so far", label, added, total)
    if budget.max_usd is not None and total > budget.max_usd:
        logger.warning(
            "the budget of $%.2f is spent: $%.2f so far. The next paid step will be refused",
            budget.max_usd,
            total,
        )
    return added


def check_budget(runs_dir: Path, budget: BudgetConfig, models: Sequence[str], about_to: str) -> None:
    """Refuse to start a paid step that the budget cannot cover.

    :param runs_dir: The project's runs directory
    :param budget: The project's budget settings
    :param models: The models this step will call
    :param about_to: What is about to run, named in the refusal
    :raises GuardError: If the ceiling is spent, or cannot be applied to these models
    """
    if budget.max_usd is None:
        return
    unpriced = sorted({model for model in models if model and price_for(budget, model) is None})
    if unpriced:
        raise GuardError(
            f"budget.max_usd is set to ${budget.max_usd:.2f}, but no price is declared for "
            f"{', '.join(unpriced)}, so {about_to} cannot be held to it. Add the model under "
            "budget.prices in config.yaml -- input_per_mtok and output_per_mtok, from your own "
            "pricing -- or clear budget.max_usd to run without a ceiling."
        )
    spent = spent_usd(runs_dir)
    if spent >= budget.max_usd:
        raise GuardError(
            f"the budget of ${budget.max_usd:.2f} is spent (${spent:.2f} so far), so {about_to} is refused. "
            f"Raising budget.max_usd is a deliberate, recorded decision; {SPEND_FILE} lists what went where."
        )
