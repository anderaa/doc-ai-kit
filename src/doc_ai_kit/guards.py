"""The gates, implemented as code rather than as advice.

Instructions in a markdown file are advisory, and a session under time pressure routes
around them. Each guard here enforces one rule that, once broken, silently invalidates
every number the project goes on to report:

* ``data/`` is read-only to every optimization path;
* the holdout is evaluated once, and a second evaluation is recorded as an override;
* the experiment budget is fixed up front rather than extended when results look close.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HOLDOUT_LOCK = ".lock"


class GuardError(RuntimeError):
    """Raised when a gate refuses. The message always says why the gate exists."""


def _digest(path: Path) -> str:
    """Return a content hash for one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(paths: Sequence[Path]) -> dict[str, str]:
    """Hash a set of files so a later check can prove they did not change.

    :param paths: The files to hash; missing files are recorded as absent
    :returns: Path string to digest
    """
    return {str(path): (_digest(path) if path.exists() else "(absent)") for path in paths}


@contextmanager
def readonly(paths: Sequence[Path], reason: str = "this path is read-only to optimization") -> Iterator[None]:
    """Refuse to let a block of code modify the given files.

    Labels and splits are ground truth. An optimizer that can edit them can improve its own
    score without improving the program, and nothing downstream would show it.

    :param paths: The files that must be identical afterwards
    :param reason: Included in the error so the refusal explains itself
    """
    before = snapshot(paths)
    try:
        yield
    except BaseException:
        # the body already failed; report the modification but let the original error through,
        # because replacing it would hide why the run actually died
        changed = _changed_since(before, paths)
        if changed:
            logger.error("%s; also modified during the failed run: %s", reason, ", ".join(changed))
        raise
    changed = _changed_since(before, paths)
    if changed:
        raise GuardError(f"{reason}; modified during the run: {', '.join(changed)}")


def _changed_since(before: Mapping[str, str], paths: Sequence[Path]) -> list[str]:
    """Return the paths whose contents differ from the recorded snapshot."""
    after = snapshot(paths)
    return sorted(path for path, digest in before.items() if after.get(path) != digest)


@dataclass(frozen=True)
class HoldoutLock:
    """The record that the holdout has been opened."""

    path: Path
    opened_at: str
    opened_by: str
    overrides: list[dict[str, str]]

    @property
    def was_overridden(self) -> bool:
        """Return whether the holdout has been evaluated more than once."""
        return bool(self.overrides)

    def to_dict(self) -> dict[str, Any]:
        """Return the lock file payload."""
        return {"opened_at": self.opened_at, "opened_by": self.opened_by, "overrides": self.overrides}


def read_holdout_lock(holdout_dir: Path) -> HoldoutLock | None:
    """Return the holdout lock if it exists."""
    path = holdout_dir / HOLDOUT_LOCK
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return HoldoutLock(
        path=path,
        opened_at=str(data.get("opened_at", "")),
        opened_by=str(data.get("opened_by", "")),
        overrides=list(data.get("overrides", [])),
    )


def open_holdout(holdout_dir: Path, opened_by: str, override: bool = False, reason: str = "") -> HoldoutLock:
    """Take the holdout lock, refusing a second evaluation unless explicitly overridden.

    The holdout is a one-shot measurement. Evaluating twice and keeping the better number
    turns it into a second validation split, and the reported accuracy stops being an
    estimate of anything.

    :param holdout_dir: The runs/holdout directory
    :param opened_by: What is opening it, recorded in the lock and the report
    :param override: Whether this is a deliberate second evaluation
    :param reason: Why the override was taken; required when overriding
    :returns: The lock, whose ``overrides`` list goes into the report
    """
    holdout_dir.mkdir(parents=True, exist_ok=True)
    path = holdout_dir / HOLDOUT_LOCK
    now = datetime.now(UTC).isoformat(timespec="seconds")
    existing = read_holdout_lock(holdout_dir)
    if existing is None:
        lock = HoldoutLock(path=path, opened_at=now, opened_by=opened_by, overrides=[])
        path.write_text(json.dumps(lock.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info("holdout opened by %s; this is the one-shot measurement", opened_by)
        return lock
    if not override:
        raise GuardError(
            f"the holdout was already evaluated on {existing.opened_at} by {existing.opened_by}. "
            "It is a one-shot measurement: evaluating again and keeping the better number turns it "
            "into a second validation split. Re-run with an explicit override if you have decided "
            "to spend it, and the override will be recorded in the report."
        )
    if not reason.strip():
        raise GuardError("an override needs a reason; it is written into the report alongside the numbers")
    overrides = [*existing.overrides, {"at": now, "by": opened_by, "reason": reason.strip()}]
    lock = HoldoutLock(path=path, opened_at=existing.opened_at, opened_by=existing.opened_by, overrides=overrides)
    path.write_text(json.dumps(lock.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.warning("holdout re-opened by %s (override #%d): %s", opened_by, len(overrides), reason)
    return lock


def require_files(paths: Sequence[Path], because: str) -> None:
    """Refuse to proceed unless every named file exists.

    :param paths: The files that must be present
    :param because: What the files are needed for, included in the error
    """
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise GuardError(f"missing {', '.join(missing)}: {because}")


def check_experiment_budget(runs_dir: Path, max_experiments: int) -> int:
    """Refuse to start an experiment beyond the budget fixed in config.yaml.

    The budget is set up front rather than extended once results look close, because
    "one more run" is how a validation split gets fitted.

    :param runs_dir: The project's runs directory
    :param max_experiments: The ceiling from config.yaml
    :returns: How many experiments have already run
    """
    existing = sorted(path for path in runs_dir.glob("exp_*") if path.is_dir()) if runs_dir.exists() else []
    if len(existing) >= max_experiments:
        raise GuardError(
            f"the experiment budget of {max_experiments} is spent ({len(existing)} run). "
            "Raise optimization.max_experiments in config.yaml only as a deliberate, recorded decision: "
            "the budget exists so the search is time-boxed rather than run until the validation split gives in."
        )
    return len(existing)


# BaseException, not Exception: the metric runs inside DSPy's own workers, which catch
# Exception, log it and carry on. A budget that only logged let a run continue past its cap
class RolloutBudgetExceeded(BaseException):
    """Raised inside the metric to stop an optimizer run that has spent its rollout budget."""

    def __init__(self, spent: int, max_rollouts: int) -> None:
        self.spent = spent
        self.max_rollouts = max_rollouts
        super().__init__(
            f"the rollout budget of {max_rollouts} is spent ({spent} used). "
            "Heavy optimizer settings have consumed thousands of rollouts in published runs; "
            "raise optimization.max_rollouts deliberately, not reflexively."
        )


def check_rollout_budget(spent: int, max_rollouts: int) -> None:
    """Refuse to continue once the rollout budget is spent."""
    if spent >= max_rollouts:
        raise RolloutBudgetExceeded(spent, max_rollouts)
