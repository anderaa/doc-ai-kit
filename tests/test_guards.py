"""Gate tests: each of these enforces something a markdown instruction cannot."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from doc_ai_kit.guards import (
    GuardError,
    check_experiment_budget,
    check_rollout_budget,
    open_holdout,
    read_holdout_lock,
    readonly,
    require_files,
    snapshot,
)


def test_readonly_allows_an_untouched_run(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("a", encoding="utf-8")
    with readonly([path]):
        _ = path.read_text(encoding="utf-8")


def test_readonly_catches_a_write(tmp_path: Path) -> None:
    """An optimizer that can edit labels can improve its score without improving anything."""
    path = tmp_path / "labels.jsonl"
    path.write_text("a", encoding="utf-8")
    with pytest.raises(GuardError, match="read-only"), readonly([path]):
        path.write_text("b", encoding="utf-8")


def test_readonly_catches_a_deletion(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text("a", encoding="utf-8")
    with pytest.raises(GuardError, match="modified during the run"), readonly([path]):
        path.unlink()


def test_readonly_catches_a_creation(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    with pytest.raises(GuardError), readonly([path]):
        path.write_text("sneaky", encoding="utf-8")


def test_snapshot_records_absence(tmp_path: Path) -> None:
    assert snapshot([tmp_path / "nope"]) == {str(tmp_path / "nope"): "(absent)"}


def test_holdout_opens_once(tmp_path: Path) -> None:
    lock = open_holdout(tmp_path, opened_by="holdout command")
    assert lock.was_overridden is False
    assert (tmp_path / ".lock").exists()

    with pytest.raises(GuardError, match="one-shot measurement"):
        open_holdout(tmp_path, opened_by="holdout command")


def test_holdout_override_is_recorded(tmp_path: Path) -> None:
    open_holdout(tmp_path, opened_by="first run")
    lock = open_holdout(tmp_path, opened_by="second run", override=True, reason="labels were corrected")
    assert lock.was_overridden is True
    assert lock.overrides[0]["reason"] == "labels were corrected"
    stored = json.loads((tmp_path / ".lock").read_text(encoding="utf-8"))
    assert stored["overrides"][0]["by"] == "second run"
    # the original opening is preserved, so the report can show when it was really spent
    assert stored["opened_by"] == "first run"


def test_override_needs_a_reason(tmp_path: Path) -> None:
    open_holdout(tmp_path, opened_by="first run")
    with pytest.raises(GuardError, match="needs a reason"):
        open_holdout(tmp_path, opened_by="second run", override=True, reason="   ")


def test_read_lock_returns_none_when_unopened(tmp_path: Path) -> None:
    assert read_holdout_lock(tmp_path) is None


def test_require_files_explains_itself(tmp_path: Path) -> None:
    with pytest.raises(GuardError, match="compile needs the labeling rules"):
        require_files([tmp_path / "annotation_rules.md"], because="compile needs the labeling rules")


def test_experiment_budget_refuses_past_the_ceiling(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    for index in range(3):
        (runs / f"exp_{index:03d}").mkdir()
    assert check_experiment_budget(runs, max_experiments=5) == 3
    with pytest.raises(GuardError, match="budget of 3 is spent"):
        check_experiment_budget(runs, max_experiments=3)


def test_experiment_budget_on_a_fresh_project(tmp_path: Path) -> None:
    assert check_experiment_budget(tmp_path / "runs", max_experiments=1) == 0


def test_rollout_budget() -> None:
    from doc_ai_kit.guards import RolloutBudgetExceeded

    check_rollout_budget(9, 10)
    with pytest.raises(RolloutBudgetExceeded, match="rollout budget of 10 is spent"):
        check_rollout_budget(10, 10)


def test_a_spent_rollout_budget_is_not_an_ordinary_exception() -> None:
    """Reported from a real run: a run carried on 11 rollouts past its cap.

    The metric runs inside DSPy's workers, which catch Exception, log it and continue. Only
    a BaseException escapes them and stops the run.
    """
    from doc_ai_kit.guards import RolloutBudgetExceeded

    assert not issubclass(RolloutBudgetExceeded, Exception)
    caught = None
    try:
        try:
            check_rollout_budget(10, 10)
        except Exception as exc:  # noqa: BLE001 - exactly what DSPy's worker does
            caught = exc
    except RolloutBudgetExceeded as exceeded:
        assert exceeded.spent == 10 and exceeded.max_rollouts == 10
    assert caught is None, "DSPy's own error handling would have swallowed this"


def test_readonly_does_not_mask_the_original_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A failed run must report why it failed, not why the guard also noticed a write."""
    path = tmp_path / "labels.jsonl"
    path.write_text("a", encoding="utf-8")
    with caplog.at_level("ERROR"), pytest.raises(ZeroDivisionError), readonly([path]):
        path.write_text("b", encoding="utf-8")
        raise ZeroDivisionError("the real failure")
    assert "also modified during the failed run" in caplog.text


def test_builtin_hooks_are_registered_before_any_thread_sees_them() -> None:
    """Found while rendering a report: every worker failed with "(none registered)".

    The flag was set before the imports ran, so a second thread arriving mid-import skipped
    the import and looked up an empty registry.
    """
    import importlib
    import threading

    from doc_ai_kit import hooks

    real_import = importlib.import_module
    started = threading.Event()

    def slow_import(name: str, package: str | None = None) -> Any:
        if name == "doc_ai_kit.normalize":
            started.set()
            time.sleep(0.05)
            # an import that has already run registers nothing the second time, so the
            # registry is refilled here to stand in for the real import's side effect
            hooks._MATCHERS.update(saved_matchers)
        return real_import(name, package)

    saved_matchers = dict(hooks._MATCHERS)
    errors: list[Exception] = []

    def look_up() -> None:
        try:
            hooks.get_matcher("identity")
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    try:
        hooks._builtins_loaded = False
        hooks._MATCHERS.clear()
        with mock.patch.object(importlib, "import_module", slow_import):
            first = threading.Thread(target=look_up)
            first.start()
            started.wait(timeout=1)
            second = threading.Thread(target=look_up)
            second.start()
            first.join()
            second.join()
        assert not errors, errors
    finally:
        hooks._MATCHERS.update(saved_matchers)
        hooks._builtins_loaded = True
