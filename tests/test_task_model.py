"""The task model's thinking settings: what reaches the provider, and what is refused."""

from __future__ import annotations

import pytest

from doc_harness.config import Config, ConfigError
from doc_harness.program import task_lm


def _config(**models: object) -> Config:
    return Config.from_mapping({"models": {"task": "anthropic/claude-sonnet-5", **models}})


def test_unset_leaves_the_provider_default_alone() -> None:
    """Nothing changes silently: no setting, nothing sent."""
    assert _config().models.task_lm_kwargs() == {}


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_an_effort_level_is_sent_as_reasoning_effort(effort: str) -> None:
    assert _config(task_effort=effort).models.task_lm_kwargs() == {"reasoning_effort": effort}


@pytest.mark.parametrize("mode", ["adaptive", "disabled"])
def test_a_thinking_mode_is_sent_as_is(mode: str) -> None:
    assert _config(task_thinking=mode).models.task_lm_kwargs() == {"thinking": {"type": mode}}


def test_effort_with_thinking_disabled_is_refused() -> None:
    """The provider mapping would quietly switch thinking back on rather than refuse."""
    with pytest.raises(ConfigError, match="cannot be combined"):
        _config(task_effort="low", task_thinking="disabled")


def test_effort_with_adaptive_thinking_is_allowed() -> None:
    assert _config(task_effort="low", task_thinking="adaptive").models.task_lm_kwargs() == {"reasoning_effort": "low"}


def test_an_unknown_effort_is_refused() -> None:
    with pytest.raises(ConfigError):
        _config(task_effort="extreme")


def test_the_task_model_is_built_as_configured() -> None:
    """One constructor everywhere, so a program is not compiled under different settings."""
    lm = task_lm(_config(task_effort="low", max_tokens=2048, temperature=1.0).models)
    assert lm.model == "anthropic/claude-sonnet-5"
    assert lm.kwargs["reasoning_effort"] == "low"
    assert lm.kwargs["max_tokens"] == 2048


def test_every_run_records_the_thinking_settings() -> None:
    described = _config(task_effort="medium").models.describe()
    assert described["task_effort"] == "medium"
    assert described["task_thinking"] is None
    assert described["task_model"] == "anthropic/claude-sonnet-5"


def test_thinking_settings_change_the_config_fingerprint() -> None:
    """Runs under different effort must never look like the same configuration."""
    assert _config().fingerprint() != _config(task_effort="low").fingerprint()
