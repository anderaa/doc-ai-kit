"""``config.yaml`` parsed into typed objects.

Like tasks.yaml, the project's configuration is validated on load and converted into typed
objects rather than passed around as dicts. Two defaults are deliberately absent: the task
model and the reflection model. A harness that silently picks a model for you is a harness
that silently spends money.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed or incomplete."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    """Which language models the project uses, in DSPy's ``provider/name`` form."""

    task: str | None = None
    reflection: str | None = None
    # how hard the task model thinks before answering. Thinking is on by default for the Claude
    # 5 family, it is billed as output, and it counts against max_tokens: every truncation seen
    # in this harness's own runs was a reply that spent its whole budget thinking and never
    # answered. Unset leaves the provider's default in place, so nothing changes silently.
    task_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    # "disabled" turns thinking off outright. Prefer a lower task_effort: switching thinking off
    # has known failure modes on some models that a low effort avoids
    task_thinking: Literal["adaptive", "disabled"] | None = None
    # 1.0 rather than 0.0: the Claude 5 family accepts only temperature=1, and a default of
    # 0.0 makes every run fail at the first call rather than at configuration time
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def _effort_and_thinking_agree(self) -> ModelConfig:
        # an effort level is a thinking depth, so it cannot sit alongside thinking switched off;
        # the provider mapping would quietly switch thinking back on rather than refuse
        if self.task_thinking == "disabled" and self.task_effort is not None:
            raise ValueError(
                "models.task_effort and task_thinking: disabled cannot be combined. An effort level "
                "sets how deeply the model thinks; set one or the other"
            )
        return self

    def task_lm_kwargs(self) -> dict[str, Any]:
        """Return the provider settings for the task model's thinking, as the LM takes them."""
        if self.task_effort is not None:
            # LiteLLM maps this to adaptive thinking plus output_config.effort for Claude 5
            return {"reasoning_effort": self.task_effort}
        if self.task_thinking is not None:
            return {"thinking": {"type": self.task_thinking}}
        return {}

    def describe(self) -> dict[str, Any]:
        """Return the model settings every run records, so two runs can be told apart."""
        return {
            "task_model": self.task,
            "task_effort": self.task_effort,
            "task_thinking": self.task_thinking,
            "reflection_model": self.reflection,
            "max_tokens": self.max_tokens,
        }

    def require_task(self) -> str:
        """Return the task model, failing loudly rather than picking one."""
        if not self.task:
            raise ConfigError("config.yaml: models.task is not set; the harness will not choose a model for you")
        return self.task

    def require_reflection(self) -> str:
        """Return the reflection model, falling back to the task model only if told to."""
        if not self.reflection:
            raise ConfigError(
                "config.yaml: models.reflection is not set; GEPA and MIPROv2 need a proposer model. "
                "Set it explicitly, even if it is the same as models.task"
            )
        return self.reflection


class TruncationConfig(_Strict):
    """How long documents are cut down, declared once and applied everywhere.

    Declared once on purpose: text truncated differently in validation and production makes
    the validation numbers describe a system that was never shipped.
    """

    max_chars: int | None = Field(default=None, gt=0)
    strategy: Literal["head", "head_tail"] = "head"
    # for head_tail, the share of the budget given to the head
    head_share: float = Field(default=0.7, gt=0.0, lt=1.0)


class ExtractionConfig(_Strict):
    """How PDFs become text."""

    extractor: str = "pymupdf"
    fallback_extractor: str | None = "pdfplumber"
    ocr_fallback: bool = True
    # below this many characters per page, the text layer is treated as missing
    ocr_chars_per_page: int = Field(default=100, ge=0)
    ocr_extractor: str = "ocr"
    ocr_language: str = "eng"
    truncation: TruncationConfig = TruncationConfig()


class SplitConfig(_Strict):
    """How the corpus is divided, and what counts as enough examples."""

    seed: int = 20260918
    support_floor: int = Field(default=30, ge=1)
    measurable_floor: int = Field(default=10, ge=1)


class OptimizationConfig(_Strict):
    """Which optimizer runs, how the program is shaped, and what it may spend."""

    optimizer: Literal[
        "BootstrapFewShot",
        "BootstrapFewShotWithRandomSearch",
        "MIPROv2",
        "GEPA",
        "SIMBA",
    ] = "MIPROv2"
    # light by default: heavy settings have consumed thousands of rollouts in published runs
    auto: Literal["light", "medium", "heavy"] = "light"
    module: Literal["predict", "chain_of_thought"] = "predict"
    max_bootstrapped_demos: int = Field(default=4, ge=0)
    max_labeled_demos: int = Field(default=4, ge=0)
    # how many candidate programs to propose, and how many to actually evaluate. Left unset,
    # the optimizer picks from `auto`, which on a small corpus means far more trials than the
    # validation split can distinguish between. max_rollouts is the tripwire; these are the brakes.
    num_candidates: int | None = Field(default=None, ge=1)
    num_trials: int | None = Field(default=None, ge=1)
    # SIMBA only: how many optimization steps to take
    max_steps: int = Field(default=8, ge=1)
    num_threads: int = Field(default=8, ge=1)
    max_experiments: int = Field(default=8, ge=1)
    max_rollouts: int = Field(default=2000, ge=1)

    @model_validator(mode="after")
    def _explicit_budget_is_complete(self) -> OptimizationConfig:
        # MIPROv2's own constraint: once it stops taking its budget from `auto` it needs both
        # numbers. The other optimizers use num_candidates alone and ignore num_trials.
        if self.optimizer == "MIPROv2" and (self.num_candidates is None) != (self.num_trials is None):
            raise ValueError(
                "optimization.num_candidates and num_trials must be set together for MIPROv2: "
                "once it is no longer choosing them from `auto`, it needs both"
            )
        return self


class MetricConfig(_Strict):
    """How scoring treats an answer the document does not support.

    ``scored`` means an abstention is a first-class answer: null against null gold is a true
    negative. ``best_guess`` tells the program to answer anyway, which changes the metric
    rather than only the prompt, so switching it is a recorded decision.
    """

    abstention: Literal["scored", "best_guess"] = "scored"
    bootstrap_resamples: int = Field(default=2000, ge=100)
    # classes recorded as report-unmeasured: still scored and reported, but dropped from
    # the optimization target; added by hand after make-splits records the decision
    excluded_classes: dict[str, list[str]] = Field(default_factory=dict)


class EvaluationConfig(_Strict):
    """How a scoring run treats a reply it could not read.

    A failed reply is retried with a fresh generation. One that still fails is scored as a
    wrong answer on every task -- never as an abstention, which would earn credit wherever
    the gold is null -- and is listed by document. Above ``max_failure_rate`` the run
    refuses to report numbers at all, because they would describe the failures as much as
    the program. The default of zero means any reply that survives every retry stops it.
    """

    max_retries: int = Field(default=2, ge=0)
    max_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class ProductionConfig(_Strict):
    """How the full-corpus run behaves."""

    max_retries: int = Field(default=3, ge=0)
    batch_size: int = Field(default=32, ge=1)
    num_threads: int = Field(default=8, ge=1)
    # submit through the Message Batches API: half the price, finished within a day and
    # usually within the hour. Used for Anthropic models; anything else runs live
    use_batch_api: bool = True
    batch_poll_seconds: int = Field(default=60, ge=1)
    spot_check_sample: int = Field(default=50, ge=0)
    max_schema_failure_rate: float = Field(default=0.005, ge=0.0, le=1.0)


class BudgetConfig(_Strict):
    """The cost ceiling for the engagement; there is no default that spends money."""

    max_usd: float | None = Field(default=None, gt=0.0)


class Config(_Strict):
    """The whole of config.yaml."""

    models: ModelConfig = ModelConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    splits: SplitConfig = SplitConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    metric: MetricConfig = MetricConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    production: ProductionConfig = ProductionConfig()
    budget: BudgetConfig = BudgetConfig()
    source: Path | None = None

    @model_validator(mode="after")
    def _floors_are_ordered(self) -> Config:
        if self.splits.measurable_floor > self.splits.support_floor:
            raise ValueError(
                f"splits.measurable_floor ({self.splits.measurable_floor}) is above "
                f"splits.support_floor ({self.splits.support_floor}); a class cannot be "
                "unmeasurable yet still count toward the optimization target"
            )
        return self

    @classmethod
    def from_mapping(cls, data: Any, source: Path | None = None) -> Config:
        """Build a config from the raw contents of a config.yaml document."""
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ConfigError("config.yaml must be a mapping")
        try:
            return cls.model_validate({**dict(data), "source": source})
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors())
            raise ConfigError(f"config.yaml: {details}") from exc

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load and validate a config.yaml file."""
        if not path.exists():
            raise ConfigError(f"{path} does not exist; run this from a project directory")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: not valid YAML: {exc}") from exc
        config = cls.from_mapping(data, source=path)
        logger.info("loaded config from %s", path)
        return config

    def fingerprint(self) -> str:
        """Return a stable hash of the configuration, recorded in every run's metadata.

        The source path is excluded so the same settings hash identically wherever the
        project lives on disk.
        """
        payload = self.model_dump(mode="json", exclude={"source"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
