"""The DSPy program: one predictor per task group, typed outputs from the registry.

The default shape is a single module covering every task. A task only needs its own module
when it needs its own tuned instruction, because DSPy tunes instructions per predictor --
that is what the optional ``group`` key in tasks.yaml buys.

``dspy.Predict`` is the default and ``dspy.ChainOfThought`` is a configuration option, to be
measured rather than assumed: reasoning fields cost tokens on every call and do not reliably
help on extraction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from doc_harness.adapters import strict_version_of
from doc_harness.config import OptimizationConfig
from doc_harness.registry import Registry

logger = logging.getLogger(__name__)

INPUT_FIELD = "document"


def make_module(signature: Any, module_type: str) -> Any:
    """Build one DSPy predictor for a signature.

    :param signature: The signature to predict
    :param module_type: Either ``predict`` or ``chain_of_thought``
    :returns: A DSPy module
    """
    import dspy

    if module_type == "predict":
        return dspy.Predict(signature)
    if module_type == "chain_of_thought":
        return dspy.ChainOfThought(signature)
    raise ValueError(f"unknown module type {module_type!r}; expected 'predict' or 'chain_of_thought'")


def build_program(
    registry: Registry,
    module_type: str = "predict",
    instructions: Mapping[str, str] | None = None,
    demos: Mapping[str, Sequence[Any]] | None = None,
) -> Any:
    """Build the program for a project.

    :param registry: The parsed tasks.yaml
    :param module_type: ``predict`` or ``chain_of_thought``
    :param instructions: Optional starting instruction per group
    :param demos: Optional hand-written demonstrations per group
    :returns: A ``dspy.Module`` whose forward returns one prediction covering every task
    """
    import dspy

    signatures = registry.signatures(instructions)
    predictors = {group: make_module(signature, module_type) for group, signature in signatures.items()}
    if demos:
        for group, group_demos in demos.items():
            if group not in predictors:
                raise ValueError(f"demos given for unknown group {group!r}")
            predictors[group].demos = list(group_demos)

    class DocumentProgram(dspy.Module):  # type: ignore[misc]
        """Answers every registered task about one document."""

        def __init__(self) -> None:
            super().__init__()
            self.task_ids = list(registry.ids)
            self.group_tasks = {group: [task.id for task in tasks] for group, tasks in registry.groups().items()}
            for group, predictor in predictors.items():
                setattr(self, _attribute_for(group), predictor)

        def forward(self, **kwargs: Any) -> Any:
            """Run every group's predictor and merge their outputs into one prediction.

            Parsing is strict wherever the program runs: a reply that omits a field raises
            rather than coming back null, so a failure is never scored as an abstention.
            """
            document = kwargs[INPUT_FIELD]
            merged: dict[str, Any] = {}
            with dspy.context(adapter=strict_version_of(dspy.settings.adapter)):
                for group, task_ids in self.group_tasks.items():
                    predictor = getattr(self, _attribute_for(group))
                    prediction = predictor(**{INPUT_FIELD: document})
                    for task_id in task_ids:
                        merged[task_id] = getattr(prediction, task_id, None)
            return dspy.Prediction(**merged)

    program = DocumentProgram()
    logger.info(
        "built %s program with %d predictor(s) over %d task(s)",
        module_type,
        len(predictors),
        len(registry),
    )
    return program


def _attribute_for(group: str) -> str:
    """Return the attribute name a group's predictor is stored under.

    DSPy discovers predictors by walking module attributes, so the name has to be a valid
    identifier and stable across save and load.
    """
    cleaned = "".join(character if character.isalnum() else "_" for character in group)
    return f"predict_{cleaned}"


def build_from_config(registry: Registry, config: OptimizationConfig, **kwargs: Any) -> Any:
    """Build the program described by the project's optimization settings."""
    return build_program(registry, module_type=config.module, **kwargs)


def save_program(program: Any, path: Path) -> None:
    """Save a compiled program so production never has to recompile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path))
    logger.info("saved compiled program to %s", path)


def load_program(registry: Registry, path: Path, module_type: str = "predict") -> Any:
    """Rebuild a program's shape from the registry and load its compiled state.

    :param registry: The parsed tasks.yaml the program was compiled against
    :param path: The saved program file
    :param module_type: The module type the program was compiled with
    :returns: The loaded program, ready to run
    """
    if not path.exists():
        raise FileNotFoundError(f"no compiled program at {path}")
    program = build_program(registry, module_type=module_type)
    program.load(str(path))
    logger.info("loaded compiled program from %s", path)
    return program


def instructions_of(program: Any) -> dict[str, str]:
    """Return the current instruction text of every predictor, for the run record."""
    found: dict[str, str] = {}
    for name, predictor in program.named_predictors():
        signature = getattr(predictor, "signature", None)
        if signature is not None:
            found[name] = signature.instructions
    return found


@contextmanager
def fresh_generation(attempt: int) -> Iterator[None]:
    """Make a retry ask the model again instead of replaying the cached answer.

    DSPy caches every response, a truncated or unparseable one included, keyed on the
    request. A plain retry sends an identical request, gets the same cached failure back,
    and fails the same way however many times it is tried -- retries only ever helped with
    errors that never reached the cache. A distinct ``rollout_id`` per attempt gives each
    retry its own cache key, so it draws a new sample; DSPy strips the id before the request
    reaches the provider. The first attempt keeps the shared cache, which is free and correct
    when the document has been answered before.

    At temperature 0 DSPy leaves the cache in place even with a rollout id, but a fresh call
    would return the same answer anyway; there the fix for truncation is a larger
    ``models.max_tokens``.
    """
    import dspy

    lm = dspy.settings.lm
    if attempt == 1 or lm is None:
        yield
        return
    with dspy.context(lm=lm.copy(rollout_id=attempt)):
        yield
