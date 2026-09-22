"""Shared pytest fixtures."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from doc_harness.registry import Registry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the directory holding the checked-in fixture files."""
    return FIXTURES


def pytest_configure(config: pytest.Config) -> None:
    """Silence the SWIG deprecation warnings PyMuPDF's bindings emit on import."""
    config.addinivalue_line("filterwarnings", "ignore:builtin type Swig.*:DeprecationWarning")
    config.addinivalue_line("filterwarnings", "ignore:builtin type swigvarlink.*:DeprecationWarning")
    # emitted from inside LiteLLM's own type definitions; nothing here can act on it
    config.addinivalue_line("filterwarnings", "ignore:Item .* is using the `ReadOnly` qualifier:UserWarning")


@pytest.fixture
def toy_registry(fixtures_dir: Path) -> Registry:
    """Return the three-task toy registry."""
    from doc_harness.registry import Registry

    return Registry.from_yaml(fixtures_dir / "toy_tasks.yaml")


@pytest.fixture
def toy_rows(fixtures_dir: Path) -> list[dict[str, Any]]:
    """Return the toy split's raw gold and predicted rows."""
    import yaml

    return list(yaml.safe_load((fixtures_dir / "toy_examples.yaml").read_text(encoding="utf-8"))["examples"])


def document_for(doc_id: str) -> str:
    """Build a document body that carries its own id, so a scripted LM can key on it."""
    return f"[doc={doc_id}] This agreement is filed under contract number A-1 and is governed by state law."


@contextmanager
def scripted_lm(answers: dict[str, dict[str, Any]]) -> Iterator[None]:
    """Configure DSPy with a deterministic fake LM keyed on each document's id marker.

    Tests never call a real model: a scripted LM makes the plumbing assertions exact and
    keeps the suite free and offline.
    """
    import dspy
    from dspy.utils.dummies import DummyLM

    keyed = {f"[doc={doc_id}]": values for doc_id, values in answers.items()}
    with dspy.context(lm=DummyLM(keyed)):
        yield


def load_example(name: str) -> ModuleType:
    """Load an example project's generate.py under a name of its own.

    Both example projects have a generate.py. Importing them by the bare name "generate"
    makes whichever loads first shadow the other for the rest of the test session.
    """
    path = Path(__file__).resolve().parents[1] / "examples" / name / "generate.py"
    module_name = f"example_{name}_generate"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_cache(tmp_path: Path) -> Iterator[None]:
    """Give DSPy a private cache, so these tests neither read nor pollute ~/.dspy_cache."""
    import dspy

    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=str(tmp_path / "cache"))
    try:
        yield
    finally:
        dspy.configure_cache()


class StubProvider:
    """Stands in for the model provider: truncates the first ``bad`` calls, answers properly after.

    It sits behind DSPy's real cache, which is the point -- the bug lived in the cache.
    """

    def __init__(self, bad: int) -> None:
        self.bad = bad
        self.calls = 0

    def __call__(self, **kwargs: Any) -> Any:
        import litellm

        self.calls += 1
        if self.calls <= self.bad:
            content, finish = "[[ ## flag ## ]]\ntr", "length"
        else:
            content = (
                "[[ ## flag ## ]]\ntrue\n\n[[ ## state ## ]]\nCA\n\n" "[[ ## number ## ]]\nA-1\n\n[[ ## completed ## ]]"
            )
            finish = "stop"
        return litellm.ModelResponse(
            model="claude-sonnet-5",
            choices=[{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


def real_lm() -> Any:
    import dspy

    return dspy.LM("anthropic/claude-sonnet-5", temperature=1.0, max_tokens=64, num_retries=0)
