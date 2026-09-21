"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
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
