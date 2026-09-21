"""Packaging tests: the wheel must build, and it must carry the scaffold.

These exist because a duplicate `force-include` on `scaffold/` made the wheel fail to build
entirely, while every other test passed: the editable install used in development never
exercises the wheel path, so nothing noticed until an install was actually attempted.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCAFFOLD = REPO / "src" / "doc_harness" / "scaffold"


def _scaffold_files() -> set[str]:
    """Return every scaffold file, as the path it should occupy inside the wheel."""
    return {
        f"doc_harness/scaffold/{path.relative_to(SCAFFOLD).as_posix()}"
        for path in SCAFFOLD.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("git") is None:
        pytest.skip("hatchling's file selection needs git")
    outdir = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir), str(REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if "No module named build" in result.stderr:
            pytest.skip("the `build` package is not installed")
        pytest.fail(f"wheel build failed:\n{result.stdout}\n{result.stderr}")
    wheels = list(outdir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    return wheels[0]


def test_wheel_builds(wheel: Path) -> None:
    assert wheel.stat().st_size > 0


def test_wheel_carries_every_scaffold_file(wheel: Path) -> None:
    """A wheel missing the scaffold produces a newproject that writes an empty directory."""
    packaged = set(zipfile.ZipFile(wheel).namelist())
    missing = sorted(_scaffold_files() - packaged)
    assert not missing, f"scaffold files absent from the wheel: {missing}"


def test_wheel_has_no_duplicate_entries(wheel: Path) -> None:
    """Two entries at one path is what broke the build; a zip can hold them silently."""
    names = zipfile.ZipFile(wheel).namelist()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, duplicates


def test_wheel_exposes_the_console_scripts(wheel: Path) -> None:
    entry_points = next(
        (
            zipfile.ZipFile(wheel).read(name).decode("utf-8")
            for name in zipfile.ZipFile(wheel).namelist()
            if name.endswith("entry_points.txt")
        ),
        "",
    )
    assert "newproject = doc_harness.cli:newproject" in entry_points
    assert "doc-harness = doc_harness.cli:main" in entry_points
