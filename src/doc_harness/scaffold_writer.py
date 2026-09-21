"""Creating a new project directory from the packaged scaffold.

``newproject`` has to run *outside* any project, because the thing it writes is the exact
harness pin the project will install. That is why it is a console script rather than a
subcommand of the project CLI.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from doc_harness import __version__

logger = logging.getLogger(__name__)

# directories a project needs that carry no files at creation time
EMPTY_DIRS = (
    "data/pdfs",
    "data/text",
    "programs/compiled",
    "runs",
)

# files copied with their template suffix removed
TEMPLATE_SUFFIX = ".template"


class ScaffoldError(RuntimeError):
    """Raised when a project directory cannot be created as asked."""


@dataclass(frozen=True)
class ScaffoldOptions:
    """The substitutions applied to the packaged scaffold."""

    project_name: str
    harness_version: str = __version__
    python_version: str = "3.12.11"
    python_requires: str = "3.12"

    @property
    def project_slug(self) -> str:
        """Return a package-safe form of the project name."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.project_name.lower()).strip("-")
        return slug or "doc-harness-project"

    def as_mapping(self) -> dict[str, str]:
        """Return the placeholder substitutions."""
        return {
            "PROJECT_NAME": self.project_name,
            "PROJECT_SLUG": self.project_slug,
            "HARNESS_VERSION": self.harness_version,
            "PYTHON_VERSION": self.python_version,
            "PYTHON_REQUIRES": self.python_requires,
        }


def _substitute(text: str, values: dict[str, str]) -> str:
    """Replace ``{{PLACEHOLDER}}`` markers, failing loudly on an unknown one."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ScaffoldError(f"scaffold references unknown placeholder {{{{{key}}}}}")
        return values[key]

    return re.sub(r"\{\{([A-Z_]+)\}\}", replace, text)


# suffixes treated as text and run through placeholder substitution
TEXT_SUFFIXES = frozenset({".md", ".yaml", ".yml", ".toml", ".py", ".txt", ".template", ".gitignore", ""})


def create_project(target: Path, options: ScaffoldOptions, force: bool = False) -> Path:
    """Create a new project directory from the packaged scaffold.

    :param target: Where to create the project
    :param options: The substitutions to apply
    :param force: Write into a directory that already has contents
    :returns: The project directory
    """
    if target.exists() and any(target.iterdir()) and not force:
        raise ScaffoldError(f"{target} is not empty; pass --force to write into it anyway")
    target.mkdir(parents=True, exist_ok=True)
    values = options.as_mapping()

    source = resources.files("doc_harness") / "scaffold"
    with resources.as_file(source) as scaffold_dir:
        root = Path(scaffold_dir)
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts:
                continue
            destination = target / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.name.endswith(TEMPLATE_SUFFIX):
                destination = destination.with_name(destination.name[: -len(TEMPLATE_SUFFIX)])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in TEXT_SUFFIXES or path.name.startswith("."):
                destination.write_text(_substitute(path.read_text(encoding="utf-8"), values), encoding="utf-8")
            else:
                shutil.copy2(path, destination)

    for empty_dir in EMPTY_DIRS:
        (target / empty_dir).mkdir(parents=True, exist_ok=True)
        keep = target / empty_dir / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    (target / ".python-version").write_text(options.project_slug + "\n", encoding="utf-8")
    logger.info("created project %s pinned to doc-harness %s", target, options.harness_version)
    return target
