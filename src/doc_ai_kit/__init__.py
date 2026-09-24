"""doc-ai-kit: a fixed package for document classification and extraction projects."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("doc-ai-kit")
except PackageNotFoundError:  # pragma: no cover - only hit in a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
