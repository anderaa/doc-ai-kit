"""Registration points for project-specific code.

A project keeps its custom normalizers, matchers and extractors in its own ``custom/``
package and registers them here. The package resolves everything by name, so a project
never edits the installed package to add behaviour.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from doc_ai_kit.values import MatchResult

logger = logging.getLogger(__name__)

NormalizerFn = Callable[[Any, Mapping[str, Any]], Any]
MatcherFn = Callable[[Any, Any, Mapping[str, Any]], MatchResult]
ExtractorFn = Callable[..., Any]

_NORMALIZERS: dict[str, NormalizerFn] = {}
_MATCHERS: dict[str, MatcherFn] = {}
_EXTRACTORS: dict[str, ExtractorFn] = {}

F = TypeVar("F", bound=Callable[..., Any])


class HookError(LookupError):
    """Raised when a name in tasks.yaml or config.yaml resolves to no registered hook."""


def _register(store: dict[str, Any], name: str, fn: Any, kind: str) -> Any:
    if name in store and store[name] is not fn:
        # a project overriding a built-in is legitimate, but it must be visible in the log
        logger.info("%s %r overridden by %s.%s", kind, name, fn.__module__, fn.__qualname__)
    store[name] = fn
    return fn


def register_normalizer(name: str) -> Callable[[F], F]:
    """Register a normalizer under a name usable in tasks.yaml."""

    def decorate(fn: F) -> F:
        return _register(_NORMALIZERS, name, fn, "normalizer")  # type: ignore[no-any-return]

    return decorate


def register_matcher(name: str) -> Callable[[F], F]:
    """Register a matcher under a name usable in tasks.yaml."""

    def decorate(fn: F) -> F:
        return _register(_MATCHERS, name, fn, "matcher")  # type: ignore[no-any-return]

    return decorate


def register_extractor(name: str) -> Callable[[F], F]:
    """Register a PDF text extractor under a name usable in config.yaml."""

    def decorate(fn: F) -> F:
        return _register(_EXTRACTORS, name, fn, "extractor")  # type: ignore[no-any-return]

    return decorate


def _lookup(store: dict[str, Any], name: str, kind: str) -> Any:
    try:
        return store[name]
    except KeyError:
        known = ", ".join(sorted(store)) or "(none registered)"
        raise HookError(f"unknown {kind} {name!r}; registered {kind}s: {known}") from None


def get_normalizer(name: str) -> NormalizerFn:
    """Look up a registered normalizer, failing loudly with the available names."""
    _ensure_builtins()
    return _lookup(_NORMALIZERS, name, "normalizer")  # type: ignore[no-any-return]


def get_matcher(name: str) -> MatcherFn:
    """Look up a registered matcher, failing loudly with the available names."""
    _ensure_builtins()
    return _lookup(_MATCHERS, name, "matcher")  # type: ignore[no-any-return]


def get_extractor(name: str) -> ExtractorFn:
    """Look up a registered extractor, failing loudly with the available names."""
    _ensure_builtins()
    return _lookup(_EXTRACTORS, name, "extractor")  # type: ignore[no-any-return]


def registered_names() -> dict[str, list[str]]:
    """Return every registered hook name by kind, for diagnostics and `status`."""
    _ensure_builtins()
    return {
        "normalizers": sorted(_NORMALIZERS),
        "matchers": sorted(_MATCHERS),
        "extractors": sorted(_EXTRACTORS),
    }


# modules whose import side effect is registering the built-in hooks
_BUILTIN_MODULES = ("doc_ai_kit.normalize", "doc_ai_kit.match", "doc_ai_kit.extract")

_builtins_loaded = False
# re-entrant: importing a built-in module must not deadlock if it looks a hook up in turn
_builtins_lock = threading.RLock()


def _ensure_builtins() -> None:
    """Import the built-in normalizer, matcher and extractor modules exactly once.

    Deferred rather than imported at module scope so that :mod:`doc_ai_kit.hooks` stays
    free of cycles: the built-ins import this module to register themselves.

    Under a lock, and the flag is set only once the imports are done. Set beforehand, a
    second thread arriving mid-import saw "already loaded" and looked up an empty registry:
    every worker in a threaded scoring run then failed with "unknown matcher: (none registered)".
    """
    global _builtins_loaded
    with _builtins_lock:
        if _builtins_loaded:
            return
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
        _builtins_loaded = True


def load_project_customizations(project_dir: Path) -> list[str]:
    """Import a project's ``custom/`` package so its registrations take effect.

    :param project_dir: The project root containing ``custom/``
    :returns: The module names imported, in import order
    """
    _ensure_builtins()
    custom_dir = project_dir / "custom"
    if not custom_dir.is_dir():
        return []
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    imported: list[str] = []
    init = custom_dir / "__init__.py"
    if init.exists():
        importlib.import_module("custom")
        imported.append("custom")
    for path in sorted(custom_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        name = f"custom.{path.stem}" if init.exists() else path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise HookError(f"could not load project customization {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        imported.append(name)
    logger.info("loaded project customizations: %s", ", ".join(imported) or "(none)")
    return imported
