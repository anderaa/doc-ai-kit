"""Project-specific normalizers, matchers and extractors.

Everything in this package is imported by the CLI at startup, so registering a hook here is
enough to make it available by name in tasks.yaml and config.yaml. The installed doc_harness
package is never edited.

Example::

    from collections.abc import Mapping
    from typing import Any

    from doc_harness.hooks import register_normalizer
    from doc_harness.normalize import collapse_whitespace, fold_unicode

    @register_normalizer("policy_number")
    def policy_number(value: Any, params: Mapping[str, Any]) -> str | None:
        \"\"\"Canonicalize this client's policy numbers, which vary in punctuation.\"\"\"
        if value is None:
            return None
        return collapse_whitespace(fold_unicode(str(value))).replace("-", "").upper()

Registering a name that already exists overrides the built-in, and the override is logged.
"""
