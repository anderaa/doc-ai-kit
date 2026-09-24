"""Strict parsing: a reply that omits a field has failed, not answered null.

DSPy fills a missing output field with ``None`` whenever the field's type allows ``None``.
Every package task is nullable, so a reply that stops after five of nine fields -- truncated,
or simply malformed -- would come back with the last four set to null, indistinguishable
from a model that deliberately abstained. Scored, those nulls read as the model declining to
answer; in production they would land in the output file as if they were answers.

These adapters change nothing the model sees: the prompt and the schema are DSPy's own. They
only refuse a reply that leaves a field out, while still accepting a field that is present
and explicitly null. A refusal raises, so retries and failure accounting take over from there.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import json_repair
import regex
from dspy.adapters.chat_adapter import ChatAdapter, field_header_pattern
from dspy.adapters.json_adapter import JSONAdapter
from dspy.utils.exceptions import AdapterParseError

# the pattern DSPy's JSONAdapter uses to pull an object out of surrounding text
_JSON_OBJECT = r"\{(?:[^{}]|(?R))*\}"


def _refuse_missing(adapter: str, signature: Any, completion: str, present: Iterable[str]) -> None:
    """Raise if the reply left out any output field, however its type treats None."""
    found = set(present)
    missing = [name for name in signature.output_fields if name not in found]
    if missing:
        raise AdapterParseError(
            adapter_name=adapter,
            signature=signature,
            lm_response=completion,
            message=(
                f"the reply omitted {', '.join(missing)}. An omitted field is a failed reply, not an "
                "answer of null; it is usually truncation, so check models.max_tokens"
            ),
        )


def chat_fields_present(completion: str) -> set[str]:
    """Return the field names a chat-format reply actually wrote a header for."""
    present = set()
    for line in completion.splitlines():
        match = field_header_pattern.match(line.strip())
        if match:
            present.add(match.group(1))
    return present


def json_fields_present(completion: str) -> set[str]:
    """Return the keys a JSON reply actually contained, extracted the way DSPy extracts them."""
    parsed = json_repair.loads(completion)
    if not isinstance(parsed, dict):
        match = regex.search(_JSON_OBJECT, completion, regex.DOTALL)
        parsed = json_repair.loads(match.group(0)) if match else {}
    return set(parsed) if isinstance(parsed, dict) else set()


class StrictJSONAdapter(JSONAdapter):  # type: ignore[misc]
    """DSPy's JSON adapter, refusing a reply that omits a field."""

    def parse(self, signature: Any, completion: str) -> dict[str, Any]:
        """Parse as DSPy does, then refuse the result if a field was never in the reply."""
        fields: dict[str, Any] = super().parse(signature, completion)
        _refuse_missing("StrictJSONAdapter", signature, completion, json_fields_present(completion))
        return fields


class StrictChatAdapter(ChatAdapter):  # type: ignore[misc]
    """DSPy's chat adapter, refusing a reply that omits a field.

    Its fallback to JSON is strict too: the stock fallback would quietly reintroduce the very
    null-filling this class exists to prevent.
    """

    def parse(self, signature: Any, completion: str) -> dict[str, Any]:
        """Parse as DSPy does, then refuse the result if a field was never in the reply."""
        fields: dict[str, Any] = super().parse(signature, completion)
        _refuse_missing("StrictChatAdapter", signature, completion, chat_fields_present(completion))
        return fields

    def _make_json_adapter_fallback(self) -> StrictJSONAdapter:
        return StrictJSONAdapter(
            use_native_function_calling=self.use_native_function_calling,
            parallel_tool_calls=self.parallel_tool_calls,
        )


def strict_version_of(adapter: Any) -> Any:
    """Return the strict counterpart of whichever adapter is configured.

    An adapter that is already strict is returned as is. JSON stays JSON, and anything else --
    including no adapter at all, which DSPy treats as chat -- becomes strict chat.
    """
    if isinstance(adapter, StrictChatAdapter | StrictJSONAdapter):
        return adapter
    if isinstance(adapter, JSONAdapter):
        return StrictJSONAdapter()
    return StrictChatAdapter()
