"""Provider-neutral Chat Completions tool loop shared by Task B and Task C.

This module knows nothing about any task's business rules: it handles the extra
assistant/tool turns needed by OpenAI-compatible function calling, runs the
bounded tools, and returns one compact audit record per case.  A caller injects
an adapter for the task contract and may inject ``request_fn`` for deterministic
offline tests.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .protocol import TaskAdapter, resolved_task_variant
from .tools import (
    MAX_ARTIFACT_BYTES,
    MAX_ARTIFACT_COUNT,
    MAX_ARTIFACT_TOTAL_BYTES,
    MAX_CODE_CHARS,
    MAX_READ_CHARS,
    MAX_TOOL_OUTPUT_CHARS,
    MAX_TIMEOUT_SECONDS,
    TOOL_SCHEMAS,
    ScratchWorkspace,
    ToolDispatcher,
    runtime_metadata,
)


RequestFunction = Callable[[dict[str, Any]], dict[str, Any]]
AUTO_PAYLOAD_SUMMARY_THRESHOLD = 120_000
HIDDEN_PAYLOAD_KEYS = {
    "truth",
    "ground_truth",
    "future",
    "target",
    "target_return",
    "target_returns",
    "target_labels",
    "future_return",
    "future_returns",
    "next_return",
    "next_returns",
    "next_month_return",
    "next_month_returns",
    "label",
    "labels",
    "y",
    "masked_to_original",
    "original_to_masked",
    "anchor_date",
    "internal_case_id",
}


def normalize_base_url(value: str) -> str:
    """Normalize a URL copied from rendered Markdown or shell formatting.

    A URL copied from a rendered Markdown link can arrive literally as
    ``[http://host/v1](http://host/v1)``.  Passing that value to urllib leads
    to the unhelpful ``unknown url type: [http`` error before any provider
    request is made.  Accept the common Markdown/code wrapper forms at the
    runtime boundary and leave ordinary URLs unchanged.
    """

    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    # A copied command may retain a pair of shell/code quotes around the URL.
    for _ in range(2):
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`"}:
            text = text[1:-1].strip()

    # Markdown link: [label](destination).  Prefer the destination because
    # the label may have been shortened or escaped by the renderer.
    if text.startswith("[") and "](" in text and text.endswith(")"):
        marker = text.find("](")
        destination = text[marker + 2 : -1].strip()
        if destination:
            text = destination

    # Markdown autolink: <https://host/v1>.
    if len(text) >= 2 and text.startswith("<") and text.endswith(">"):
        text = text[1:-1].strip()
    return text.rstrip("/")


@dataclass(frozen=True)
class AgenticConfig:
    """Settings shared by all agentic cases in one run."""

    model: str
    base_url: str
    temperature: float = 0.0
    seed: int | None = 1
    max_output_tokens: int = 8_000
    timeout_seconds: float = 300.0
    retries: int = 0
    max_tool_calls: int = 8
    max_turns: int = 12
    tool_choice: str = "auto"
    json_mode: bool = True
    thinking: str = "provider_default"
    reasoning_effort: str | None = None
    repair_invalid: bool = False
    include_payload: bool = False
    keep_artifacts: bool = True
    record_raw_turns: bool = False
    payload_prompt_mode: str = "auto"
    payload_summary_threshold: int = AUTO_PAYLOAD_SUMMARY_THRESHOLD

    def __post_init__(self) -> None:
        # Canonicalize once so endpoint construction, request metadata, and
        # settings hashes all describe the same URL.
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))

    def validate(self) -> None:
        if not str(self.model).strip():
            raise ValueError("model must not be empty")
        if not str(self.base_url).strip():
            raise ValueError("base_url must not be empty")
        if not math.isfinite(float(self.temperature)) or self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed must be an integer or null")
        if self.seed is not None and not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32-1")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if not math.isfinite(float(self.timeout_seconds)) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if self.max_turns < 2:
            raise ValueError("max_turns must be at least 2")
        if self.tool_choice not in {"auto", "required", "none"}:
            raise ValueError("tool_choice must be auto, required, or none")
        if self.thinking not in {"provider_default", "enabled", "disabled"}:
            raise ValueError("thinking must be provider_default, enabled, or disabled")
        if self.thinking == "disabled" and self.reasoning_effort:
            raise ValueError("reasoning_effort cannot be used when thinking is disabled")
        if self.reasoning_effort is not None and self.reasoning_effort not in {"low", "medium", "high", "max"}:
            raise ValueError("reasoning_effort must be low, medium, high, or max")
        if self.payload_prompt_mode not in {"auto", "full", "summary"}:
            raise ValueError("payload_prompt_mode must be auto, full, or summary")
        if self.payload_summary_threshold < 1:
            raise ValueError("payload_summary_threshold must be positive")


class ProviderHTTPError(RuntimeError):
    """Bounded HTTP error retaining the status for compatibility fallback."""

    def __init__(self, status: int, message: str) -> None:
        self.status = int(status)
        super().__init__(message)


def chat_endpoint(base_url: str) -> str:
    base = normalize_base_url(base_url)
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def safe_endpoint_for_record(value: str) -> str:
    """Remove URL credentials/query tokens before persisting endpoint metadata."""

    text = str(value)
    try:
        parts = urlsplit(text)
    except ValueError:
        return "[redacted invalid endpoint]"
    if not parts.scheme or not parts.netloc:
        return text
    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    try:
        port = parts.port
    except ValueError:
        return "[redacted invalid endpoint]"
    if port is not None:
        netloc += f":{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bounded(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated at {limit} characters]"


def _redact_secret_text(value: Any, secret: str) -> str:
    """Prevent an accidentally echoed API key from entering the audit row."""

    text = str(value)
    if secret and len(secret) >= 8:
        text = text.replace(secret, "[redacted api key]")
    return text


# A provider may return private chain-of-thought under different field names.
# Keep the raw response useful for protocol auditing while replacing those
# fields with a marker.  The auditable ``provider_reasoning_summary`` field is
# extracted separately and is bounded; task-level reasoning belongs in the
# final response's public ``reasoning_trace`` field.
_PRIVATE_REASONING_KEYS = {
    "reasoning",
    "reasoning_content",
    "thinking",
    "thoughts",
    "chain_of_thought",
    "cot",
    "private_reasoning",
}


def _redact_private_reasoning(value: Any) -> tuple[Any, bool]:
    """Return a JSON-compatible response with private reasoning redacted."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        redacted = False
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _PRIVATE_REASONING_KEYS:
                output[key_text] = "[redacted private reasoning]"
                redacted = True
                continue
            cleaned, child_redacted = _redact_private_reasoning(child)
            output[key_text] = cleaned
            redacted = redacted or child_redacted
        return output, redacted
    if isinstance(value, list):
        output_list: list[Any] = []
        redacted = False
        for child in value:
            cleaned, child_redacted = _redact_private_reasoning(child)
            output_list.append(cleaned)
            redacted = redacted or child_redacted
        return output_list, redacted
    if isinstance(value, tuple):
        cleaned, redacted = _redact_private_reasoning(list(value))
        return cleaned, redacted
    return value, False


def _payload_shape(value: Any, *, depth: int = 0) -> Any:
    """Describe a public payload value without copying large numeric arrays."""

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:240] + "..."
    if isinstance(value, Mapping):
        keys = [str(key) for key in value.keys()]
        result: dict[str, Any] = {
            "type": "object",
            "key_count": len(keys),
            "keys": keys[:80],
        }
        if depth < 1:
            fields: dict[str, Any] = {}
            for key in keys[:40]:
                fields[key] = _payload_shape(value.get(key), depth=depth + 1)
            result["fields"] = fields
        return result
    if isinstance(value, (list, tuple)):
        result = {"type": "array", "length": len(value)}
        if not value:
            return result
        first = value[0]
        if isinstance(first, Mapping):
            item_keys: list[str] = []
            factor_ids: list[str] = []
            for item in value[:300]:
                for key in item.keys():
                    key_text = str(key)
                    if key_text not in item_keys:
                        item_keys.append(key_text)
                if "factor_id" in item:
                    factor_ids.append(str(item.get("factor_id")))
            result["item_keys"] = item_keys[:80]
            if factor_ids:
                result["factor_ids"] = factor_ids
            result["first_item"] = _payload_shape(first, depth=2)
        elif isinstance(first, (list, tuple)):
            result["row_lengths"] = [len(row) if isinstance(row, (list, tuple)) else None for row in value[:8]]
            result["first_row"] = _payload_shape(first, depth=2)
        elif all(isinstance(item, str) for item in value[:300]):
            result["items"] = [str(item) for item in value[:300]]
        else:
            result["item_type"] = type(first).__name__
        return result
    return {"type": type(value).__name__}


def _payload_summary(payload: Mapping[str, Any], *, original_chars: int) -> dict[str, Any]:
    """Build the compact user-message view used for oversized public inputs."""

    fields: dict[str, Any] = {}
    track = str(payload.get("track", "numeric_only")).strip().lower()
    for key, value in payload.items():
        key_text = str(key)
        # Semantic cards are the deliberate treatment difference between the
        # two tracks.  Preserve their complete public name/definition/formula
        # sidecar even when the historical rows are summarized; otherwise a
        # summary-mode request silently collapses semantic into numeric-only.
        # The cards are small (the formal 38/76/153-feature cases are bounded),
        # and the complete payload remains available to ``python_exec`` too.
        if key_text == "factor_cards" and track in {"semantic", "semantic_numeric"}:
            fields[key_text] = value
        else:
            # Preserve small scalar routing metadata verbatim.  All other
            # nested or array-valued data is represented by shape/field
            # information; complete values remain available to python_exec.
            fields[key_text] = _payload_shape(value)
    return {
        "_agentic_payload_mode": "summary",
        "_agentic_payload_note": (
            "The complete public payload is available as `payload` inside "
            "python_exec. Semantic `factor_cards` are also preserved in this "
            "summary; use the tool to inspect exact rows or card fields. No "
            "hidden truth or future data is mounted there."
        ),
        "_serialized_full_payload_chars": int(original_chars),
        "fields": fields,
    }


def _payload_prompt_content(payload: Mapping[str, Any], config: AgenticConfig) -> tuple[str, str, int]:
    """Return ``(user_message, effective_mode, full_serialized_chars)``."""

    serialized = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False)
    full_chars = len(serialized)
    mode = config.payload_prompt_mode
    if mode == "auto":
        mode = "summary" if full_chars > config.payload_summary_threshold else "full"
    if mode == "summary":
        content = json.dumps(
            _payload_summary(payload, original_chars=full_chars),
            ensure_ascii=False,
        )
    else:
        content = serialized
    return content, mode, full_chars


def _no_tools_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add an explicit finalization notice when a provider rejects tools.

    The canonical system prompts describe the tools as available.  If a
    gateway rejects the tool schema, merely removing ``tools`` from the retry
    request leaves the model with contradictory instructions and often causes
    it to emit pseudo tool calls or wait for a tool result.  Keep the original
    public context, then make the compatibility boundary explicit in one
    short user turn.
    """

    output = [dict(message) for message in messages]
    output.append(
        {
            "role": "user",
            "content": (
                "COMPATIBILITY NOTICE: tools are unavailable for this request. "
                "Do not call python_exec, read_artifact, or write_artifact. "
                "Return the required task JSON object directly using only the "
                "public payload already shown."
            ),
        }
    )
    return output


def _post_json(endpoint: str, api_key: str, body: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        endpoint,
        data=json.dumps(dict(body), ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - endpoint is user-configured
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            detail = ""
        if len(detail) > 2_000:
            detail = detail[:2_000] + "..."
        message = f"HTTP {exc.code}: {exc.reason}"
        if detail:
            message += "; " + detail
        raise ProviderHTTPError(exc.code, message) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"provider transport/JSON error: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("provider response must be a JSON object")
    if isinstance(parsed.get("error"), Mapping):
        error = parsed["error"]
        raise RuntimeError(str(error.get("message") or json.dumps(error, ensure_ascii=False)))
    return parsed


def _call_with_retry(
    endpoint: str,
    api_key: str,
    body: dict[str, Any],
    config: AgenticConfig,
    request_fn: RequestFunction | None,
) -> tuple[dict[str, Any], int]:
    last_error: Exception | None = None
    retryable = {408, 409, 425, 429, 500, 502, 503, 504}
    for attempt in range(config.retries + 1):
        try:
            response = request_fn(body) if request_fn is not None else _post_json(
                endpoint,
                api_key,
                body,
                config.timeout_seconds,
            )
            if not isinstance(response, dict):
                raise RuntimeError("request function returned a non-object response")
            return response, attempt
        except ProviderHTTPError as exc:
            last_error = exc
            if exc.status not in retryable or attempt >= config.retries:
                break
        except (RuntimeError, OSError, ValueError) as exc:
            last_error = exc
            if attempt >= config.retries:
                break
        time.sleep(min(2**attempt, 8))
    if isinstance(last_error, ProviderHTTPError):
        raise last_error
    raise RuntimeError(str(last_error or "provider request failed"))


def _first_choice(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return {}
    return choices[0]


def _message(response: Mapping[str, Any]) -> Mapping[str, Any]:
    message = _first_choice(response).get("message")
    return message if isinstance(message, Mapping) else {}


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    if value is None and isinstance(message.get("parsed"), Mapping):
        return json.dumps(dict(message["parsed"]), ensure_ascii=False, default=str)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=False, default=str)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    if value is None:
        return ""
    return str(value)


def _reasoning_summary(response: Mapping[str, Any], message: Mapping[str, Any]) -> str | None:
    # Only explicit summary fields are retained.  ``reasoning_content`` and a
    # generic top-level ``reasoning`` field are commonly provider-private
    # chain-of-thought, so they are redacted from the audit record instead.
    for source in (
        message.get("reasoning_summary"),
        response.get("reasoning_summary"),
        response.get("reasoning_summary_text"),
    ):
        if source is not None and str(source).strip():
            return _bounded(source, 4_000)
    return None


def _usage(response: Mapping[str, Any]) -> dict[str, Any] | None:
    value = response.get("usage")
    if not isinstance(value, Mapping):
        return None
    # Provider usage objects contain counts and small nested count breakdowns;
    # preserve them without retaining any prompt/completion text.
    cleaned, _ = _redact_private_reasoning(dict(value))
    return cleaned if isinstance(cleaned, dict) else None


def _aggregate_usage(turns: Sequence[Mapping[str, Any]]) -> dict[str, int] | None:
    """Sum token counts across all conversation turns (including repair turns).

    Handles both the DeepSeek-style flat fields (``prompt_cache_hit_tokens``)
    and the OpenAI-style nested ``prompt_tokens_details.cached_tokens`` so the
    result is useful regardless of which gateway is in use.
    """
    totals: dict[str, int] = {}
    for turn in turns:
        u = turn.get("usage")
        if not isinstance(u, Mapping):
            continue
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cache_hit_tokens",
            "prompt_cache_miss_tokens",
        ):
            val = u.get(key)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                totals[key] = totals.get(key, 0) + int(val)
        # OpenAI nested form: usage.prompt_tokens_details.cached_tokens
        details = u.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            cached = details.get("cached_tokens")
            if isinstance(cached, (int, float)) and math.isfinite(float(cached)):
                totals["cached_tokens"] = totals.get("cached_tokens", 0) + int(cached)
    return totals if totals else None


def _tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = message.get("tool_calls")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    # A few older OpenAI-compatible gateways still emit the deprecated
    # ``function_call`` object.  Normalize it to the same internal shape.
    legacy = message.get("function_call")
    if isinstance(legacy, Mapping):
        return [{"id": "legacy_function_call", "type": "function", "function": dict(legacy)}]
    return []


def _parse_tool_arguments(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    if not isinstance(value, str):
        return None, "tool arguments must be a JSON object or JSON string"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, f"invalid tool argument JSON: {exc}"
    if not isinstance(parsed, Mapping):
        return None, "tool arguments JSON must be an object"
    return dict(parsed), None


def _tool_name(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(call.get("name") or "")


def _tool_arguments(call: Mapping[str, Any]) -> Any:
    function = call.get("function")
    if isinstance(function, Mapping):
        return function.get("arguments")
    return call.get("arguments")


def _tool_id(call: Mapping[str, Any], turn: int, index: int) -> str:
    value = str(call.get("id") or "").strip()
    return value or f"agentic_call_{turn}_{index}"


def _assistant_history_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Keep provider tool-call fields while dropping unrelated response keys."""

    output: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content"),
    }
    calls = _tool_calls(message)
    if calls:
        normalized_calls: list[dict[str, Any]] = []
        for call in calls:
            copied = dict(call)
            function = copied.get("function")
            if isinstance(function, Mapping):
                function_copy = dict(function)
                arguments = function_copy.get("arguments")
                if isinstance(arguments, Mapping):
                    function_copy["arguments"] = json.dumps(
                        dict(arguments), ensure_ascii=False, separators=(",", ":")
                    )
                copied["function"] = function_copy
            normalized_calls.append(copied)
        output["tool_calls"] = normalized_calls
    # Reasoning providers sometimes require this field on the next turn.  Do
    # not synthesize it; only preserve a value the provider actually returned.
    if message.get("reasoning_content") is not None:
        output["reasoning_content"] = message.get("reasoning_content")
    if message.get("reasoning_summary") is not None:
        output["reasoning_summary"] = message.get("reasoning_summary")
    return output


def _tool_result_message(call_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": _bounded(result, max(MAX_TOOL_OUTPUT_CHARS, MAX_READ_CHARS)),
    }


def _audit_tool_arguments(arguments: Any) -> Any:
    """Keep tool-call audit rows bounded even for malformed provider calls."""

    if isinstance(arguments, Mapping):
        output: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                output[str(key)] = _bounded(value, MAX_CODE_CHARS)
            elif isinstance(value, Mapping):
                output[str(key)] = _audit_tool_arguments(value)
            elif isinstance(value, list):
                output[str(key)] = [_audit_tool_arguments(item) for item in value[:100]]
            else:
                output[str(key)] = value
        return output
    return _bounded(arguments, MAX_CODE_CHARS)


def _strip_json_fence(content: str) -> str:
    """Return the first JSON-looking fenced block, or the stripped text.

    Compatible reasoning models occasionally prepend a short sentence before
    the fenced object even when the prompt asks for JSON only.  The previous
    implementation handled only a fence at character zero, which turned a
    perfectly usable response into an ``invalid JSON`` failure.  Keep this
    helper deliberately conservative: it removes a fence but does not try to
    repair arbitrary malformed JSON (validation remains the adapter's job).
    """

    text = str(content or "").strip()
    if "```" not in text:
        return text
    start = text.find("```")
    opening_end = text.find("\n", start + 3)
    if opening_end < 0:
        return text
    closing = text.find("```", opening_end + 1)
    if closing < 0:
        return text
    block = text[opening_end + 1 : closing].strip()
    # The opening fence may be ```json, ```JSON, or an unlabeled fence.
    # A non-JSON code block is left untouched so the normal parser reports a
    # useful error rather than silently selecting unrelated text.
    header = text[start + 3 : opening_end].strip().lower()
    if header in {"", "json", "jsonc"} or block.startswith("{"):
        return block
    return text


def parse_json_object(content: str) -> dict[str, Any]:
    original = str(content or "").strip()
    text = _strip_json_fence(original)
    if not text:
        raise ValueError("final assistant content is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as first_error:
        # As a final compatibility fallback, decode the first complete JSON
        # object embedded in a short prose prefix.  ``raw_decode`` avoids the
        # brittle first-{ / last-} slicing approach and stops at the object's
        # actual end.  We only accept mappings, never a nested scalar/list.
        decoder = json.JSONDecoder()
        parsed = None
        for index, char in enumerate(original):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(original[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            raise first_error
    if not isinstance(parsed, dict):
        raise ValueError("final assistant content must be a JSON object")
    return parsed


def _base_request_body(
    config: AgenticConfig,
    messages: Sequence[Mapping[str, Any]],
    *,
    tool_choice: str,
    include_tools: bool,
    json_mode: bool | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [dict(message) for message in messages],
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
    }
    if config.seed is not None:
        body["seed"] = config.seed
    json_mode_enabled = config.json_mode if json_mode is None else bool(json_mode)
    if json_mode_enabled:
        body["response_format"] = {"type": "json_object"}
    if config.thinking != "provider_default":
        body["thinking"] = {"type": config.thinking}
    if config.reasoning_effort:
        body["reasoning_effort"] = config.reasoning_effort
    if include_tools and tool_choice != "none":
        body["tools"] = TOOL_SCHEMAS
        body["tool_choice"] = tool_choice
    elif include_tools:
        # Explicitly tell compatible providers that no further tool call is
        # allowed while forcing the final JSON response.
        body["tools"] = TOOL_SCHEMAS
        body["tool_choice"] = "none"
    return body


def _api_parameters_for_record(body: Mapping[str, Any], *, include_payload: bool) -> dict[str, Any]:
    """Return non-secret request metadata without duplicating a huge payload."""

    output = {key: value for key, value in body.items() if key != "messages"}
    output["message_count"] = len(body.get("messages", [])) if isinstance(body.get("messages"), list) else None
    output["tools_count"] = len(body.get("tools", [])) if isinstance(body.get("tools"), list) else 0
    if include_payload:
        output["messages"] = body.get("messages")
    return output


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_agentic_case(
    payload: Mapping[str, Any],
    *,
    adapter: TaskAdapter,
    config: AgenticConfig,
    api_key: str = "",
    scratch_root: Path,
    artifact_destination: Path | None = None,
    public_artifact_files: Mapping[str, str] | None = None,
    truth: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    request_fn: RequestFunction | None = None,
) -> dict[str, Any]:
    """Run one payload through the tool loop and return its audit record.

    ``truth`` is consumed only after the model/tool loop has finished.  It is
    never written into the scratch directory and never appears in a request.
    """

    config.validate()
    payload_dict = dict(payload)
    leaked_keys = sorted(
        str(key) for key in payload_dict
        if str(key).strip().lower() in HIDDEN_PAYLOAD_KEYS
    )
    case_id = str(payload_dict.get("case_id") or uuid.uuid4().hex)
    task_variant = resolved_task_variant(adapter.task_name, payload_dict)
    endpoint = chat_endpoint(config.base_url)
    prompt = adapter.prompt(payload_dict)
    schema_path = adapter.schema_path(payload_dict)
    prompt_hash = _sha256({"system": prompt, "payload": payload_dict})
    payload_hash = _sha256(payload_dict)
    record: dict[str, Any] = {
        "record_type": "llm",
        "run_id": uuid.uuid4().hex,
        "case_id": case_id,
        "task": adapter.task_name,
        "task_variant": task_variant,
        "track": payload_dict.get("track"),
        "prompt_language": payload_dict.get("prompt_language"),
        "reasoning_protocol": payload_dict.get("reasoning_protocol", "prediction_only"),
        "payload_variant": payload_dict.get("payload_variant", "summary"),
        "model_id": config.model,
        "prompt_semantic_version": adapter.prompt_version(payload_dict),
        "prompt_hash": prompt_hash,
        "payload_hash": payload_hash,
        "settings_hash": _sha256(
            {
                "model": config.model,
                "base_url": config.base_url,
                "temperature": config.temperature,
                "seed": config.seed,
                "max_output_tokens": config.max_output_tokens,
                "timeout_seconds": config.timeout_seconds,
                "retries": config.retries,
                "max_tool_calls": config.max_tool_calls,
                "max_turns": config.max_turns,
                "tool_choice": config.tool_choice,
                "json_mode": config.json_mode,
                "thinking": config.thinking,
                "reasoning_effort": config.reasoning_effort,
                "repair_invalid": config.repair_invalid,
                "include_payload": config.include_payload,
                "keep_artifacts": config.keep_artifacts,
                "record_raw_turns": config.record_raw_turns,
                "payload_prompt_mode": config.payload_prompt_mode,
                "payload_summary_threshold": config.payload_summary_threshold,
            }
        ),
        "schema_hash": _file_sha256(schema_path) if schema_path.exists() else None,
        "model_settings": {
            "model": config.model,
            "base_url": safe_endpoint_for_record(config.base_url),
            "temperature": config.temperature,
            "seed": config.seed,
            "max_output_tokens": config.max_output_tokens,
            "timeout_seconds": config.timeout_seconds,
            "configured_retries": config.retries,
            "max_tool_calls": config.max_tool_calls,
            "max_turns": config.max_turns,
            "tool_choice": config.tool_choice,
            "json_mode": config.json_mode,
            "thinking": config.thinking,
            "reasoning_effort": config.reasoning_effort,
            "payload_prompt_mode": config.payload_prompt_mode,
            "payload_summary_threshold": config.payload_summary_threshold,
        },
        "runtime": runtime_metadata(),
        "tool_limits": {
            "max_code_chars": MAX_CODE_CHARS,
            "max_timeout_seconds": MAX_TIMEOUT_SECONDS,
            "max_tool_output_chars": MAX_TOOL_OUTPUT_CHARS,
            "max_read_chars": MAX_READ_CHARS,
            "max_artifact_bytes": MAX_ARTIFACT_BYTES,
            "max_artifact_count": MAX_ARTIFACT_COUNT,
            "max_artifact_total_bytes": MAX_ARTIFACT_TOTAL_BYTES,
        },
        "request": {
            "endpoint": safe_endpoint_for_record(endpoint),
            "payload_hash": payload_hash,
            "payload_included": bool(config.include_payload),
            "payload_prompt_mode": config.payload_prompt_mode,
            "payload_summary_threshold": config.payload_summary_threshold,
        },
        "started_at": _now(),
        "finished_at": None,
        "status": "failed",
        "dry_run": bool(dry_run),
        "truth_available_to_evaluator": truth is not None,
        "conversation_turns": [],
        "tool_trace": [],
        "public_artifacts": [],
        "raw_output": None,
        "raw_response": None,
        "raw_turns": [] if config.record_raw_turns else None,
        "provider_reasoning_summary": None,
        # Public, task-level reasoning is copied from the validated response
        # (normally ``reasoning_trace`` for process_trace).  Provider-private
        # chain-of-thought is kept out of this field.
        "reasoning": None,
        "parsed_response": None,
        "validation": None,
        "metrics": None,
        "tool_call_count": 0,
        "tool_success_count": 0,
        "tool_failure_count": 0,
        "repair_count": 0,
        "retry_count": 0,
        "usage": None,
        "latency_ms": None,
        "error": None,
        "warnings": [],
    }
    if config.include_payload:
        record["request"]["payload"] = payload_dict
    # Include the exact request metadata only after constructing the body; the
    # API key is never part of it.
    started = time.perf_counter()
    scratch: ScratchWorkspace | None = None
    try:
        if leaked_keys:
            raise ValueError(
                "public payload contains hidden/evaluation fields: "
                + ", ".join(leaked_keys)
            )
        if dry_run:
            # No model response exists in a dry run, so an empty response is
            # not a schema failure.  The row records routing/configuration only.
            record["validation"] = {"valid": None, "errors": [], "dry_run": True}
            record["status"] = "dry_run"
            return record
        if not api_key and request_fn is None:
            raise ValueError("API key is required unless an offline request_fn is supplied")

        scratch = ScratchWorkspace(
            scratch_root,
            payload_dict,
            keep_artifacts=config.keep_artifacts,
            seed=config.seed,
            public_artifact_files=public_artifact_files,
        )
        if getattr(scratch, "public_artifact_manifest", None):
            record["public_artifacts"] = list(scratch.public_artifact_manifest)
        dispatcher = ToolDispatcher(scratch)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
        ]
        payload_content, effective_payload_mode, full_payload_chars = _payload_prompt_content(payload_dict, config)
        messages.append({"role": "user", "content": payload_content})
        record["request"]["payload_prompt_mode"] = effective_payload_mode
        record["request"]["payload_prompt_chars"] = len(payload_content)
        record["request"]["full_payload_chars"] = full_payload_chars
        record["request"]["payload_prompt_sha256"] = _sha256(payload_content)
        final_content = ""
        final_response: dict[str, Any] | None = None
        final_message: Mapping[str, Any] = {}
        total_retries = 0
        tool_calls_used = 0
        forced_final = False
        json_mode_enabled = config.json_mode
        tools_enabled = config.tool_choice != "none" and config.max_tool_calls > 0
        record["request"]["effective_json_mode"] = json_mode_enabled
        record["request"]["effective_tools"] = tools_enabled

        # Leave one turn for a final no-tools response after the last allowed
        # tool call.  The reservation also applies when max_turns is set close
        # to the tool-call budget, so a partial analysis still gets a final
        # response opportunity.
        for turn in range(config.max_turns):
            reserve_final_turn = turn >= config.max_turns - 1
            force_no_tools = (
                config.tool_choice == "none"
                or not tools_enabled
                or tool_calls_used >= config.max_tool_calls
                or reserve_final_turn
            )
            if force_no_tools and tool_calls_used >= config.max_tool_calls and config.max_tool_calls > 0:
                forced_final = True
            requested_choice = "none" if force_no_tools else config.tool_choice
            body = _base_request_body(
                config,
                messages,
                tool_choice=requested_choice,
                # Once the per-case budget is exhausted (or the final turn
                # is reserved), omit the tool declarations entirely.  A few
                # OpenAI-compatible gateways ignore ``tool_choice=none`` if
                # ``tools`` is still present and keep emitting empty tool
                # calls, leaving no usable final JSON.  Tools remain enabled
                # on ordinary analysis turns.
                include_tools=tools_enabled and not force_no_tools,
                json_mode=json_mode_enabled,
            )
            if turn == 0:
                record["request"]["api_parameters"] = _api_parameters_for_record(
                    body,
                    include_payload=config.include_payload,
                )
            effective_body = body
            try:
                raw, retries = _call_with_retry(endpoint, api_key, body, config, request_fn)
            except ProviderHTTPError as exc:
                # Some compatible gateways implement tools but reject the
                # optional JSON-mode hint.  Retry this same turn once without
                # the hint, and retain the compatibility decision in the log.
                if json_mode_enabled and exc.status in {400, 404, 422}:
                    fallback_body = dict(body)
                    fallback_body.pop("response_format", None)
                    json_mode_enabled = False
                    record["request"]["effective_json_mode"] = False
                    record["warnings"].append("provider rejected response_format; retried without JSON mode")
                    effective_body = fallback_body
                    try:
                        raw, retries = _call_with_retry(endpoint, api_key, fallback_body, config, request_fn)
                    except ProviderHTTPError as tool_exc:
                        if not tools_enabled or tool_exc.status not in {400, 404, 422}:
                            raise
                        tools_enabled = False
                        record["request"]["effective_tools"] = False
                        record["warnings"].append("provider rejected tools; used prompt-only compatibility fallback")
                        no_tools_body = dict(fallback_body)
                        no_tools_body.pop("tools", None)
                        no_tools_body.pop("tool_choice", None)
                        no_tools_body["messages"] = _no_tools_messages(messages)
                        effective_body = no_tools_body
                        raw, retries = _call_with_retry(endpoint, api_key, no_tools_body, config, request_fn)
                elif tools_enabled and exc.status in {400, 404, 422}:
                    tools_enabled = False
                    record["request"]["effective_tools"] = False
                    record["warnings"].append("provider rejected tools; used prompt-only compatibility fallback")
                    no_tools_body = dict(body)
                    no_tools_body.pop("tools", None)
                    no_tools_body.pop("tool_choice", None)
                    no_tools_body["messages"] = _no_tools_messages(messages)
                    effective_body = no_tools_body
                    raw, retries = _call_with_retry(endpoint, api_key, no_tools_body, config, request_fn)
                else:
                    raise
            total_retries += retries
            final_response = raw
            message = _message(raw)
            final_message = message
            content = _content(message)
            summary = _reasoning_summary(raw, message)
            if summary:
                record["provider_reasoning_summary"] = summary
            calls = _tool_calls(message)
            turn_record: dict[str, Any] = {
                "turn": turn,
                "request_sha256": _sha256(effective_body),
                "finish_reason": _first_choice(raw).get("finish_reason"),
                "content_chars": len(content),
                "reasoning_chars": len(summary or ""),
                "usage": _usage(raw),
                "tool_calls": [],
            }
            for index, call in enumerate(calls):
                name = _tool_name(call)
                call_id = _tool_id(call, turn, index)
                arguments_value = _tool_arguments(call)
                arguments, argument_error = _parse_tool_arguments(arguments_value)
                turn_record["tool_calls"].append(
                    {
                        "id": call_id,
                        "name": name,
                        "arguments_chars": len(str(arguments_value or "")),
                        "arguments_valid": argument_error is None,
                    }
                )
            record["conversation_turns"].append(turn_record)
            if config.record_raw_turns:
                cleaned_raw, redacted = _redact_private_reasoning(raw)
                record["raw_turns"].append(cleaned_raw)
                if redacted:
                    record["warnings"].append("provider private reasoning fields were redacted from raw_turns")

            if calls and not force_no_tools:
                messages.append(_assistant_history_message(message))
                for index, call in enumerate(calls):
                    call_id = _tool_id(call, turn, index)
                    name = _tool_name(call)
                    arguments_value = _tool_arguments(call)
                    arguments, argument_error = _parse_tool_arguments(arguments_value)
                    if tool_calls_used >= config.max_tool_calls:
                        result = {
                            "ok": False,
                            "error_type": "tool_budget_exhausted",
                            "error": "per-case tool-call budget exhausted; return the final JSON",
                        }
                        record["warnings"].append("provider returned more tool calls than the per-case budget")
                    elif argument_error is not None:
                        result: dict[str, Any] = {
                            "ok": False,
                            "error_type": "invalid_arguments",
                            "error": argument_error,
                        }
                    else:
                        result = dispatcher.dispatch(name, arguments or {})
                    if tool_calls_used < config.max_tool_calls:
                        tool_calls_used += 1
                    record["tool_call_count"] = int(record["tool_call_count"]) + 1
                    if result.get("ok"):
                        record["tool_success_count"] += 1
                    else:
                        record["tool_failure_count"] += 1
                    record["tool_trace"].append(
                        {
                            "turn": turn,
                            "call_id": call_id,
                            "name": name,
                            "arguments": _audit_tool_arguments(
                                arguments if arguments is not None else arguments_value
                            ),
                            "result": result,
                        }
                    )
                    messages.append(_tool_result_message(call_id, result))
                # If a caller reserved a repair turn, go straight to the
                # clean finalization path once the tool budget is consumed.
                # This avoids one extra provider request that some gateways
                # answer with another empty/pseudo tool call.  The normal
                # multi-turn path remains unchanged when repair is disabled.
                if (
                    config.repair_invalid
                    and config.max_tool_calls > 0
                    and tool_calls_used >= config.max_tool_calls
                ):
                    forced_final = True
                    final_content = ""
                    final_message = message
                    record["warnings"].append(
                        "tool-call budget reached; used direct clean finalization"
                    )
                    break
                continue

            # A provider that returns a tool call even after tool_choice=none
            # cannot produce a reliable final answer.  Preserve any text but
            # mark the case as failed below if it is not valid JSON.
            if calls and force_no_tools:
                record["warnings"].append("provider returned tool_calls after the tool-call limit; final answer was forced")
            final_content = content
            break

        record["retry_count"] = total_retries
        cleaned_response, redacted = _redact_private_reasoning(final_response)
        record["raw_response"] = cleaned_response
        if redacted:
            record["warnings"].append("provider private reasoning fields were redacted from raw_response")
        record["raw_output"] = final_content
        if forced_final:
            record["warnings"].append("tool-call budget reached; requested a no-tools final response")

        parsed: dict[str, Any] | None = None
        validation_errors: list[str] = []
        try:
            parsed = parse_json_object(final_content)
            validation_errors = adapter.validate(payload_dict, parsed)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            validation_errors = [f"invalid final JSON: {exc}"]

        # One bounded repair turn is optional and still cannot call tools.
        if validation_errors and config.repair_invalid:
            # Do not continue the full tool-call transcript for repair.  A
            # number of gateways (notably reasoning-first DeepSeek variants)
            # will keep emitting pseudo tool calls when an assistant/tool
            # history is present, even if tools are omitted from the repair
            # request.  Start a short, clean finalization conversation instead;
            # the public payload and bounded tool summaries are sufficient
            # context, and hidden truth is still never mounted here.
            tool_evidence: list[str] = []
            for item in record.get("tool_trace", []):
                if not isinstance(item, Mapping):
                    continue
                result = item.get("result")
                if not isinstance(result, Mapping):
                    continue
                stdout = _bounded(str(result.get("stdout") or ""), 2_000)
                stderr = _bounded(str(result.get("stderr") or ""), 800)
                if stdout or stderr:
                    tool_evidence.append(
                        f"tool {item.get('name')} output:\n{stdout}\n{stderr}"
                    )
            evidence_text = "\n\n".join(tool_evidence)
            repair_messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        prompt
                        + "\n\nFINALIZATION MODE: the allowed analysis step is over. "
                        "Do not call tools, emit tool tags, XML, Markdown, or code. "
                        "Return the required JSON object directly, even if some "
                        "statistics are unavailable."
                    ),
                },
                {"role": "user", "content": payload_content},
                {
                    "role": "user",
                    "content": (
                        "The analysis step has already finished. Use the bounded "
                        "public tool notes below only as evidence. Return only one "
                        "corrected JSON object matching the task contract. Validation "
                        "errors to fix. Every field whose schema says array must be "
                        "a JSON array (for example input_refs and uncertainties), "
                        "not a comma-separated string; do not add explanatory "
                        "prose or tool/XML tags. Validation errors: "
                        + _bounded(validation_errors, 2_000)
                        + ("\n\nPublic tool notes:\n" + _bounded(evidence_text, 6_000) if evidence_text else "")
                        + (
                            "\n\nThe previous candidate (possibly invalid) was:\n"
                            + _bounded(final_content, 12_000)
                            if final_content
                            else ""
                        )
                    ),
                },
            ]
            repair_body = _base_request_body(
                config,
                repair_messages,
                tool_choice="none",
                # A repair is necessarily a no-tools turn.  Omitting the
                # declarations is more portable than relying on gateways to
                # honor tool_choice=none.
                include_tools=False,
                json_mode=json_mode_enabled,
            )
            repair_raw, repair_retries = _call_with_retry(endpoint, api_key, repair_body, config, request_fn)
            record["repair_count"] = 1
            record["retry_count"] += repair_retries
            cleaned_repair, redacted = _redact_private_reasoning(repair_raw)
            record["raw_response"] = cleaned_repair
            if redacted:
                record["warnings"].append("provider private reasoning fields were redacted from raw_response")
            record["raw_output"] = _content(_message(repair_raw))
            repair_summary = _reasoning_summary(repair_raw, _message(repair_raw))
            if repair_summary:
                record["provider_reasoning_summary"] = repair_summary
            record["conversation_turns"].append(
                {
                    "turn": len(record["conversation_turns"]),
                    "repair": True,
                    "request_sha256": _sha256(repair_body),
                    "finish_reason": _first_choice(repair_raw).get("finish_reason"),
                    "content_chars": len(str(record["raw_output"] or "")),
                    "reasoning_chars": len(repair_summary or ""),
                    "usage": _usage(repair_raw),
                    "tool_calls": [],
                }
            )
            if config.record_raw_turns:
                record["raw_turns"].append(cleaned_repair)
            try:
                parsed = parse_json_object(record["raw_output"])
                validation_errors = adapter.validate(payload_dict, parsed)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                validation_errors = [f"invalid repaired JSON: {exc}"]

        record["parsed_response"] = parsed
        record["reasoning"] = (
            parsed.get("reasoning_trace")
            if isinstance(parsed, Mapping) and isinstance(parsed.get("reasoning_trace"), Mapping)
            else None
        )
        record["validation"] = {
            "valid": not validation_errors,
            "errors": validation_errors,
        }
        record["metrics"] = adapter.score(payload_dict, parsed, truth)
        if validation_errors:
            record["error"] = "; ".join(validation_errors)
        elif (
            truth is not None
            and isinstance(record["metrics"], Mapping)
            and (
                record["metrics"].get("metric_error")
                or record["metrics"].get("schema_valid") is False
            )
        ):
            metric_error = record["metrics"].get("metric_error") or "local evaluator rejected the response"
            record["error"] = f"metric evaluation failed: {metric_error}"
        else:
            record["status"] = "success"
    except Exception as exc:  # noqa: BLE001 - one case must not cancel siblings
        record["error"] = _redact_secret_text(f"{type(exc).__name__}: {exc}", api_key)
        record["status"] = "failed"
    finally:
        if scratch is not None:
            try:
                artifact_info = scratch.finalize(artifact_destination)
                record["artifact_summary"] = artifact_info
            except Exception as exc:  # noqa: BLE001 - retain primary case outcome
                record["warnings"].append(f"artifact finalization failed: {type(exc).__name__}: {exc}")
        record["finished_at"] = _now()
        record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        record["usage"] = _aggregate_usage(record.get("conversation_turns") or [])
    return record
