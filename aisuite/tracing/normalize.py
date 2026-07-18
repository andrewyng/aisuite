from __future__ import annotations

import copy
import json
from typing import Any

DEFAULT_PREVIEW_CHARS = 200


def preview_text(value: Any, max_chars: int = DEFAULT_PREVIEW_CHARS) -> dict[str, Any]:
    text = "" if value is None else str(value)
    return {
        "text_preview": text[:max_chars],
        "text_length": len(text),
        "truncated": len(text) > max_chars,
        "content_redacted": False,
    }


def normalize_model_input(
    messages: list[Any],
    *,
    model: str,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any]:
    items = [_normalize_message(message, preview_chars) for message in messages]
    modalities = sorted({modality for item in items for modality in item["modalities"]})
    return {
        "model": model,
        "input": {
            "message_count": len(messages),
            "modalities": modalities,
            "items": items,
        },
    }


def normalize_model_response(
    response: Any,
    *,
    model: str,
    preview_chars: int = DEFAULT_PREVIEW_CHARS,
) -> dict[str, Any]:
    message = _response_message(response)
    text = _message_content(message)
    tool_calls = _message_tool_calls(message)
    text_payload = preview_text(text, preview_chars) if text else {}
    modalities = []
    if text:
        modalities.append("text")
    if tool_calls:
        modalities.append("tool_call")
    if text and tool_calls:
        kind = "mixed"
    elif tool_calls:
        kind = "tool_calls"
    elif text:
        kind = "text"
    else:
        kind = "empty"
    return {
        "model": model,
        "response": {
            "kind": kind,
            "modalities": modalities,
            **text_payload,
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "finish_reason": _finish_reason(response),
        },
        "usage": normalize_usage(getattr(response, "usage", None)),
    }


def _normalize_message(message: Any, preview_chars: int) -> dict[str, Any]:
    message_dict = _to_dict(message)
    role = message_dict.get("role", "message")
    item_type = {
        "system": "system_message",
        "user": "user_message",
        "assistant": "assistant_message",
        "tool": "tool_result",
    }.get(role, f"{role}_message")
    content = message_dict.get("content")
    modalities = _content_modalities(content)
    item = {
        "type": item_type,
        "role": role,
        "modalities": modalities,
    }
    text = _content_text(content)
    if text:
        item.update(preview_text(text, preview_chars))
    if role == "tool":
        item["tool_name"] = message_dict.get("name")
        item["tool_call_id"] = message_dict.get("tool_call_id")
    if message_dict.get("tool_calls"):
        item["tool_call_count"] = len(message_dict["tool_calls"])
        if "tool_call" not in item["modalities"]:
            item["modalities"].append("tool_call")
    return item


def _content_modalities(content: Any) -> list[str]:
    modalities = set()
    if isinstance(content, str):
        if content:
            modalities.add("text")
    elif isinstance(content, list):
        for part in content:
            part_dict = _to_dict(part)
            part_type = part_dict.get("type")
            if part_type in {"text", "input_text"}:
                modalities.add("text")
            elif part_type in {"image", "image_url", "input_image"}:
                modalities.add("image")
            elif part_type:
                modalities.add(str(part_type))
    elif content is not None:
        modalities.add("structured")
    return sorted(modalities)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            part_dict = _to_dict(part)
            text = part_dict.get("text") or part_dict.get("content")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return ""


def _response_message(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "message", None)


def _message_content(message: Any) -> str:
    if message is None:
        return ""
    return _content_text(getattr(message, "content", None))


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    if message is None:
        return []
    tool_calls = getattr(message, "tool_calls", None) or []
    normalized = []
    for call in tool_calls:
        call_dict = _to_dict(call)
        function = call_dict.get("function") or {}
        normalized.append(
            {
                "id": call_dict.get("id"),
                "name": function.get("name"),
                "arguments": _safe_arguments(function.get("arguments")),
            }
        )
    return normalized


def _finish_reason(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)


def normalize_usage(usage: Any) -> Any:
    if usage is None:
        return None
    raw = _to_dict(usage)
    input_tokens = _first_int(raw, "input_tokens", "prompt_tokens")
    output_tokens = _first_int(raw, "output_tokens", "completion_tokens")
    total_tokens = _first_int(raw, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens": _first_int(raw, "prompt_tokens", "input_tokens"),
        "completion_tokens": _first_int(raw, "completion_tokens", "output_tokens"),
        "provider_raw": raw,
    }


def _first_int(data: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _safe_arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return copy.deepcopy(value)


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "__dict__"):
        return {
            key: copy.deepcopy(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_") and item is not None
        }
    return {"value": value}
