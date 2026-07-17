from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def merge_tags(*tag_lists: Optional[list[str]]) -> list[str]:
    merged = []
    seen = set()
    for tags in tag_lists:
        for tag in tags or []:
            if tag not in seen:
                merged.append(tag)
                seen.add(tag)
    return merged


def message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return copy.deepcopy(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if hasattr(message, "__dict__"):
        return {
            key: copy.deepcopy(value)
            for key, value in message.__dict__.items()
            if not key.startswith("_") and value is not None
        }
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    return [message_to_dict(message) for message in messages]


def build_input_messages(input: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(input, str):
        return [{"role": "user", "content": input}]
    if isinstance(input, list):
        return messages_to_dicts(input)
    raise ValueError("Input must be a string, message list, or RunState.")


def extract_final_message(response: Any) -> Optional[dict[str, Any]]:
    if not hasattr(response, "choices") or not response.choices:
        return None
    message = getattr(response.choices[0], "message", None)
    if message is None:
        return None
    return message_to_dict(message)


def extract_final_output(response: Any) -> Any:
    if not hasattr(response, "choices") or not response.choices:
        return None
    message = getattr(response.choices[0], "message", None)
    return getattr(message, "content", None)


def extract_response_messages(
    response: Any, starting_messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        intermediate = getattr(choice, "intermediate_messages", None)
        if intermediate:
            return [
                *copy.deepcopy(starting_messages),
                *messages_to_dicts(intermediate),
            ]

    final_message = extract_final_message(response)
    if final_message:
        return [*copy.deepcopy(starting_messages), final_message]
    return copy.deepcopy(starting_messages)
