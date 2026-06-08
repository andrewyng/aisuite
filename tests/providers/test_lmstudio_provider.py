from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aisuite.providers.lmstudio_provider import LmstudioProvider


def test_init_points_at_openai_compatible_v1_endpoint(monkeypatch):
    monkeypatch.delenv("LMSTUDIO_API_URL", raising=False)
    provider = LmstudioProvider()
    assert "localhost:1234/v1" in str(provider.client.base_url)
    assert provider.client.api_key == "lmstudio"


def test_init_honors_api_url_override(monkeypatch):
    monkeypatch.setenv("LMSTUDIO_API_URL", "http://remote-host:4321")
    provider = LmstudioProvider()
    assert "remote-host:4321/v1" in str(provider.client.base_url)


def test_completion_surfaces_tool_calls(monkeypatch):
    """Tool calls must survive (the bug this change fixes)."""
    monkeypatch.delenv("LMSTUDIO_API_URL", raising=False)
    provider = LmstudioProvider()
    messages = [{"role": "user", "content": "Weather in SF?"}]
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments="{}"),
    )
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant", content=None, tool_calls=[tool_call]
                ),
            )
        ]
    )

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ):
        response = provider.chat_completions_create(model="qwen2.5", messages=messages)

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
