from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aisuite.providers.ollama_provider import OllamaProvider


def test_init_points_at_openai_compatible_v1_endpoint(monkeypatch):
    """Ollama should drive the OpenAI SDK against the local /v1 endpoint."""
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)
    provider = OllamaProvider()
    assert "localhost:11434/v1" in str(provider.client.base_url)
    # Ollama ignores the key, but the OpenAI SDK requires one.
    assert provider.client.api_key == "ollama"


def test_init_honors_api_url_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_URL", "http://remote-host:1234")
    provider = OllamaProvider()
    assert "remote-host:1234/v1" in str(provider.client.base_url)
    # api_url is an aisuite-level key and must not leak into the OpenAI client.
    provider2 = OllamaProvider(api_url="http://other-host:9999")
    assert "other-host:9999/v1" in str(provider2.client.base_url)


def test_completion_passes_through_content(monkeypatch):
    """A plain content response flows back unchanged via the OpenAI SDK."""
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)
    provider = OllamaProvider()
    messages = [{"role": "user", "content": "Howdy!"}]
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant", content="hi there", tool_calls=None
                ),
            )
        ]
    )

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        response = provider.chat_completions_create(
            model="llama3.1", messages=messages, temperature=0.7
        )

    assert response.choices[0].message.content == "hi there"
    assert mock_create.call_args.kwargs["model"] == "llama3.1"
    assert mock_create.call_args.kwargs["temperature"] == 0.7


def test_completion_surfaces_tool_calls(monkeypatch):
    """Tool calls must survive (the bug this change fixes)."""
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)
    provider = OllamaProvider()
    messages = [{"role": "user", "content": "Weather in SF?"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {}},
        }
    ]
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(
            name="get_weather", arguments='{"city": "San Francisco"}'
        ),
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
    ) as mock_create:
        response = provider.chat_completions_create(
            model="llama3.1", messages=messages, tools=tools
        )

    assert response.choices[0].finish_reason == "tool_calls"
    returned = response.choices[0].message.tool_calls
    assert returned and returned[0].function.name == "get_weather"
    # tools are forwarded to the model unchanged.
    assert mock_create.call_args.kwargs["tools"] == tools


def test_base_url_normalisation(monkeypatch):
    """/v1 is appended idempotently across all the ways a host can be supplied."""
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)
    for kwargs in (
        {"base_url": "http://localhost:8080"},
        {"base_url": "http://localhost:8080/"},
        {"base_url": "http://localhost:8080/v1"},
        {"api_url": "http://localhost:8080/v1"},
    ):
        provider = OllamaProvider(**kwargs)
        assert str(provider.client.base_url) == "http://localhost:8080/v1/"
