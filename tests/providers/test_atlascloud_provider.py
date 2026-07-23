from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aisuite.providers.atlascloud_provider import AtlascloudProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_KEY", "test-atlascloud-api-key")
    monkeypatch.delenv("ATLASCLOUD_API_BASE", raising=False)


def test_init_points_at_atlascloud_openai_compatible_endpoint():
    provider = AtlascloudProvider()
    assert str(provider.client.base_url) == "https://api.atlascloud.ai/v1/"
    assert provider.client.api_key == "test-atlascloud-api-key"


def test_init_honors_base_url_overrides(monkeypatch):
    monkeypatch.setenv("ATLASCLOUD_API_BASE", "https://proxy.example.com")
    provider = AtlascloudProvider()
    assert str(provider.client.base_url) == "https://proxy.example.com/v1/"

    provider2 = AtlascloudProvider(api_url="https://other.example.com/v1")
    assert str(provider2.client.base_url) == "https://other.example.com/v1/"


def test_init_requires_api_key(monkeypatch):
    monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Atlas Cloud API key is missing"):
        AtlascloudProvider()


def test_completion_passes_through_content():
    provider = AtlascloudProvider()
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
            model="qwen/qwen3.5-flash", messages=messages, temperature=0.7
        )

    assert response.choices[0].message.content == "hi there"
    assert mock_create.call_args.kwargs["model"] == "qwen/qwen3.5-flash"
    assert mock_create.call_args.kwargs["temperature"] == 0.7


def test_completion_surfaces_tool_calls():
    provider = AtlascloudProvider()
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
            model="qwen/qwen3.5-flash", messages=messages, tools=tools
        )

    assert response.choices[0].finish_reason == "tool_calls"
    returned = response.choices[0].message.tool_calls
    assert returned and returned[0].function.name == "get_weather"
    assert mock_create.call_args.kwargs["tools"] == tools
