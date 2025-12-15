from unittest.mock import patch, MagicMock

import pytest

from aisuite.providers.ollama_provider import OllamaProvider


@pytest.fixture(autouse=True)
def set_api_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_URL", "http://localhost:11434/v1")


def test_parsing_with_tool_call():
    provider_under_test = OllamaProvider()

    mock_tool_call_response = MagicMock()
    mock_tool_call_response.model_dump.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {
                                "name": "will_it_rain",
                                "arguments": '{"location": "San Francisco", "time_of_day": "2pm"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    with patch.object(
        provider_under_test.client.chat.completions,
        "create",
        return_value=mock_tool_call_response,
    ) as mock_create:

        response = provider_under_test.chat_completions_create(
            model="ollama:qwen3:4b",
            messages=[
                {"role": "user", "content": "Will it rain in San Francisco at 2pm?"}
            ],
        )

        assert mock_create.call_count == 1
        assert response.choices[0].message.content is None
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].message.tool_calls[0].type == "function"
        assert response.choices[0].message.tool_calls[0].function.name == "will_it_rain"


def test_parsing_without_tool_call():
    provider_under_test = OllamaProvider()

    mock_without_tool_call_response = MagicMock()
    mock_without_tool_call_response.model_dump.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "I can not help you to determine real time weather!",
                }
            }
        ]
    }

    with patch.object(
        provider_under_test.client.chat.completions,
        "create",
        return_value=mock_without_tool_call_response,
    ) as mock_create:

        response = provider_under_test.chat_completions_create(
            model="ollama:qwen3:4b",
            messages=[
                {"role": "user", "content": "Will it rain in San Francisco at 2pm?"}
            ],
        )

        assert mock_create.call_count == 1
        assert (
            response.choices[0].message.content
            == "I can not help you to determine real time weather!"
        )
        assert response.choices[0].message.tool_calls is None


def test_base_url_normalisation():
    ollama = OllamaProvider(base_url="http://localhost:8080")
    assert ollama.client.base_url == "http://localhost:8080/v1/"

    ollama = OllamaProvider(base_url="http://localhost:8080/")
    assert ollama.client.base_url == "http://localhost:8080/v1/"

    ollama = OllamaProvider(base_url="http://localhost:8080/v1")
    assert ollama.client.base_url == "http://localhost:8080/v1/"

    ollama = OllamaProvider(api_url="http://localhost:8080/v1")
    assert ollama.client.base_url == "http://localhost:8080/v1/"

    ollama = OllamaProvider()
    assert ollama.client.base_url == "http://localhost:11434/v1/"
