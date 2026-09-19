"""Tests for the Y-API provider."""

from unittest.mock import MagicMock, patch
import pytest

from aisuite.provider import ProviderFactory
from aisuite.providers.yapi_provider import YapiProvider, YAPI_BASE_URL
from aisuite.framework.chat_completion_response import ChatCompletionResponse

# Y-API fronts several vendors behind one endpoint, so model ids keep their
# vendor prefix.
SELECTED_MODEL = "deepseek/deepseek-v4-flash"


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set the Y-API key environment variable for tests.

    YAPI_API_BASE is cleared so the default-endpoint assertions do not depend on
    whether the ambient environment happens to define it.
    """
    monkeypatch.setenv("YAPI_API_KEY", "test-api-key")
    monkeypatch.delenv("YAPI_API_BASE", raising=False)


def _mock_response(mock_response, response_text_content, usage=None):
    payload = {
        "choices": [
            {"message": {"content": response_text_content, "role": "assistant"}}
        ],
        "model": SELECTED_MODEL,
        "created": 12345,
        "id": "chatcmpl-mockid",
    }
    if usage is not None:
        payload["usage"] = usage
    # The mock response from the client is an object, so we mock the
    # .model_dump() method
    mock_response.model_dump.return_value = payload
    return mock_response


def test_yapi_provider_defaults_to_the_gateway_endpoint():
    """Without config or env override, the provider targets the Y-API gateway."""
    provider = YapiProvider()

    assert str(provider.client.base_url).rstrip("/") == YAPI_BASE_URL


def test_yapi_provider_honours_a_base_url_override():
    """base_url in config wins, so an alternative gateway can be pointed at."""
    provider = YapiProvider(base_url="https://gateway.example.com/v1")

    assert str(provider.client.base_url).rstrip("/") == "https://gateway.example.com/v1"


def test_yapi_provider_reads_base_url_from_the_environment(monkeypatch):
    """YAPI_API_BASE is used when no base_url is passed in config."""
    monkeypatch.setenv("YAPI_API_BASE", "https://env.example.com/v1")
    provider = YapiProvider()

    assert str(provider.client.base_url).rstrip("/") == "https://env.example.com/v1"


def test_yapi_provider_requires_an_api_key(monkeypatch):
    """A missing key is a clear ValueError, not a confusing SDK error."""
    monkeypatch.delenv("YAPI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Y-API key is missing"):
        YapiProvider()


def test_yapi_provider():
    """Test that the provider is initialized and chat completions are requested."""

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    chosen_temperature = 0.75
    response_text_content = "mocked-text-response-from-model"

    provider = YapiProvider()
    mock_response = _mock_response(MagicMock(), response_text_content)

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        response = provider.chat_completions_create(
            messages=message_history,
            model=SELECTED_MODEL,
            temperature=chosen_temperature,
        )

        mock_create.assert_called_once_with(
            messages=message_history,
            model=SELECTED_MODEL,
            temperature=chosen_temperature,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.choices[0].message.content == response_text_content
        assert response.usage is None


def test_yapi_provider_with_usage():
    """Tests that usage data is correctly parsed when present in the response."""

    message_history = [{"role": "user", "content": "Hello!"}]
    response_text_content = "mocked-text-response-from-model"

    provider = YapiProvider()
    mock_response = _mock_response(
        MagicMock(),
        response_text_content,
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )

    with patch.object(
        provider.client.chat.completions, "create", return_value=mock_response
    ):
        response = provider.chat_completions_create(
            messages=message_history,
            model=SELECTED_MODEL,
            temperature=0.75,
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20
        assert response.usage.total_tokens == 30


def test_yapi_provider_streams_openai_shaped_chunks():
    """Streaming passes the gateway's chunks through unchanged."""

    provider = YapiProvider()
    chunks = ["chunk-one", "chunk-two"]

    with patch.object(
        provider.client.chat.completions,
        "create",
        return_value=iter(chunks),
    ) as mock_create:
        received = list(
            provider.chat_completions_create_stream(
                messages=[{"role": "user", "content": "Hello!"}],
                model=SELECTED_MODEL,
            )
        )

    assert received == chunks
    mock_create.assert_called_once()
    assert mock_create.call_args.kwargs["stream"] is True


def test_yapi_provider_is_discoverable_by_the_factory():
    """The factory finds the provider purely by filename convention.

    `yapi_provider.py` -> "yapi" -> class `YapiProvider`. A rename that breaks
    this convention would not fail any import, so it is asserted here.
    """
    provider = ProviderFactory.create_provider("yapi", {})

    assert isinstance(provider, YapiProvider)


def test_yapi_provider_is_listed_as_supported():
    """`get_supported_providers()` globs the providers directory."""
    assert "yapi" in ProviderFactory.get_supported_providers()
