"""Tests for ModelsLab provider functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from aisuite.providers.modelslab_provider import ModelslabProvider


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("MODELSLAB_API_KEY", "test-api-key")


@pytest.fixture
def modelslab_provider():
    """Create a ModelsLab provider instance for testing."""
    return ModelslabProvider()


class TestModelslabProvider:
    """Test suite for ModelsLab provider functionality."""

    def test_provider_initialization(self, modelslab_provider):
        """Test that ModelsLab provider initializes correctly."""
        assert modelslab_provider is not None
        assert modelslab_provider.api_key == "test-api-key"
        assert modelslab_provider.base_url == "https://api.modelslab.com/v1"

    def test_provider_initialization_with_custom_base_url(self):
        """Test initialization with custom base URL."""
        provider = ModelslabProvider(
            api_key="custom-key",
            base_url="https://custom.endpoint.com/v1"
        )
        assert provider.api_key == "custom-key"
        assert provider.base_url == "https://custom.endpoint.com/v1"

    def test_provider_initialization_missing_api_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("MODELSLAB_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ModelsLab API key is missing"):
            ModelslabProvider()

    def test_chat_completions_create_success(self, modelslab_provider):
        """Test successful chat completion creation."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?"
                },
                "finish_reason": "stop"
            }]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            messages = [
                {"role": "user", "content": "Hello"}
            ]
            result = modelslab_provider.chat_completions_create(
                model="gpt-4o",
                messages=messages
            )
            
            assert result["id"] == "chatcmpl-123"
            assert result["choices"][0]["message"]["content"] == "Hello! How can I help you?"
            
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "https://api.modelslab.com/v1/chat/completions" in call_args[0]

    def test_chat_completions_with_temperature(self, modelslab_provider):
        """Test chat completion with temperature parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop"
            }]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            messages = [{"role": "user", "content": "Hello"}]
            result = modelslab_provider.chat_completions_create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            assert payload["temperature"] == 0.7

    def test_chat_completions_with_max_tokens(self, modelslab_provider):
        """Test chat completion with max_tokens parameter."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop"
            }]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            messages = [{"role": "user", "content": "Hello"}]
            result = modelslab_provider.chat_completions_create(
                model="gpt-4o",
                messages=messages,
                max_tokens=100
            )
            
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            assert payload["max_tokens"] == 100

    def test_chat_completions_api_error(self, modelslab_provider):
        """Test error handling for API failures."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("requests.post", return_value=mock_response):
            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(Exception) as exc_info:
                modelslab_provider.chat_completions_create(
                    model="gpt-4o",
                    messages=messages
                )
            assert "ModelsLab API error" in str(exc_info.value)

    def test_chat_completions_with_stream(self, modelslab_provider):
        """Test chat completion with streaming."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None
            }]
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            messages = [{"role": "user", "content": "Hello"}]
            result = modelslab_provider.chat_completions_create(
                model="gpt-4o",
                messages=messages,
                stream=True
            )
            
            call_kwargs = mock_post.call_args.kwargs
            payload = call_kwargs["json"]
            assert payload["stream"] is True