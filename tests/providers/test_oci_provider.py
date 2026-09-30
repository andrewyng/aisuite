"""
Author: L. Saetta
Date last modified: 2026-07-29
License: MIT
Description: Unit tests for OCI Generative AI provider configuration and requests.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aisuite.provider import LLMError
from aisuite.providers.oci_provider import OciProvider


@pytest.fixture(autouse=True)
def oci_environment(monkeypatch):
    """Set safe OCI environment defaults for provider construction."""
    monkeypatch.setenv("OCI_GENAI_REGION", "eu-frankfurt-1")
    monkeypatch.setenv("OCI_GENAI_API_KEY", "test-oci-api-key")
    monkeypatch.delenv("OCI_GENAI_PROJECT", raising=False)


@patch("aisuite.providers.oci_provider.openai.AsyncOpenAI")
@patch("aisuite.providers.oci_provider.openai.OpenAI")
def test_api_key_configuration_uses_oci_endpoint(mock_openai, mock_async_openai):
    """API-key mode must configure both SDK clients against OCI's endpoint."""
    OciProvider()

    expected_url = (
        "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com/openai/v1"
    )
    assert mock_openai.call_args.kwargs == {
        "api_key": "test-oci-api-key",
        "base_url": expected_url,
    }
    assert mock_async_openai.call_args.kwargs == {
        "api_key": "test-oci-api-key",
        "base_url": expected_url,
    }


@patch("aisuite.providers.oci_provider.openai.AsyncOpenAI")
@patch("aisuite.providers.oci_provider.openai.OpenAI")
def test_explicit_api_key_and_configuration_override_environment(
    mock_openai, mock_async_openai
):
    """Explicit provider settings take precedence over environment defaults."""
    OciProvider(
        api_key="configured-key",
        region="us-chicago-1",
        project="ocid1.generativeaiproject.example",
    )

    assert mock_openai.call_args.kwargs["api_key"] == "configured-key"
    assert (
        mock_openai.call_args.kwargs["project"] == "ocid1.generativeaiproject.example"
    )
    assert "us-chicago-1" in mock_async_openai.call_args.kwargs["base_url"]


@patch("aisuite.providers.oci_provider.openai.AsyncOpenAI")
@patch("aisuite.providers.oci_provider.openai.OpenAI")
@patch("aisuite.providers.oci_provider.httpx.AsyncClient")
@patch("aisuite.providers.oci_provider.httpx.Client")
@patch("oci_genai_auth.OciResourcePrincipalAuth")
def test_resource_principal_uses_authenticated_sync_and_async_clients(
    mock_auth,
    mock_httpx_client,
    mock_httpx_async_client,
    mock_openai,
    mock_async_openai,
):
    """Resource Principal mode must give each SDK client the proper HTTPX type."""
    sync_http_client = MagicMock()
    async_http_client = MagicMock()
    mock_httpx_client.return_value = sync_http_client
    mock_httpx_async_client.return_value = async_http_client

    OciProvider(auth_type="resource_principal")

    assert mock_auth.call_count == 2
    assert mock_openai.call_args.kwargs["api_key"] == "not-used"
    assert mock_openai.call_args.kwargs["http_client"] is sync_http_client
    assert mock_async_openai.call_args.kwargs["http_client"] is async_http_client


def test_missing_region_without_base_url_fails(monkeypatch):
    """A region is required when no endpoint override is configured."""
    monkeypatch.delenv("OCI_GENAI_REGION")
    with pytest.raises(ValueError, match="OCI region is missing"):
        OciProvider()


def test_invalid_authentication_type_fails():
    """Unsupported authentication modes fail before creating SDK clients."""
    with pytest.raises(ValueError, match="Invalid OCI auth_type"):
        OciProvider(auth_type="instance_principal")


@patch("aisuite.providers.oci_provider.openai.AsyncOpenAI")
@patch("aisuite.providers.oci_provider.openai.OpenAI")
def test_chat_request_is_forwarded_in_openai_shape(mock_openai, mock_async_openai):
    """Chat calls preserve aisuite's standardized request shape."""
    provider = OciProvider()
    response = MagicMock()
    provider.client.chat.completions.create.return_value = response

    actual = provider.chat_completions_create(
        "meta.llama-3.3-70b-instruct",
        [{"role": "user", "content": "Hello"}],
        temperature=0.2,
    )

    assert actual is response
    assert provider.client.chat.completions.create.call_args.kwargs == {
        "model": "meta.llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.2,
    }


@patch("aisuite.providers.oci_provider.openai.AsyncOpenAI")
@patch("aisuite.providers.oci_provider.openai.OpenAI")
def test_chat_error_is_normalized(mock_openai, mock_async_openai):
    """SDK failures are reported as aisuite LLM errors."""
    provider = OciProvider()
    provider.client.chat.completions.create.side_effect = RuntimeError(
        "OCI rejected request"
    )

    with pytest.raises(LLMError, match="OCI chat completion failed"):
        provider.chat_completions_create(
            "model", [{"role": "user", "content": "Hello"}]
        )


@patch("aisuite.providers.oci_provider.openai.AsyncOpenAI")
@patch("aisuite.providers.oci_provider.openai.OpenAI")
def test_async_chat_request_is_forwarded(mock_openai, mock_async_openai):
    """Async requests use the asynchronous OCI OpenAI client."""
    provider = OciProvider()
    response = MagicMock()
    provider.aclient.chat.completions.create = AsyncMock(return_value=response)

    actual = asyncio.run(
        provider.achat_completions_create(
            "model", [{"role": "user", "content": "Hello"}]
        )
    )

    assert actual is response
    assert provider.aclient.chat.completions.create.call_args.kwargs["model"] == "model"
