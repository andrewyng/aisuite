"""
Author: L. Saetta
Date last modified: 2026-07-29
License: MIT
Description: OCI Generative AI provider for the OpenAI-compatible Chat Completions API.
"""

import os
from typing import Any

import httpx
import openai

from aisuite.provider import LLMError, Provider
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class OciProvider(Provider):
    """Call OCI Generative AI's OpenAI-compatible Chat Completions endpoint.

    The provider accepts an OCI Generative AI API key or OCI Resource Principal
    authentication. It deliberately uses the OpenAI SDK because OCI's endpoint
    implements the OpenAI Chat Completions contract that aisuite exposes.

    Args:
        **config: Provider configuration. `region` or `base_url` is required.
            `auth_type` is `api_key` (default) or `resource_principal`.
            API-key mode reads `api_key` or `OCI_GENAI_API_KEY`; Resource
            Principal mode uses the OCI runtime credentials.

    Raises:
        ImportError: If Resource Principal support is requested without the
            optional `oci-genai-auth` dependency.
        ValueError: If configuration is missing or invalid.
    """

    def __init__(self, **config: Any):
        """Initialize OCI clients for synchronous and asynchronous requests.

        Args:
            **config: Provider configuration. Supported options include `region`,
                `base_url`, `auth_type`, `api_key`, and `project`. API-key mode
                reads missing credentials from the OCI_GENAI_* environment
                variables.

        Raises:
            ImportError: If Resource Principal authentication is requested without
                the optional OCI authentication dependency.
            ValueError: If configuration is missing or invalid.
        """
        super().__init__()
        config = dict(config)
        auth_type = config.pop("auth_type", "api_key").lower()
        base_url = self._resolve_base_url(
            config.pop("base_url", None), config.pop("region", None)
        )
        project = config.pop("project", os.getenv("OCI_GENAI_PROJECT"))

        client_config = {"base_url": base_url, **config}
        if project:
            client_config["project"] = project

        if auth_type == "api_key":
            api_key = client_config.pop("api_key", None) or os.getenv(
                "OCI_GENAI_API_KEY"
            )
            if not api_key:
                raise ValueError(
                    "OCI Generative AI API key is missing. Provide `api_key` in "
                    "provider configuration or set OCI_GENAI_API_KEY."
                )
            self.client = openai.OpenAI(api_key=api_key, **client_config)
            self.aclient = openai.AsyncOpenAI(api_key=api_key, **client_config)
        elif auth_type == "resource_principal":
            if "api_key" in client_config:
                raise ValueError(
                    "`api_key` cannot be used with resource_principal authentication."
                )
            self.client, self.aclient = self._create_resource_principal_clients(
                client_config
            )
        else:
            raise ValueError(
                "Invalid OCI auth_type {!r}. Supported values are `api_key` and "
                "`resource_principal`.".format(auth_type)
            )

        self.transformer = OpenAICompliantMessageConverter()

    @staticmethod
    def _resolve_base_url(base_url: str | None, region: str | None) -> str:
        """Return the OCI OpenAI-compatible base URL.

        Args:
            base_url: Explicit endpoint override.
            region: OCI region identifier.

        Returns:
            The endpoint base URL without a trailing slash.

        Raises:
            ValueError: If neither endpoint nor region is configured.
        """
        if base_url:
            return base_url.rstrip("/")
        resolved_region = region or os.getenv("OCI_GENAI_REGION")
        if not resolved_region:
            raise ValueError(
                "OCI region is missing. Provide `region` in provider configuration "
                "or set OCI_GENAI_REGION."
            )
        return f"https://inference.generativeai.{resolved_region}.oci.oraclecloud.com/openai/v1"

    @staticmethod
    def _create_resource_principal_clients(client_config: dict[str, Any]):
        """Create SDK clients signed with OCI Resource Principal credentials.

        Args:
            client_config: OpenAI client configuration shared by both clients.

        Returns:
            A synchronous and an asynchronous OpenAI SDK client.

        Raises:
            ImportError: If the OCI authentication helper is unavailable.
        """
        try:
            from oci_genai_auth import OciResourcePrincipalAuth
        except ImportError as exc:
            raise ImportError(
                "OCI Resource Principal authentication requires `oci-genai-auth`. "
                "Install it with `pip install 'aisuite[oci]'`."
            ) from exc

        sync_client = httpx.Client(auth=OciResourcePrincipalAuth())
        async_client = httpx.AsyncClient(auth=OciResourcePrincipalAuth())
        return (
            openai.OpenAI(api_key="not-used", http_client=sync_client, **client_config),
            openai.AsyncOpenAI(
                api_key="not-used", http_client=async_client, **client_config
            ),
        )

    def chat_completions_create(self, model: str, messages: list, **kwargs: Any):
        """Create an OCI chat completion.

        Args:
            model: OCI model identifier without the `oci:` prefix.
            messages: OpenAI-shaped conversation messages.
            **kwargs: Supported Chat Completions request parameters.

        Returns:
            The OpenAI-shaped response returned by OCI.

        Raises:
            LLMError: If the OCI or OpenAI SDK request fails.
        """
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"OCI chat completion failed: {exc}") from exc

    async def achat_completions_create(self, model: str, messages: list, **kwargs: Any):
        """Asynchronously create an OCI chat completion.

        Args:
            model: OCI model identifier without the `oci:` prefix.
            messages: OpenAI-shaped conversation messages.
            **kwargs: Supported Chat Completions request parameters.

        Returns:
            The OpenAI-shaped response returned by OCI.

        Raises:
            LLMError: If the OCI or OpenAI SDK request fails.
        """
        try:
            return await self.aclient.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"OCI chat completion failed: {exc}") from exc

    def chat_completions_create_stream(self, model: str, messages: list, **kwargs: Any):
        """Yield streamed OCI Chat Completions chunks.

        Args:
            model: OCI model identifier without the `oci:` prefix.
            messages: OpenAI-shaped conversation messages.
            **kwargs: Supported Chat Completions request parameters.

        Yields:
            OpenAI-shaped streaming chunks.

        Raises:
            LLMError: If the stream cannot be created.
        """
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                stream=True,
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"OCI chat completion stream failed: {exc}") from exc
        yield from stream

    async def achat_completions_create_stream(
        self, model: str, messages: list, **kwargs: Any
    ):
        """Asynchronously yield streamed OCI Chat Completions chunks.

        Args:
            model: OCI model identifier without the `oci:` prefix.
            messages: OpenAI-shaped conversation messages.
            **kwargs: Supported Chat Completions request parameters.

        Yields:
            OpenAI-shaped streaming chunks.

        Raises:
            LLMError: If the stream cannot be created.
        """
        try:
            stream = await self.aclient.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                stream=True,
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"OCI chat completion stream failed: {exc}") from exc
        async for chunk in stream:
            yield chunk
