"""
Asynchronous client implementation for aisuite.
"""
from typing import Dict, Optional, List
import asyncio
import httpx

from .provider import ProviderFactory
from .framework import ChatCompletionResponse

class AsyncCompletions:
    def __init__(self, client: "AsyncClient"):
        self.client = client

    async def create(self, model: str, messages: List[Dict], stream: bool = False, **kwargs):
        """
        Create chat completion based on the model, messages, and any extra arguments asynchronously.
        
        Args:
            model: Model identifier in format "provider:model"
            messages: List of message dictionaries
            stream: If True, returns an async generator of responses
            **kwargs: Additional arguments for the provider
        """
        # Check that correct format is used
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'provider:model', got '{model}'"
            )

        # Extract the provider key from the model identifier, e.g., "google:gemini-xx"
        provider_key, model_name = model.split(":", 1)

        # Validate if the provider is supported
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Make sure the model string is formatted correctly as 'provider:model'."
            )

        # Initialize provider if not already initialized
        if provider_key not in self.client.providers:
            config = self.client.provider_configs.get(provider_key, {})
            self.client.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config, async_mode=True
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")

        # Use streaming or non-streaming method based on the stream parameter
        if stream:
            return provider.async_chat_completions_create_stream(model_name, messages, **kwargs)
        else:
            return await provider.async_chat_completions_create(model_name, messages, **kwargs)

class AsyncChat:
    def __init__(self, client: "AsyncClient"):
        self.client = client
        self._completions = AsyncCompletions(self.client)

    @property
    def completions(self):
        """Return the completions interface."""
        return self._completions

class AsyncClient:
    """
    Asynchronous client for interacting with various LLM providers.
    """
    def __init__(self, provider_configs: dict = {}):
        """
        Initialize the async client with provider configurations.
        Use the ProviderFactory to create provider instances.

        Args:
            provider_configs (dict): A dictionary containing provider configurations.
                Each key should be a provider string (e.g., "google" or "aws-bedrock"),
                and the value should be a dictionary of configuration options for that provider.
        """
        self.providers = {}
        self.provider_configs = provider_configs
        self._chat = None
        self._initialize_providers()

    def _initialize_providers(self):
        """Helper method to initialize or update providers."""
        for provider_key, config in self.provider_configs.items():
            provider_key = self._validate_provider_key(provider_key)
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config, async_mode=True
            )

    def _validate_provider_key(self, provider_key: str) -> str:
        """Validate and normalize the provider key."""
        provider_key = provider_key.lower().strip()
        if provider_key not in ProviderFactory.get_supported_providers():
            raise ValueError(f"Unsupported provider: {provider_key}")
        return provider_key

    def configure(self, provider_configs: dict = None):
        """
        Configure the client with provider configurations.
        """
        if provider_configs is None:
            return

        self.provider_configs.update(provider_configs)
        self._initialize_providers()  # NOTE: This will override existing provider instances.

    @property
    def chat(self):
        """Return the chat API interface."""
        if not self._chat:
            self._chat = AsyncChat(self)
        return self._chat

    async def close(self):
        """Close all provider resources."""
        for provider in self.providers.values():
            if hasattr(provider, 'close'):
                await provider.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
