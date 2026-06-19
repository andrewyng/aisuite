from .provider import ProviderFactory
import os
from .utils.tools import Tools
from typing import Union, BinaryIO, Optional, Any, Literal
from contextlib import ExitStack
from .framework.message import (
    TranscriptionResponse,
)
from .framework.asr_params import ParamValidator
from .tracing.normalize import normalize_model_input, normalize_model_response
from .tracing.sinks import TraceEvent, emit_event

# Import MCP utilities for config dict support
try:
    from .mcp.config import is_mcp_config
    from .mcp.client import MCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class Client:
    def __init__(
        self,
        provider_configs: dict = {},
        extra_param_mode: Literal["strict", "warn", "permissive"] = "warn",
    ):
        """
        Initialize the client with provider configurations.
        Use the ProviderFactory to create provider instances.
        Args:
            provider_configs (dict): A dictionary containing provider configurations.
                Each key should be a provider string (e.g., "google" or "aws-bedrock"),
                and the value should be a dictionary of configuration options for that provider.
                For example:
                {
                    "openai": {"api_key": "your_openai_api_key"},
                    "aws-bedrock": {
                        "aws_access_key": "your_aws_access_key",
                        "aws_secret_key": "your_aws_secret_key",
                        "aws_region": "us-west-2"
                    }
                }
            extra_param_mode (str): How to handle unknown ASR parameters.
                - "strict": Raise ValueError on unknown params (production)
                - "warn": Log warning on unknown params (default, development)
                - "permissive": Allow all params without validation (testing)
        """
        self.providers = {}
        self.provider_configs = provider_configs
        self.extra_param_mode = extra_param_mode
        self.param_validator = ParamValidator(extra_param_mode)
        self._chat = None
        self._audio = None

    def _initialize_providers(self):
        """Helper method to initialize or update providers."""
        for provider_key, config in self.provider_configs.items():
            provider_key = self._validate_provider_key(provider_key)
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )

    def _validate_provider_key(self, provider_key):
        """
        Validate if the provider key corresponds to a supported provider.
        """
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Make sure the model string is formatted correctly as 'provider:model'."
            )
        return provider_key

    def _get_provider(self, provider_key):
        """Retrieve or initialize a provider."""
        provider_key = self._validate_provider_key(provider_key)
        if provider_key not in self.providers:
            config = self.provider_configs.get(provider_key, {})
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )
        return self.providers[provider_key]

    @property
    def chat(self):
        if self._chat is None:
            from .chat import Chat

            self._chat = Chat(self)
        return self._chat

    @property
    def audio(self):
        if self._audio is None:
            from .audio import Audio

            self._audio = Audio(self)
        return self._audio

    def create(self, model: str, messages: list, **kwargs):
        """
        Create a chat completion.
        Args:
            model (str): The model to use for completion.
            messages (list): A list of messages for the chat.
            **kwargs: Additional parameters for the provider.
        """
        if model is None:
            raise ValueError("The 'model' parameter cannot be None. Please provide a valid model string (e.g., 'openai:gpt-4').")

        # Handle MCP configs in messages if present
        mcp_configs = self._process_mcp_configs(messages)

        provider_key, model_name = model.split(":", 1)
        provider = self._get_provider(provider_key)

        # Normalize input
        normalized_messages = normalize_model_input(messages)

        # Emit trace event
        emit_event(TraceEvent.CHAT_COMPLETION_START, model=model, messages=messages)

        # Call provider
        response = provider.chat.completions.create(model_name, normalized_messages, **kwargs)

        # Normalize response
        normalized_response = normalize_model_response(response)

        # Emit trace event
        emit_event(TraceEvent.CHAT_COMPLETION_END, response=normalized_response)

        return response

    def _process_mcp_configs(self, messages):
        """Extract and initialize MCP clients from messages."""
        mcp_configs = []
        for message in messages:
            if isinstance(message, dict) and "mcp_config" in message:
                config = message["mcp_config"]
                if not MCP_AVAILABLE:
                    raise ImportError("The 'mcp' package is required to use MCP configurations. Please install it with 'pip install mcp'.")
                if is_mcp_config(config):
                    mcp_configs.append(config)
        return mcp_configs
