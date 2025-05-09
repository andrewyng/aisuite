from .provider import ProviderFactory
from .base_client import BaseClient
import os
from .utils.tools import Tools
from .tool_runner import ToolRunner


class Client(BaseClient):
    def __init__(self, provider_configs: dict = {}):
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
        """
        super().__init__(provider_configs, is_async=False)

    def configure(self, provider_configs: dict = None):
        super().configure(provider_configs, False)

    @property
    def chat(self):
        """Return the chat API interface."""
        if not self._chat:
            self._chat = Chat(self)
        return self._chat


class Chat:
    def __init__(self, client: "Client"):
        self.client = client
        self._completions = Completions(self.client)

    @property
    def completions(self):
        """Return the completions interface."""
        return self._completions


class Completions:
    def __init__(self, client: "Client"):
        self.client = client

    def _extract_thinking_content(self, response):
        """
        Extract content between <think> tags if present and store it in reasoning_content.

        Args:
            response: The response object from the provider

        Returns:
            Modified response object
        """
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content") and message.content:
                content = message.content.strip()
                if content.startswith("<think>") and "</think>" in content:
                    # Extract content between think tags
                    start_idx = len("<think>")
                    end_idx = content.find("</think>")
                    thinking_content = content[start_idx:end_idx].strip()

                    # Store the thinking content
                    message.reasoning_content = thinking_content

                    # Remove the think tags from the original content
                    message.content = content[end_idx + len("</think>") :].strip()

        return response

    def create(self, model: str, messages: list, **kwargs):
        """
        Create chat completion based on the model, messages, and any extra arguments.
        Supports automatic tool execution when max_turns is specified.
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
                provider_key, config
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")

        # Extract tool-related parameters
        max_turns = kwargs.pop("max_turns", None)
        tools = kwargs.get("tools", None)
        automatic_tool_calling = kwargs.get("automatic_tool_calling", False)

        # Check environment variable before allowing multi-turn tool execution
        if max_turns is not None and tools is not None:
            tool_runner = ToolRunner(provider, model_name, messages.copy(), tools, max_turns, automatic_tool_calling)
            return tool_runner.run(
                provider,
                model_name,
                messages.copy(),
                tools,
                max_turns,
            )

        # Default behavior without tool execution
        # Delegate the chat completion to the correct provider's implementation
        response = provider.chat_completions_create(model_name, messages, **kwargs)
        return self._extract_thinking_content(response)
