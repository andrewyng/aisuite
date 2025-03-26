from .client import Client, Chat, Completions
from .provider import ProviderFactory
from .tool_runner import ToolRunner


class AsyncClient(Client):
    @property
    def chat(self):
        """Return the async chat API interface."""
        if not self._chat:
            self._chat = AsyncChat(self)
        return self._chat


class AsyncChat(Chat):
    def __init__(self, client: "AsyncClient"):
        self.client = client
        self._completions = AsyncCompletions(self.client)


class AsyncCompletions(Completions):
    async def create(self, model: str, messages: list, **kwargs):
        """
        Create async chat completion based on the model, messages, and any extra arguments.
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
                provider_key, config, is_async=True
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")

        # Extract tool-related parameters
        max_turns = kwargs.pop("max_turns", None)
        tools = kwargs.get("tools", None)

        # Check environment variable before allowing multi-turn tool execution
        if max_turns is not None and tools is not None:
            tool_runner = ToolRunner(provider, model_name, messages.copy(), tools, max_turns)
            return await tool_runner.run_async(
                provider,
                model_name,
                messages.copy(),
                tools,
                max_turns,
            )

        # Default behavior without tool execution
        # Delegate the chat completion to the correct provider's async implementation
        response = await provider.chat_completions_create_async(model_name, messages, **kwargs)
        return self._extract_thinking_content(response)
