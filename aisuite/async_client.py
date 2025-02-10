from .client import Client, Chat, Completions
from .provider import ProviderFactory


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

        # Delegate the chat completion to the correct provider's async implementation
        return await provider.chat_completions_create_async(
            model_name, messages, **kwargs
        )
