from typing import Dict, Any, Optional
from .provider import ProviderFactory


class Client:
    def __init__(self, provider_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the client with provider configurations using the ProviderFactory.

        Args:
            provider_configs (Optional[Dict[str, Dict[str, Any]]]): 
                A dictionary containing provider configurations.
                Example:
                    {
                        "openai": {"api_key": "your_openai_api_key"},
                        "aws-bedrock": {
                            "aws_access_key": "your_aws_access_key",
                            "aws_secret_key": "your_aws_secret_key",
                            "aws_region": "us-west-2"
                        }
                    }
        """
        self.provider_configs = provider_configs or {}
        self.providers = {}
        self._chat = None
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize providers using the ProviderFactory."""
        for provider_key, config in self.provider_configs.items():
            self._validate_provider_key(provider_key)
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )

    @staticmethod
    def _validate_provider_key(provider_key: str):
        """Validate if the provider key corresponds to a supported provider."""
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Ensure the model string is formatted as 'provider:model'."
            )

    def configure(self, provider_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Configure or update provider configurations.

        Args:
            provider_configs (Optional[Dict[str, Dict[str, Any]]]): New provider configurations.
        """
        if provider_configs:
            self.provider_configs.update(provider_configs)
            self._initialize_providers()

    @property
    def chat(self):
        """Return the chat API interface, initializing it lazily."""
        if not self._chat:
            self._chat = Chat(self)
        return self._chat


class Chat:
    def __init__(self, client: Client):
        """Initialize Chat with a reference to the Client."""
        self.client = client
        self._completions = None

    @property
    def completions(self):
        """Return the completions interface, initializing it lazily."""
        if not self._completions:
            self._completions = Completions(self.client)
        return self._completions


class Completions:
    def __init__(self, client: Client):
        """Initialize Completions with a reference to the Client."""
        self.client = client

    def create(self, model: str, messages: list, **kwargs):
        """
        Create chat completion based on the model, messages, and any extra arguments.

        Args:
            model (str): Model identifier in the format 'provider:model'.
            messages (list): List of message objects.
            **kwargs: Additional arguments for the provider's chat completion.

        Returns:
            Response from the provider's chat completion.
        """
        provider_key, model_name = self._extract_provider_and_model(model)

        # Ensure provider is initialized
        if provider_key not in self.client.providers:
            self.client.providers[provider_key] = self._initialize_provider(provider_key)

        provider = self.client.providers[provider_key]
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")

        # Delegate the chat completion to the provider
        return provider.chat_completions_create(model_name, messages, **kwargs)

    def _extract_provider_and_model(self, model: str):
        """Extract provider and model from the model string."""
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'provider:model', got '{model}'."
            )
        provider_key, model_name = model.split(":", 1)
        self._validate_provider_key(provider_key)
        return provider_key, model_name

    def _initialize_provider(self, provider_key: str):
        """Initialize a provider if not already done."""
        config = self.client.provider_configs.get(provider_key, {})
        return ProviderFactory.create_provider(provider_key, config)

    @staticmethod
    def _validate_provider_key(provider_key: str):
        """Validate if the provider key corresponds to a supported provider."""
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Ensure the model string is formatted as 'provider:model'."
            )
