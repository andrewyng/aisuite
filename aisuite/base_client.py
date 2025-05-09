from .provider import ProviderFactory
from abc import ABC, abstractproperty

class BaseClient(ABC):
    def __init__(self, provider_configs: dict = {}, is_async: bool = False):
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
        self.providers = {}
        self.provider_configs = provider_configs
        self._chat = None
        self._initialize_providers(is_async)


    def _initialize_providers(self, is_async):
        """Helper method to initialize or update providers."""
        for provider_key, config in self.provider_configs.items():
            provider_key = self._validate_provider_key(provider_key)
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config, is_async
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

    def configure(self, provider_configs: dict = None, is_async: bool = False):
        """
        Configure the client with provider configurations.
        """
        if provider_configs is None:
            return

        self.provider_configs.update(provider_configs)
        self._initialize_providers(is_async)  # NOTE: This will override existing provider instances.


    @property
    @abstractproperty
    def chat(self):
        """Return the chat API interface."""
        raise NotImplementedError("Chat is not implemented for this client.")
