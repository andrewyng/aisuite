from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import os
import functools


class LLMError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message):
        super().__init__(message)


class Provider(ABC):
    @abstractmethod
    def chat_completions_create(self, model, messages):
        """Abstract method for chat completion calls, to be implemented by each provider."""
        pass


class ProviderFactory:
    """Factory for creating provider instances."""

    @staticmethod
    def create_provider(provider_key: str, config: dict = None, async_mode: bool = False):
        """Create a provider instance.

        Args:
            provider_key: The key identifying the provider
            config: Configuration for the provider
            async_mode: If True, returns async version of the provider

        Returns:
            Provider instance
        """
        if config is None:
            config = {}

        # Map provider keys to their proper class name prefixes
        provider_class_names = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "google": "Google",
            "aws": "AWS",
            "azure": "Azure",
            "groq": "Groq",
            "mistral": "Mistral",
            "ollama": "Ollama",
            "huggingface": "HuggingFace",
            "together": "Together",
            "fireworks": "Fireworks"
        }

        provider_key = provider_key.lower().strip()
        if provider_key not in provider_class_names:
            raise ValueError(f"Unsupported provider: {provider_key}")

        provider_class_prefix = provider_class_names[provider_key]
        
        try:
            provider_module = importlib.import_module(
                f".providers.{'async_' if async_mode else ''}{provider_key}_provider",
                package="aisuite",
            )
            provider_class = getattr(
                provider_module,
                f"{'Async' if async_mode else ''}{provider_class_prefix}Provider",
            )
            return provider_class(**config)
        except ImportError as e:
            raise ImportError(
                f"Could not import provider module for {provider_key}: {str(e)}. "
                "Please ensure the provider is supported."
            )
        except AttributeError as e:
            raise AttributeError(
                f"Could not find provider class for {provider_key}. "
                f"Expected class name: {'Async' if async_mode else ''}{provider_class_prefix}Provider"
            )

    @staticmethod
    def get_supported_providers():
        """Get a list of supported providers."""
        provider_dir = os.path.join(os.path.dirname(__file__), "providers")
        providers = []
        for file in os.listdir(provider_dir):
            if file.endswith("_provider.py") and not file.startswith("async_"):
                providers.append(file.replace("_provider.py", ""))
        return providers
