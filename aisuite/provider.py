from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import os
import functools
from typing import Union, BinaryIO, Optional


class LLMError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message):
        super().__init__(message)


class ASRError(Exception):
    """Custom exception for ASR errors."""

    def __init__(self, message):
        super().__init__(message)


class Provider(ABC):
    def __init__(self):
        """Initialize provider with optional audio functionality."""
        self.audio: Optional[Audio] = None

    @abstractmethod
    def chat_completions_create(self, model, messages):
        """Abstract method for chat completion calls, to be implemented by each provider."""
        pass


class ProviderFactory:
    """Factory to dynamically load provider instances based on naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key, config):
        """Dynamically load and create an instance of a provider based on the naming convention."""
        # Convert provider_key to the expected module and class names
        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}_provider"

        module_path = f"aisuite.providers.{provider_module_name}"

        # Lazily load the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module {module_path}: {str(e)}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            )

        # Instantiate the provider class
        provider_class = getattr(module, provider_class_name)
        return provider_class(**config)

    @classmethod
    @functools.cache
    def get_supported_providers(cls):
        """List all supported provider names based on files present in the providers directory."""
        provider_files = Path(cls.PROVIDERS_DIR).glob("*_provider.py")
        return {file.stem.replace("_provider", "") for file in provider_files}


class Audio:
    """Base class for all audio functionality."""

    def __init__(self):
        self.transcriptions: Optional["Audio.Transcription"] = None

    class Transcription(ABC):
        """Base class for audio transcription functionality."""

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options=None,
            **kwargs,
        ):
            """Create audio transcription."""
            raise NotImplementedError("Transcription not supported by this provider")

        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options=None,
            **kwargs,
        ):
            """Create streaming audio transcription."""
            raise NotImplementedError(
                "Streaming transcription not supported by this provider"
            )
