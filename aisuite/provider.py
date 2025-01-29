from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import os
import functools
import logging
import time
from opentelemetry import trace


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message):
        super().__init__(message)


class Provider(ABC):
    tracer = trace.get_tracer(__name__)

    def log_request(self, method_name, **kwargs):
        """Log details of the API call."""
        logger.info(f"API call to {method_name} with parameters: {kwargs}")

    def log_error(self, method_name, error):
        """Log errors for debugging."""
        logger.error(f"Error in {method_name}: {error}")

    def trace_execution(self, method_name):
        """Wrap a method with tracing."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(method_name):
                    start_time = time.time()
                    try:
                        result = func(*args, **kwargs)
                        latency = time.time() - start_time
                        logger.info(f"{method_name} executed in {latency:.2f}s")
                        return result
                    except Exception as e:
                        self.log_error(method_name, e)
                        raise e

            return wrapper

        return decorator

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
