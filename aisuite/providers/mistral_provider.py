import os
from mistralai import Mistral
from aisuite.framework.message import Message
from aisuite.framework import ChatCompletionResponse
from aisuite.provider import Provider, AsyncProvider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


# Implementation of Mistral provider.
# Mistral's message format is same as OpenAI's. Just different class names, but fully cross-compatible.
# Links:
# https://docs.mistral.ai/capabilities/function_calling/


class MistralMessageConverter(OpenAICompliantMessageConverter):
    """
    Mistral-specific message converter
    """

    @staticmethod
    def convert_response(response_data) -> ChatCompletionResponse:
        """Convert Mistral's response to our standard format."""
        # Convert Mistral's response object to dict format
        response_dict = response_data.model_dump()
        return super(MistralMessageConverter, MistralMessageConverter).convert_response(
            response_dict
        )


# Function calling is available for the following models:
# [As of 01/19/2025 from https://docs.mistral.ai/capabilities/function_calling/]
# Mistral Large
# Mistral Small
# Codestral 22B
# Ministral 8B
# Ministral 3B
# Pixtral 12B
# Mixtral 8x22B
# Mistral Nemo
class MistralProvider(Provider):
    """
    Mistral AI Provider using the official Mistral client.
    """

    def __init__(self, **config):
        """
        Initialize the Mistral provider with the given configuration.
        Pass the entire configuration dictionary to the Mistral client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Mistral API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            )
        self.client = Mistral(**config)
        self.transformer = MistralMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to Mistral using the official client.
        """
        try:
            # Transform messages using converter
            transformed_messages = self.transformer.convert_request(messages)

            # Make the request to Mistral
            response = self.client.chat.complete(
                model=model, messages=transformed_messages, **kwargs
            )

            return self.transformer.convert_response(response)
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

class MistralAsyncProvider(AsyncProvider):
    """
    Mistral AI Provider using the official Mistral client.
    """

    def __init__(self, **config):
        """
        Initialize the Mistral provider with the given configuration.
        Pass the entire configuration dictionary to the Mistral client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Mistral API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            )
        self.client = Mistral(**config)
        self.transformer = MistralMessageConverter()

    async def chat_completions_create_async(self, model, messages, **kwargs):
        """
        Makes a request to Mistral using the official client.
        """
        try:
            # Transform messages using converter
            transformed_messages = self.transformer.convert_request(messages)

            response = await self.client.chat.complete_async(
                model=model, messages=transformed_messages, **kwargs
            )

            return self.transformer.convert_response(response)
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
