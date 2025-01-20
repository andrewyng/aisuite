import os

from mistralai import Mistral
from aisuite.framework.message import Message
from aisuite.framework import ChatCompletionResponse
from aisuite.provider import Provider, LLMError


# Implementation of Mistral provider.
# Mistral's message format is same as OpenAI's. Just different class names, but fully cross-compatible.
# Links:
# https://docs.mistral.ai/capabilities/function_calling/


class MistralMessageConverter:
    @staticmethod
    def convert_request(messages):
        """Convert messages to Mistral format."""
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                message_dict = message.model_dump(mode="json")
                message_dict.pop("refusal", None)  # Remove refusal field if present
                transformed_messages.append(message_dict)
            else:
                transformed_messages.append(message)
        return transformed_messages

    @staticmethod
    def convert_response(response) -> ChatCompletionResponse:
        """Normalize the response from Mistral to match OpenAI's response format."""
        completion_response = ChatCompletionResponse()
        choice = response.choices[0]
        message = choice.message

        # Set basic message content
        completion_response.choices[0].message.content = message.content
        completion_response.choices[0].message.role = message.role

        # Handle tool calls if present
        if hasattr(message, "tool_calls") and message.tool_calls:
            completion_response.choices[0].message.tool_calls = message.tool_calls

        return completion_response


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
    def __init__(self, **config):
        """
        Initialize the Mistral provider with the given configuration.
        Pass the entire configuration dictionary to the Mistral client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MISTRAL_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                " API key is missing. Please provide it in the config or set the MISTRAL_API_KEY environment variable."
            )
        self.client = Mistral(**config)
        self.transformer = MistralMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
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
