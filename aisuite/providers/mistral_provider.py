import os

from mistralai import Mistral
from aisuite.framework.message import Message

from aisuite.provider import Provider


# Implementation of Mistral provider.
# Mistral's message format is same as OpenAI's. Just different class names, but fully cross-compatible.
# Links:
# https://docs.mistral.ai/capabilities/function_calling/
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

    def chat_completions_create(self, model, messages, **kwargs):
        # If message is of type Message, transform it to a format that Mistral understands.
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                transformed_messages.append(self.transform_from_messages(message))
            else:
                transformed_messages.append(message)

        return self.client.chat.complete(
            model=model, messages=transformed_messages, **kwargs
        )

    # Transform framework Message to a format that Mistral understands.
    def transform_from_messages(self, message: Message):
        return message.model_dump(mode="json")

    # Transform Mistral message (dict) to a format that the framework Message understands.
    def transform_to_message(self, message_dict: dict):
        return Message(**message_dict)
