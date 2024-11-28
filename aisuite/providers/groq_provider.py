import os

import groq
from aisuite.provider import Provider
from aisuite.framework.message import Message

# Implementation of Groq provider.
# Groq's message format is same as OpenAI's.
# Tool calling specification is also exactly the same as OpenAI's.
# Links:
# https://console.groq.com/docs/tool-use
# Groq supports tool calling for the following models, as of 16th Nov 2024:
#   llama3-groq-70b-8192-tool-use-preview
#   llama3-groq-8b-8192-tool-use-preview
#   llama-3.1-70b-versatile
#   llama-3.1-8b-instant
#   llama3-70b-8192
#   llama3-8b-8192
#   mixtral-8x7b-32768 (parallel tool use not supported)
#   gemma-7b-it (parallel tool use not supported)
#   gemma2-9b-it (parallel tool use not supported)


class GroqProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Groq provider with the given configuration.
        Pass the entire configuration dictionary to the Groq client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("GROQ_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                " API key is missing. Please provide it in the config or set the GROQ_API_KEY environment variable."
            )
        self.client = groq.Groq(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                transformed_messages.append(self.transform_from_messages(message))
            else:
                transformed_messages.append(message)
        return self.client.chat.completions.create(
            model=model,
            messages=transformed_messages,
            **kwargs  # Pass any additional arguments to the Groq API
        )

    # Transform framework Message to a format that Groq understands.
    def transform_from_messages(self, message: Message):
        return message.model_dump(mode="json")

    # Transform Groq message (dict) to a format that the framework Message understands.
    def transform_to_message(self, message_dict: dict):
        return Message(**message_dict)
