import os
import cohere

from aisuite.framework import ChatCompletionResponse
from aisuite.provider import Provider


class CohereProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Cohere provider with the given configuration.
        Pass the entire configuration dictionary to the Cohere client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("CO_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                " API key is missing. Please provide it in the config or set the CO_API_KEY environment variable."
            )
        self.client = cohere.ClientV2(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        response = self.client.chat(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the Cohere API
        )

        return self.normalize_response(response)

    def normalize_response(self, response):
        """Normalize the reponse from Cohere API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response.message.content[
            0
        ].text
        return normalized_response
