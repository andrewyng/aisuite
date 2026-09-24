"""Cheaper Inference provider for the aisuite."""

import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


# pylint: disable=too-few-public-methods
class CheaperinferenceProvider(Provider):
    """Provider for Cheaper Inference.

    Cheaper Inference (https://cheaperinference.com) is an OpenAI-compatible
    gateway that gives access to models from many labs through a single API key.
    Each model costs 15–60% less than the list price of its lab.
    Model ids are bare, e.g. ``gpt-5.4-mini``, ``claude-sonnet-5`` or
    ``gemini-3.1-pro``.
    """

    def __init__(self, **config):
        """
        Initialize the Cheaper Inference provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("CHEAPER_INFERENCE_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Cheaper Inference API key is missing. Please provide it in the "
                "config or set the CHEAPER_INFERENCE_API_KEY environment variable."
            )
        config["base_url"] = "https://api.cheaperinference.com/v1"

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        # Using OpenAICompliantMessageConverter since Cheaper Inference's response
        # format is the same as OpenAI's.
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to the OpenAI API
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
