"""Eden AI provider for the aisuite."""

import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


# pylint: disable=too-few-public-methods
class EdenaiProvider(Provider):
    """Provider for Eden AI.

    Eden AI (https://www.edenai.co) is an EU-hosted, OpenAI-compatible gateway
    that exposes 100+ models from many providers through a single API key.
    Models use the ``provider/model`` naming scheme, e.g.
    ``anthropic/claude-sonnet-4-5`` or ``mistral/codestral-latest``.
    """

    def __init__(self, **config):
        """
        Initialize the Eden AI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("EDENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Eden AI API key is missing. Please provide it in the config or "
                "set the EDENAI_API_KEY environment variable."
            )
        config["base_url"] = "https://api.edenai.run/v3"

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        # Using OpenAICompliantMessageConverter since Eden AI's response format is
        # the same as OpenAI's.
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
