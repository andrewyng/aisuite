"""The Grid provider for the aisuite."""

import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


# pylint: disable=too-few-public-methods
class ThegridProvider(Provider):
    """Provider for The Grid.

    The Grid (https://thegrid.ai) is an OpenAI-compatible inference API whose
    model ids are *market instruments* rather than fixed foundation models. You
    request a quality tier such as ``text-standard``, ``code-prime`` or
    ``agent-max``, and The Grid acquires qualifying inference on its market and
    serves the result. Lab-specific markets are also available, e.g.
    ``claude-opus-latest`` or ``kimi-latest``.

    Because an instrument pools several backing models, the ``model`` field of a
    response names the model that actually served the request rather than the
    instrument that was requested. Per-instrument context limits, capability
    flags and live prices are published at
    ``GET https://api.thegrid.ai/v1/models``.
    """

    def __init__(self, **config):
        """
        Initialize The Grid provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("THEGRID_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "The Grid API key is missing. Please provide it in the config or "
                "set the THEGRID_API_KEY environment variable. You can create a key "
                "at https://thegrid.ai."
            )
        config.setdefault("base_url", "https://api.thegrid.ai/v1")

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        # Using OpenAICompliantMessageConverter since The Grid's response format
        # is the same as OpenAI's.
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to The Grid API
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e
