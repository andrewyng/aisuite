import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter

# Implementation of Pinstripes provider.
# Pinstripes exposes an OpenAI-compatible API at https://pinstripes.io/v1
# for cheap, fast MoE (Mixture-of-Experts) inference.
# Available models include:
#   ps/deepseek-v4-flash
#   ps/qwen3.6-35b-a3b
#   ps/qwen3-30b-a3b
#   ps/glm-4.5-air
#   ps/minimax-m2.7


BASE_URL = "https://pinstripes.io/v1"


class PinestripesMessageConverter(OpenAICompliantMessageConverter):
    """
    Pinstripes-specific message converter. Message format is the same as OpenAI's.
    """

    pass


class PinestripesProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Pinstripes provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("PINSTRIPES_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Pinstripes API key is missing. Please provide it in the config or set the "
                "PINSTRIPES_API_KEY environment variable."
            )

        config.setdefault("base_url", BASE_URL)

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        self.transformer = PinestripesMessageConverter()

        super().__init__()

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the Pinstripes API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
