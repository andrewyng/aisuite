import openai
import os
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class OpperProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Opper provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPPER_API_KEY"))
        config.setdefault("base_url", "https://api.opper.ai/v3/compat")

        if not config["api_key"]:
            raise ValueError(
                "Opper API key is missing. Please provide it in the config or set the OPPER_API_KEY environment variable."
            )

        self.client = openai.OpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

        super().__init__()

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by Opper will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the Opper API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
