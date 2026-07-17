import os
from aisuite.provider import Provider, LLMError
from openai import OpenAI
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class SambanovaMessageConverter(OpenAICompliantMessageConverter):
    """
    SambaNova-specific message converter.
    """

    pass


class SambanovaProvider(Provider):
    """
    SambaNova Provider using OpenAI client for API calls.
    """

    def __init__(self, **config):
        """
        Initialize the SambaNova provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        self.api_key = config.get("api_key", os.getenv("SAMBANOVA_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Sambanova API key is missing. Please provide it in the config or set the SAMBANOVA_API_KEY environment variable."
            )

        config["api_key"] = self.api_key
        config["base_url"] = "https://api.sambanova.ai/v1/"
        # Pass the entire config to the OpenAI client constructor
        self.client = OpenAI(**config)
        self.transformer = SambanovaMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the SambaNova chat completions endpoint using the OpenAI client.
        """
        try:
            # Transform messages using converter
            transformed_messages = self.transformer.convert_request(messages)

            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the Sambanova API
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
