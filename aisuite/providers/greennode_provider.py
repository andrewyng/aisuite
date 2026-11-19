import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class GreennodeMessageConverter(OpenAICompliantMessageConverter):
    """
    GreenNode-specific message converter if needed
    """

    pass


class GreennodeProvider(Provider):
    def __init__(self, **config) -> None:
        self.api_key = config.get("api_key", os.getenv("GREENNODE_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "GreenNode API key is missing. Please provide it in the config or set the GREENNODE_API_KEY environment variable.",
                "See https://github.com/greennode-ai/greennode-python?tab=readme-ov-file#authentication for more details"
            )
        self.url = config.get("api_url", os.getenv("GREENNODE_BASE_URL"))

        self.transformer = GreennodeMessageConverter()
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the GreenNode chat completions endpoint using the OpenAI client.
        """
        kwargs["stream"] = False
        try:
            # Transform messages using converter
            transformed_messages = self.transformer.convert_request(messages)

            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
