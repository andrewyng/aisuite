import os

from openai import OpenAI

from aisuite.provider import LLMError, Provider
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class OpenvinoMessageConverter(OpenAICompliantMessageConverter):
    """
    Openvino-specific message converter.
    """

    pass


class OpenvinoProvider(Provider):
    """
    OpenVINO Provider that makes chat completions requests using the OpenAI client.
    This provider can be used with OpenVINO Model Server.
    """

    def __init__(self, **config):
        """
        Initialize the OpenVINO provider with the given configuration.
        """
        self.url = config.get("api_url") or os.getenv(
            "OVMS_API_URL", "http://localhost:8000/v3"
        )
        config["base_url"] = self.url
        config["api_key"] = "unused"
        self.client = OpenAI(**config)
        self.transformer = OpenvinoMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the OpenVINO Model Server chat completions endpoint using the OpenAI client.
        """
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the OpenVINO Chat API
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
