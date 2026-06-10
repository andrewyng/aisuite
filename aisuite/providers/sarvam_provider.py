import os
from aisuite.provider import Provider, LLMError
from openai import OpenAI
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class SarvamMessageConverter(OpenAICompliantMessageConverter):
    """
    Sarvam-specific message converter.
    """

    pass


class SarvamProvider(Provider):
    """
    Sarvam AI Provider using OpenAI-compatible client.
    Sarvam uses a custom auth header (api-subscription-key) instead of Bearer token,
    so we inject it via default_headers and use a placeholder api_key for the OpenAI SDK.
    """

    def __init__(self, **config):
        """
        Initialize the Sarvam provider with the given configuration.
        """
        # Ensure API key is provided either in config or via environment variable
        self.api_key = config.get("api_key", os.getenv("SARVAM_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Sarvam API key is missing. Please provide it in the config or set the SARVAM_API_KEY environment variable."
            )

        self.client = OpenAI(
            api_key="placeholder",  # Required by OpenAI SDK, not used by Sarvam
            base_url="https://api.sarvam.ai/v1",
            default_headers={"api-subscription-key": self.api_key},
        )
        self.transformer = SarvamMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Sarvam chat completions endpoint using the OpenAI client.
        """
        try:
            transformed_messages = self.transformer.convert_request(messages)

            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
