"""Y-API provider for the aisuite."""

import os
import openai
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter

# Default endpoint of the Y-API gateway. Override it with `base_url` in the
# provider config, or with the YAPI_API_BASE environment variable.
YAPI_BASE_URL = "https://api.y-api.bestvirtualgoods.com/v1"


class YapiProvider(Provider):
    """Provider for Y-API.

    Y-API fronts several vendors behind a single OpenAI-compatible endpoint
    and API key, so model ids keep their vendor prefix:
    "yapi:deepseek/deepseek-v4-flash", "yapi:z-ai/glm-5.3".
    """

    def __init__(self, **config):
        """
        Initialize the Y-API provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("YAPI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Y-API key is missing. Please provide it in the config or set the "
                "YAPI_API_KEY environment variable."
            )
        # setdefault (rather than assignment) so an alternative gateway can be
        # pointed at without patching the provider.
        config.setdefault("base_url", os.getenv("YAPI_API_BASE") or YAPI_BASE_URL)

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        # Async client shares the same config for true non-blocking I/O.
        self.aclient = openai.AsyncOpenAI(**config)
        # Using OpenAICompliantMessageConverter since Y-API's response format is
        # the same as OpenAI's.
        self.transformer = OpenAICompliantMessageConverter()

        super().__init__()

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the Y-API
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e

    async def achat_completions_create(self, model, messages, **kwargs):
        # Native async path via openai.AsyncOpenAI.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = await self.aclient.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}") from e

    def chat_completions_create_stream(self, model, messages, **kwargs):
        # Y-API speaks OpenAI's wire format, so its chunks already have the
        # unified shape — pass them through.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            stream = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
        yield from stream

    async def achat_completions_create_stream(self, model, messages, **kwargs):
        # Native async streaming via openai.AsyncOpenAI.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            stream = await self.aclient.chat.completions.create(
                model=model,
                messages=transformed_messages,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
        async for chunk in stream:
            yield chunk
