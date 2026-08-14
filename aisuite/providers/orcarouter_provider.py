import openai
import os
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class OrcarouterProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OrcaRouter provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("ORCAROUTER_API_KEY"))
        config.setdefault("base_url", "https://api.orcarouter.ai/v1")

        # Support optional OrcaRouter attribution headers
        default_headers = config.get("default_headers", {})
        if os.getenv("ORCAROUTER_SITE_URL") and "HTTP-Referer" not in default_headers:
            default_headers["HTTP-Referer"] = os.getenv("ORCAROUTER_SITE_URL")
        if os.getenv("ORCAROUTER_APP_NAME") and "X-Title" not in default_headers:
            default_headers["X-Title"] = os.getenv("ORCAROUTER_APP_NAME")

        if default_headers:
            config["default_headers"] = default_headers

        if not config["api_key"]:
            raise ValueError(
                "OrcaRouter API key is missing. Please provide it in the config or set the ORCAROUTER_API_KEY environment variable."
            )

        self.client = openai.OpenAI(**config)
        # Async client shares the same config for true non-blocking I/O.
        self.aclient = openai.AsyncOpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

        super().__init__()

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OrcaRouter will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the OrcaRouter API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    async def achat_completions_create(self, model, messages, **kwargs):
        # Native async path via openai.AsyncOpenAI.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = await self.aclient.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the OrcaRouter API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    def chat_completions_create_stream(self, model, messages, **kwargs):
        # OrcaRouter streams OpenAI-shaped chat.completion.chunk objects — pass them through.
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
