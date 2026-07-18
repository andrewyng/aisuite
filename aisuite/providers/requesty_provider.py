import openai
import os
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class RequestyProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Requesty provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("REQUESTY_API_KEY"))
        config.setdefault("base_url", "https://router.requesty.ai/v1")

        # Support optional Requesty attribution headers
        default_headers = config.get("default_headers", {})
        if os.getenv("REQUESTY_SITE_URL") and "HTTP-Referer" not in default_headers:
            default_headers["HTTP-Referer"] = os.getenv("REQUESTY_SITE_URL")
        if os.getenv("REQUESTY_APP_NAME") and "X-Title" not in default_headers:
            default_headers["X-Title"] = os.getenv("REQUESTY_APP_NAME")

        if default_headers:
            config["default_headers"] = default_headers

        if not config["api_key"]:
            raise ValueError(
                "Requesty API key is missing. Please provide it in the config or set the REQUESTY_API_KEY environment variable."
            )

        self.client = openai.OpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

        super().__init__()

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by Requesty will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the Requesty API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
