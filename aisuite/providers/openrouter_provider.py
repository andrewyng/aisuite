import os
from aisuite.provider import Provider
from openai import OpenAI


class OpenrouterProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenRouter provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENROUTER_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenRouter API key is missing. Please provide it in the config or set the OPENROUTER_API_KEY environment variable."
            )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_BASE_URL, etc.

        # Pass the entire config to the OpenAI client constructor
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config["api_key"],
        )

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )