import openai
import os
from aisuite.provider import Provider, LLMError


class DeepseekProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the DeepSeek provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("DEEPSEEK_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "DeepSeek API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )
        config["base_url"] = "https://api.deepseek.com"

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID. Except for OPEN_AI_BASE_URL which has to be the deepseek url

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )
        return response
