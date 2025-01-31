import openai
from pydantic import BaseModel
from typing import Type
import os
from aisuite.provider import Provider, LLMError


class OpenaiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_BASE_URL, etc.

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)

    def chat_completions_create(
        self, model, messages, response_format: Type[BaseModel] | None = None, **kwargs
    ):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        if response_format is not None:
            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                **kwargs  # Pass any additional arguments to the OpenAI API
            )
            response.choices[0].message.content = response.choices[0].message.parsed
            return response

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )

        return response
