import openai
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
                "OpenAI API key is missing. Please provide it in the config "
                "or set the OPENAI_API_KEY environment variable."
            )

        # Pass the entire config to the OpenAI client constructor
        # (Note: This assumes openai.OpenAI(...) is valid in your environment.
        #  If you typically do `openai.api_key = ...`, adapt as needed.)
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Create chat completion using the OpenAI API.
        If 'stream=True' is passed via kwargs, return a generator that yields
        chunked responses in the OpenAI streaming format.
        """
        stream = kwargs.pop("stream", False)

        if not stream:
            # Non-streaming call
            return self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
        else:
            # Streaming call: return a generator that yields each chunk
            return self._streaming_chat_completions_create(model, messages, **kwargs)

    def _streaming_chat_completions_create(self, model, messages, **kwargs):
        """
        Internal helper method that yields chunked responses for streaming.
        Each chunk is already in the OpenAI streaming format:

        {
          "id": ...,
          "object": "chat.completion.chunk",
          "created": ...,
          "model": ...,
          "choices": [
            {
              "delta": {
                "role": "assistant" or "content": ...
              }
            }
          ]
        }
        """
        response_gen = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )

        # Yield chunks as they arrive
        for chunk in response_gen:
            yield chunk



