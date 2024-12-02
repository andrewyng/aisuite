import os
import dashscope
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse


class TongyiProvider(Provider):
    """TongyiProvider is a class that provides an interface to the Tongyi's model."""

    def __init__(self, **config):
        self.api_key = config.get("api_key") or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise EnvironmentError(
                "Dashscope API key is missing. Please provide it in the config or set the DASHSCOPE_API_KEY environment variable."
            )

    def chat_completions_create(self, model, messages, **kwargs):
        """Send a chat completion request to the Tongyi's model."""

        response = dashscope.Generation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            result_format="message",
            **kwargs
        )
        return self.normalize_response(response)

    def normalize_response(self, response):
        """Normalize the response from Dashscope to match OpenAI's response format."""

        openai_response = ChatCompletionResponse()
        openai_response.choices[0].message.content = response["output"]["choices"][0][
            "message"
        ].get("content")
        return openai_response
