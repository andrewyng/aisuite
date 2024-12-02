import os
import qianfan
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse


class BaiduProvider(Provider):
    """BaiduProvider is a class that provides an interface to the Baidu's model."""

    def __init__(self, **config):
        os.environ["QIANFAN_ACCESS_KEY"] = config.get("access_key") or os.getenv(
            "QIANFAN_ACCESS_KEY"
        )
        os.environ["QIANFAN_SECRET_KEY"] = config.get("secret_key") or os.getenv(
            "QIANFAN_SECRET_KEY"
        )

        if not os.getenv("QIANFAN_ACCESS_KEY"):
            raise EnvironmentError(
                "Qanfan access key is missing. Please provide it in the config or set the QIANFAN_ACCESS_KEY environment variable."
            )
        if not os.getenv("QIANFAN_SECRET_KEY"):
            raise EnvironmentError(
                "Qanfan secret key is missing. Please provide it in the config or set the QIANFAN_SECRET_KEY environment variable."
            )

        self.client = qianfan.ChatCompletion()

    def chat_completions_create(self, model, messages, **kwargs):
        """Send a chat completion request to the Baidu's model."""

        response = self.client.do(model=model, messages=messages, **kwargs)
        return self.normalize_response(response)

    def normalize_response(self, response):
        """Normalize the response from Qianfan to match OpenAI's response format."""

        openai_response = ChatCompletionResponse()
        openai_response.choices[0].message.content = response["body"]["result"]
        return openai_response
