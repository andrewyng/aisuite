import urllib.request
import json
import os

from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse


class AzureopenaiProvider(Provider):
    def __init__(self, **config):
        self.base_url = config.get("base_url") or os.getenv("AZURE_OPENAI_BASE_URL")
        self.api_key = config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = config.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
        if not self.api_key:
            raise ValueError("For Azure, api_key is required.")
        if not self.base_url:
            raise ValueError(
                "For Azure, base_url is required. Check your deployment page for a URL like this - https://<base_url>.openai.azure.com"
            )

    def chat_completions_create(self, model, messages, **kwargs):
        url = f"{self.base_url}/openai/deployments/{model}/chat/completions?api-version={self.api_version}"

        # Remove 'stream' from kwargs if present
        kwargs.pop("stream", None)
        data = {"messages": messages, **kwargs}

        body = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json", "api-key": self.api_key}

        try:
            req = urllib.request.Request(url, body, headers)
            with urllib.request.urlopen(req) as response:
                result = response.read()
                resp_json = json.loads(result)
                completion_response = ChatCompletionResponse()
                # TODO: Add checks for fields being present in resp_json.
                completion_response.choices[0].message.content = resp_json["choices"][
                    0
                ]["message"]["content"]
                return completion_response

        except urllib.error.HTTPError as error:
            error_message = f"The request failed with status code: {error.code}\n"
            error_message += f"Headers: {error.info()}\n"
            error_message += error.read().decode("utf-8", "ignore")
            raise Exception(error_message)
