import urllib.request
import json
import os

from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function

# Azure provider is based on the documentation here -
# https://learn.microsoft.com/en-us/azure/machine-learning/reference-model-inference-api?view=azureml-api-2&source=recommendations&tabs=python
# Azure AI Model Inference API is used.
# From the documentation -
# """
# The Azure AI Model Inference is an API that exposes a common set of capabilities for foundational models
# and that can be used by developers to consume predictions from a diverse set of models in a uniform and consistent way.
# Developers can talk with different models deployed in Azure AI Foundry portal without changing the underlying code they are using.
#
# The Azure AI Model Inference API is available in the following models:
#
# Models deployed to serverless API endpoints:
#   Cohere Embed V3 family of models
#   Cohere Command R family of models
#   Meta Llama 2 chat family of models
#   Meta Llama 3 instruct family of models
#   Mistral-Small
#   Mistral-Large
#   Jais family of models
#   Jamba family of models
#   Phi-3 family of models
#
# Models deployed to managed inference:
#   Meta Llama 3 instruct family of models
#   Phi-3 family of models
#   Mixtral famility of models
#
# The API is compatible with Azure OpenAI model deployments.
# """


class AzureMessageConverter:
    @staticmethod
    def convert_request(messages):
        """Convert messages to Azure format."""
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                transformed_messages.append(message.model_dump(mode="json"))
            else:
                transformed_messages.append(message)
        return transformed_messages

    @staticmethod
    def convert_response(resp_json) -> ChatCompletionResponse:
        """Normalize the response from the Azure API to match OpenAI's response format."""
        completion_response = ChatCompletionResponse()
        choice = resp_json["choices"][0]
        message = choice["message"]

        # Set basic message content
        completion_response.choices[0].message.content = message.get("content")
        completion_response.choices[0].message.role = message.get("role", "assistant")

        # Handle tool calls if present
        if "tool_calls" in message and message["tool_calls"] is not None:
            tool_calls = []
            for tool_call in message["tool_calls"]:
                new_tool_call = ChatCompletionMessageToolCall(
                    id=tool_call["id"],
                    type=tool_call["type"],
                    function={
                        "name": tool_call["function"]["name"],
                        "arguments": tool_call["function"]["arguments"],
                    },
                )
                tool_calls.append(new_tool_call)
            completion_response.choices[0].message.tool_calls = tool_calls

        return completion_response


class AzureProvider(Provider):
    def __init__(self, **config):
        self.base_url = config.get("base_url") or os.getenv("AZURE_BASE_URL")
        self.api_key = config.get("api_key") or os.getenv("AZURE_API_KEY")
        self.api_version = config.get("api_version") or os.getenv("AZURE_API_VERSION")
        if not self.api_key:
            raise ValueError("For Azure, api_key is required.")
        if not self.base_url:
            raise ValueError(
                "For Azure, base_url is required. Check your deployment page for a URL like this - https://<model-deployment-name>.<region>.models.ai.azure.com"
            )
        self.transformer = AzureMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        url = f"{self.base_url}/chat/completions"

        if self.api_version:
            url = f"{url}?api-version={self.api_version}"

        # Remove 'stream' from kwargs if present
        kwargs.pop("stream", None)

        # Transform messages using converter
        transformed_messages = self.transformer.convert_request(messages)

        # Prepare the request payload
        data = {"messages": transformed_messages}

        # Add tools if provided
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
            kwargs.pop("tools")

        # Add tool_choice if provided
        if "tool_choice" in kwargs:
            data["tool_choice"] = kwargs["tool_choice"]
            kwargs.pop("tool_choice")

        # Add remaining kwargs
        data.update(kwargs)

        body = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": self.api_key}

        try:
            req = urllib.request.Request(url, body, headers)
            with urllib.request.urlopen(req) as response:
                result = response.read()
                resp_json = json.loads(result)
                return self.transformer.convert_response(resp_json)

        except urllib.error.HTTPError as error:
            error_message = f"The request failed with status code: {error.code}\n"
            error_message += f"Headers: {error.info()}\n"
            error_message += error.read().decode("utf-8", "ignore")
            raise Exception(error_message)
