import urllib.request
import json
import os

from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function


class AzureProvider(Provider):
    def __init__(self, **config):
        self.base_url = config.get("base_url") or os.getenv("AZURE_BASE_URL")
        self.api_key = config.get("api_key") or os.getenv("AZURE_API_KEY")
        if not self.api_key:
            raise ValueError("For Azure, api_key is required.")
        if not self.base_url:
            raise ValueError(
                "For Azure, base_url is required. Check your deployment page for a URL like this - https://<model-deployment-name>.<region>.models.ai.azure.com"
            )

    def chat_completions_create(self, model, messages, **kwargs):
        url = f"{self.base_url}/chat/completions"

        # Remove 'stream' from kwargs if present
        kwargs.pop("stream", None)

        # Transform messages if they are Message objects
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                transformed_messages.append(message.model_dump(mode="json"))
            else:
                transformed_messages.append(message)

        # Prepare the request payload with transformed messages
        data = {"messages": transformed_messages}

        # Add tools if provided
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
            # Remove from kwargs to avoid duplication
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
                completion_response = ChatCompletionResponse()

                # Process the response
                choice = resp_json["choices"][0]
                message = choice["message"]

                # Set basic message content
                completion_response.choices[0].message.content = message.get("content")
                completion_response.choices[0].message.role = message.get(
                    "role", "assistant"
                )

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

        except urllib.error.HTTPError as error:
            error_message = f"The request failed with status code: {error.code}\n"
            error_message += f"Headers: {error.info()}\n"
            error_message += error.read().decode("utf-8", "ignore")
            raise Exception(error_message)
