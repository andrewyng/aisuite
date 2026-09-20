import urllib.request
import json
import os

from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function

# Alibaba Cloud EAS (Elastic Algorithm Service) PAI Provider
# EAS is part of Alibaba Cloud's PAI (Platform for AI) platform.
# Documentation: https://www.alibabacloud.com/help/en/pai/user-guide/eas-model-serving
#
# EAS provides model inference services with OpenAI-compatible API endpoints.
# The endpoint URL format is typically:
#   https://<service_name>.<region>.pai-eas.aliyuncs.com/api/predict/<service_name>
# Or for OpenAI-compatible endpoints:
#   https://<endpoint>/v1/chat/completions
#
# Authentication is done via Token in the Authorization header.


class Eas_paiMessageConverter:
    @staticmethod
    def convert_request(messages):
        """Convert messages to EAS PAI format (OpenAI-compatible)."""
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                transformed_messages.append(message.model_dump(mode="json"))
            else:
                transformed_messages.append(message)
        return transformed_messages

    @staticmethod
    def convert_response(resp_json) -> ChatCompletionResponse:
        """Normalize the response from EAS PAI API to match OpenAI's response format."""
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


class Eas_paiProvider(Provider):
    """
    Alibaba Cloud EAS PAI Provider for aisuite.

    Configuration options:
        - base_url: The EAS endpoint URL (or set EAS_PAI_BASE_URL env var)
        - api_key: The EAS service token (or set EAS_PAI_API_KEY env var)

    Usage:
        client = aisuite.Client()
        client.configure({
            "eas_pai": {
                "base_url": "https://your-service.region.pai-eas.aliyuncs.com/api/predict/your-service",
                "api_key": "your-eas-token"
            }
        })
        response = client.chat.completions.create(
            model="eas_pai:your-model-name",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """

    def __init__(self, **config):
        self.base_url = config.get("base_url") or os.getenv("EAS_PAI_BASE_URL")
        self.api_key = config.get("api_key") or os.getenv("EAS_PAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "For EAS PAI, api_key is required. "
                "Set it via config or EAS_PAI_API_KEY environment variable."
            )
        if not self.base_url:
            raise ValueError(
                "For EAS PAI, base_url is required. "
                "Set it via config or EAS_PAI_BASE_URL environment variable. "
                "Example: https://<service>.<region>.pai-eas.aliyuncs.com/api/predict/<service>"
            )

        self.transformer = Eas_paiMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        # Determine the endpoint URL
        # If base_url already contains /chat/completions, use it directly
        # Otherwise, append /v1/chat/completions for OpenAI-compatible endpoints
        if "/chat/completions" in self.base_url:
            url = self.base_url
        elif self.base_url.endswith("/v1"):
            url = f"{self.base_url}/chat/completions"
        else:
            # For standard EAS endpoints, the URL is used as-is
            # The model inference happens at the base URL
            url = self.base_url

        # Remove 'stream' from kwargs if present (streaming not supported)
        kwargs.pop("stream", None)

        # Transform messages using converter
        transformed_messages = self.transformer.convert_request(messages)

        # Prepare the request payload
        data = {
            "model": model,
            "messages": transformed_messages,
        }

        # Add tools if provided
        if "tools" in kwargs:
            data["tools"] = kwargs.pop("tools")

        # Add tool_choice if provided
        if "tool_choice" in kwargs:
            data["tool_choice"] = kwargs.pop("tool_choice")

        # Add remaining kwargs (temperature, max_tokens, etc.)
        data.update(kwargs)

        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key,
        }

        try:
            req = urllib.request.Request(url, body, headers)
            with urllib.request.urlopen(req) as response:
                result = response.read()
                resp_json = json.loads(result)
                return self.transformer.convert_response(resp_json)

        except urllib.error.HTTPError as error:
            error_message = f"EAS PAI request failed with status code: {error.code}\n"
            error_message += f"Headers: {error.info()}\n"
            error_message += error.read().decode("utf-8", "ignore")
            raise Exception(error_message)
