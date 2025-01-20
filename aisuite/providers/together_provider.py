import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall


class TogetherMessageConverter:
    @staticmethod
    def convert_request(messages):
        """Convert messages to Together format."""
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                message_dict = message.model_dump(mode="json")
                message_dict.pop("refusal", None)  # Remove refusal field if present
                transformed_messages.append(message_dict)
            else:
                transformed_messages.append(message)
        return transformed_messages

    @staticmethod
    def convert_response(response_data) -> ChatCompletionResponse:
        """Normalize the response from Together to match OpenAI's response format."""
        completion_response = ChatCompletionResponse()
        choice = response_data["choices"][0]
        message = choice["message"]

        # Set basic message content
        completion_response.choices[0].message.content = message["content"]
        completion_response.choices[0].message.role = message.get("role", "assistant")

        # Handle tool calls if present
        if "tool_calls" in message and message["tool_calls"] is not None:
            tool_calls = []
            for tool_call in message["tool_calls"]:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool_call.get("id"),
                        type=tool_call.get("type"),
                        function=tool_call.get("function"),
                    )
                )
            completion_response.choices[0].message.tool_calls = tool_calls

        return completion_response


class TogetherProvider(Provider):
    """
    Together AI Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.together.xyz/v1/chat/completions"

    def __init__(self, **config):
        """
        Initialize the Together provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("TOGETHER_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Together API key is missing. Please provide it in the config or set the TOGETHER_API_KEY environment variable."
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)
        self.transformer = TogetherMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Together AI chat completions endpoint using httpx.
        """
        # Transform messages using converter
        transformed_messages = self.transformer.convert_request(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": transformed_messages,
            **kwargs,  # Pass any additional arguments to the API
        }

        try:
            # Make the request to Together AI endpoint.
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return self.transformer.convert_response(response.json())
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Together AI request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
