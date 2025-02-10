import os
import httpx
import json
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall


class FireworksMessageConverter:
    @staticmethod
    def convert_request(messages):
        """Convert messages to Fireworks format."""
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
    def convert_response(resp_json) -> ChatCompletionResponse:
        """Normalize the response from the Fireworks API to match OpenAI's response format."""
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


# Models that support tool calls:
# [As of 01/20/2025 from https://docs.fireworks.ai/guides/function-calling]
# Llama 3.1 405B Instruct
# Llama 3.1 70B Instruct
# Qwen 2.5 72B Instruct
# Mixtral MoE 8x22B Instruct
# Firefunction-v2: Latest and most performant model, optimized for complex function calling scenarios (on-demand only)
# Firefunction-v1: Previous generation, Mixtral-based function calling model optimized for fast routing and structured output (on-demand only)
class FireworksProvider(Provider):
    """
    Fireworks AI Provider using httpx for direct API calls.
    """

    BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

    def __init__(self, **config):
        """
        Initialize the Fireworks provider with the given configuration.
        The API key is fetched from the config or environment variables.
        """
        self.api_key = config.get("api_key", os.getenv("FIREWORKS_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Fireworks API key is missing. Please provide it in the config or set the FIREWORKS_API_KEY environment variable."
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)
        self.transformer = FireworksMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Fireworks AI chat completions endpoint using httpx.
        """
        # Remove 'stream' from kwargs if present
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
            data["tools"] = kwargs["tools"]
            kwargs.pop("tools")

        # Add tool_choice if provided
        if "tool_choice" in kwargs:
            data["tool_choice"] = kwargs["tool_choice"]
            kwargs.pop("tool_choice")

        # Add remaining kwargs
        data.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            # Make the request to Fireworks AI endpoint.
            response = httpx.post(
                self.BASE_URL, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return self.transformer.convert_response(response.json())
        except httpx.HTTPStatusError as error:
            error_message = (
                f"The request failed with status code: {error.status_code}\n"
            )
            error_message += f"Headers: {error.headers}\n"
            error_message += error.response.text
            raise LLMError(error_message)
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    async def chat_completions_create_async(self, model, messages, **kwargs):
        """
        Makes an async request to the Fireworks AI chat completions endpoint.
        """
        # Remove 'stream' from kwargs if present
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
            data["tools"] = kwargs["tools"]
            kwargs.pop("tools")

        # Add tool_choice if provided
        if "tool_choice" in kwargs:
            data["tool_choice"] = kwargs["tool_choice"]
            kwargs.pop("tool_choice")

        # Add remaining kwargs
        data.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                # Make the async request to Fireworks AI endpoint
                response = await client.post(
                    self.BASE_URL, json=data, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()
                return self.transformer.convert_response(response.json())
            except httpx.HTTPStatusError as error:
                error_message = (
                    f"The request failed with status code: {error.status_code}\n"
                )
                error_message += f"Headers: {error.headers}\n"
                error_message += error.response.text
                raise LLMError(error_message)
            except Exception as e:
                raise LLMError(f"An error occurred: {e}")

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["choices"][0][
            "message"
        ]["content"]
        return normalized_response
