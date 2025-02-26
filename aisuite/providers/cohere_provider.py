import os
import cohere
import json
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function
from aisuite.provider import Provider, LLMError


class CohereMessageConverter:
    """
    Cohere-specific message converter
    """

    def convert_request(self, messages):
        """Convert framework messages to Cohere format."""
        converted_messages = []

        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                tool_plan = message.get("tool_plan")
            else:
                role = message.role
                content = message.content
                tool_calls = message.tool_calls
                tool_plan = getattr(message, "tool_plan", None)

            # Convert to Cohere's format
            if role == "tool":
                # Handle tool response messages
                converted_message = {
                    "role": role,
                    "tool_call_id": (
                        message.get("tool_call_id")
                        if isinstance(message, dict)
                        else message.tool_call_id
                    ),
                    "content": self._convert_tool_content(content),
                }
            elif role == "assistant" and tool_calls:
                # Handle assistant messages with tool calls
                converted_message = {
                    "role": role,
                    "tool_calls": [
                        {
                            "id": tc.id if not isinstance(tc, dict) else tc["id"],
                            "function": {
                                "name": (
                                    tc.function.name
                                    if not isinstance(tc, dict)
                                    else tc["function"]["name"]
                                ),
                                "arguments": (
                                    tc.function.arguments
                                    if not isinstance(tc, dict)
                                    else tc["function"]["arguments"]
                                ),
                            },
                            "type": "function",
                        }
                        for tc in tool_calls
                    ],
                    "tool_plan": tool_plan,
                }
                if content:
                    converted_message["content"] = content
            else:
                # Handle regular messages
                converted_message = {"role": role, "content": content}

            converted_messages.append(converted_message)

        return converted_messages

    def _convert_tool_content(self, content):
        """Convert tool response content to Cohere's expected format."""
        if isinstance(content, str):
            try:
                # Try to parse as JSON first
                data = json.loads(content)
                return [{"type": "document", "document": {"data": json.dumps(data)}}]
            except json.JSONDecodeError:
                # If not JSON, return as plain text
                return content
        elif isinstance(content, list):
            # If content is already in Cohere's format, return as is
            return content
        else:
            # For other types, convert to string
            return str(content)

    @staticmethod
    def convert_response(response_data) -> ChatCompletionResponse:
        """Convert Cohere's response to our standard format."""
        normalized_response = ChatCompletionResponse()

        # Set usage information
        normalized_response.usage = {
            "prompt_tokens": response_data.usage.tokens.input_tokens,
            "completion_tokens": response_data.usage.tokens.output_tokens,
            "total_tokens": response_data.usage.tokens.input_tokens
            + response_data.usage.tokens.output_tokens,
        }

        # Handle tool calls
        if response_data.finish_reason == "TOOL_CALL":
            tool_call = response_data.message.tool_calls[0]
            function = Function(
                name=tool_call.function.name, arguments=tool_call.function.arguments
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call.id, function=function, type="function"
            )
            normalized_response.choices[0].message = Message(
                content=response_data.message.tool_plan,  # Use tool_plan as content
                tool_calls=[tool_call_obj],
                role="assistant",
                refusal=None,
            )
            normalized_response.choices[0].finish_reason = "tool_calls"
        else:
            # Handle regular text response
            normalized_response.choices[0].message.content = (
                response_data.message.content[0].text
            )
            normalized_response.choices[0].finish_reason = "stop"

        return normalized_response


class CohereProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Cohere provider with the given configuration.
        Pass the entire configuration dictionary to the Cohere client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("CO_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "Cohere API key is missing. Please provide it in the config or set the CO_API_KEY environment variable."
            )
        self.client = cohere.ClientV2(**config)
        self.transformer = CohereMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to Cohere using the official client.
        """
        try:
            # Transform messages using converter
            transformed_messages = self.transformer.convert_request(messages)

            # Make the request to Cohere
            response = self.client.chat(
                model=model, messages=transformed_messages, **kwargs
            )

            return self.transformer.convert_response(response)
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
