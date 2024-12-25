from typing import Dict, Any, List
import json
import anthropic
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.function_call import FunctionCall

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Anthropic provider with the given configuration.
        Pass the entire configuration dictionary to the Anthropic client constructor.
        """

        self.client = anthropic.Anthropic(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        """Create a chat completion with function calling support."""
        # Check if the first message is a system message
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = []

        # Set default max tokens if not provided
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS

        # Initial API call
        response = self.client.messages.create(
            model=model, system=system_message, messages=messages, **kwargs
        )

        # Check if there are any tool calls
        if not any(content.type == "tool_use" for content in response.content):
            return self.normalize_response(response)

        # Handle tool calls
        messages_copy = messages.copy()
        messages_copy.append(
            {
                "role": "assistant",
                "content": (
                    [{"type": "text", "text": response.content[0].text}]
                    if response.content[0].text
                    else None
                ),
            }
        )

        # Process each tool call
        for content in response.content:
            if content.type == "tool_use":
                tool_name = content.name
                tool_input = content.input
                tool_id = content.id

                # Execute the tool
                if "tools" in kwargs:
                    try:
                        # Find and execute the matching tool
                        tool_result = self.execute_tool(tool_name, tool_input, **kwargs)
                        # Add tool result to messages
                        messages_copy.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_response",
                                        "tool_call_id": tool_id,
                                        "name": tool_name,
                                        "content": str(tool_result),
                                    }
                                ],
                            }
                        )
                    except Exception as e:
                        messages_copy.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_response",
                                        "tool_call_id": tool_id,
                                        "name": tool_name,
                                        "content": f"Error: {str(e)}",
                                    }
                                ],
                            }
                        )

        # Get final response after tool execution
        final_response = self.client.messages.create(
            model=model, system=system_message, messages=messages_copy, **kwargs
        )

        return self.normalize_response(final_response)

    def normalize_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
        print("Full response:")
        print(response.content)
        print("======================================")

        # Set the text content if available
        text_content = next(
            (content.text for content in response.content if content.type == "text"),
            None,
        )
        if text_content:
            normalized_response.choices[0].message.content = text_content

        # Parse function calls if any
        function_calls = self._parse_function_call(response)
        if function_calls:
            normalized_response.function_calls = function_calls

        return normalized_response

    def _parse_function_call(self, response):
        function_calls = []
        for content in response.content:
            if content.type == "tool_use":
                function_calls.append(
                    FunctionCall(name=content.name, arguments=content.input)
                )
        return function_calls

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any], **kwargs):
        """Execute a tool call"""
        # Validate that tools are provided in kwargs
        if "tools" not in kwargs:
            raise ValueError("tools must be provided in kwargs")

        tools = kwargs["tools"]
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")

        # Find the matching tool by name
        matching_tool = None
        for tool in tools:
            if tool.get("name") == tool_name:
                matching_tool = tool
                break

        if not matching_tool:
            raise ValueError(f"Tool '{tool_name}' not found in provided tools")

        # Get the actual function to execute
        tool_function = matching_tool.get("function")
        if not callable(tool_function):
            raise ValueError(f"Tool '{tool_name}' does not have a callable function")

        # Execute the function with the provided input
        try:
            return tool_function(**tool_input)
        except Exception as e:
            raise Exception(f"Error executing tool: {str(e)}")
