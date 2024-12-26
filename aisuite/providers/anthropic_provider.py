from typing import Dict, Any, List, Callable
import anthropic
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.function_call import generate_function_calling_schema

from loguru import logger

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Anthropic provider with the given configuration.
        Pass the entire configuration dictionary to the Anthropic client constructor.
        """
        self.client = anthropic.Anthropic(**config)

    def chat_completions_create(self, model, messages, tools=None, **kwargs):
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
        if tools:
            logger.info("Generating function calling schema for tools")
            tools_with_schema = [
                generate_function_calling_schema(tool) for tool in tools
            ]

        # Initial API call
        logger.info("Initial API call")
        response = self.client.messages.create(
            model=model,
            messages=messages,
            tools=tools_with_schema,
            **kwargs,
        )

        # Check if there are any tool calls
        if not any(content.type == "tool_use" for content in response.content):
            logger.info("No tool calls found")
            return self.normalize_response(response)

        self._process_tool_calls(
            response, messages, tools, model, system_message, **kwargs
        )

    def _process_tool_calls(
        self, response, messages, tools, model, system_message, **kwargs
    ):
        logger.info("Processing tool calls")
        messages.append({"role": "assistant", "content": response.content})
        for content in response.content:
            if content.type == "tool_use":
                logger.info(f"Tool use: {content}")
                tool_name = content.name
                tool_input = content.input
                tool_id = content.id

                # Execute the tool
                if tools:
                    try:
                        logger.info("Executing tool")
                        tool_result = self.execute_tool(
                            tool_name, tool_input, tools=tools
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": str(tool_result),
                                    }
                                ],
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool: {str(e)}")
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_id,
                                        "content": f"Error: {str(e)}",
                                    }
                                ],
                            }
                        )

        # Get final response after tool execution
        logger.info("Getting final response after tool execution")
        # TODO: recursive call to handle nested tool calls
        final_response = self.client.messages.create(
            model=model,
            system=system_message,
            messages=messages,
            **kwargs,
        )

        return self.normalize_response(final_response)

    def normalize_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
        logger.info("Normalizing response")
        normalized_response.choices[0].message.content = response.content[0].text
        return normalized_response

    def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tools: List[Callable],
    ):
        """Execute a tool call"""
        matching_tool = None
        for tool in tools:
            if tool.__name__ == tool_name:
                matching_tool = tool
                break

        if matching_tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in provided tools")

        try:
            return matching_tool(**tool_input)
        except Exception as e:
            raise Exception(f"Error executing tool: {str(e)}")
