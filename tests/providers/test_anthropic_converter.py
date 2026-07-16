"""Tests for the AnthropicMessageConverter."""

import unittest
from unittest.mock import MagicMock
from aisuite.providers.anthropic_provider import AnthropicMessageConverter
from aisuite.framework import ChatCompletionResponse


class TestAnthropicMessageConverter(unittest.TestCase):
    """Test suite for the AnthropicMessageConverter class."""

    def setUp(self):
        """Set up the test case."""
        self.converter = AnthropicMessageConverter()

    def test_convert_request_single_user_message(self):
        """Test converting a single user message."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        system_message, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(system_message, [])
        self.assertEqual(
            converted_messages, [{"role": "user", "content": "Hello, how are you?"}]
        )

    def test_convert_request_with_system_message(self):
        """Test converting a request with a system message."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather?"},
        ]
        system_message, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(system_message, "You are a helpful assistant.")
        self.assertEqual(
            converted_messages, [{"role": "user", "content": "What is the weather?"}]
        )

    def test_convert_request_with_tool_use_message(self):
        """Test converting a request with a tool use message."""
        messages = [
            {"role": "tool", "tool_call_id": "tool123", "content": "Weather data here."}
        ]
        system_message, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(system_message, [])
        self.assertEqual(
            converted_messages,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool123",
                            "content": "Weather data here.",
                        }
                    ],
                }
            ],
        )

    def test_convert_response_normal_message(self):
        """Test converting a normal text response."""
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        content_mock = MagicMock()
        content_mock.type = "text"
        content_mock.text = "The weather is sunny."
        response.content = [content_mock]

        normalized_response = self.converter.convert_response(response)

        self.assertIsInstance(normalized_response, ChatCompletionResponse)
        self.assertEqual(normalized_response.choices[0].finish_reason, "stop")
        self.assertEqual(normalized_response.usage.prompt_tokens, 10)
        self.assertEqual(normalized_response.usage.completion_tokens, 5)
        self.assertEqual(normalized_response.usage.total_tokens, 15)
        self.assertEqual(
            normalized_response.choices[0].message.content, "The weather is sunny."
        )

    def test_convert_response_with_tool_use(self):
        """Test converting a response containing a tool use request."""
        response = MagicMock()
        response.id = "msg_01Aq9w938a90dw8q"
        response.model = "claude-3-5-sonnet-20241022"
        response.role = "assistant"
        response.stop_reason = "tool_use"
        response.usage.input_tokens = 20
        response.usage.output_tokens = 10
        tool_use_mock = MagicMock()
        tool_use_mock.type = "tool_use"
        tool_use_mock.id = "tool123"
        tool_use_mock.name = "get_weather"
        tool_use_mock.input = {"location": "Paris"}

        text_mock = MagicMock()
        text_mock.type = "text"
        text_mock.text = "<thinking>I need to call the get_weather function</thinking>"

        response.content = [tool_use_mock, text_mock]

        normalized_response = self.converter.convert_response(response)

        self.assertIsInstance(normalized_response, ChatCompletionResponse)
        self.assertEqual(normalized_response.choices[0].finish_reason, "tool_calls")
        self.assertEqual(normalized_response.usage.prompt_tokens, 20)
        self.assertEqual(normalized_response.usage.completion_tokens, 10)
        self.assertEqual(normalized_response.usage.total_tokens, 30)
        self.assertEqual(
            normalized_response.choices[0].message.content,
            "<thinking>I need to call the get_weather function</thinking>",
        )
        self.assertEqual(len(normalized_response.choices[0].message.tool_calls), 1)
        self.assertEqual(
            normalized_response.choices[0].message.tool_calls[0].id, "tool123"
        )
        self.assertEqual(
            normalized_response.choices[0].message.tool_calls[0].function.name,
            "get_weather",
        )

    def test_convert_tool_spec(self):
        """Test converting OpenAI tool specifications to Anthropic format."""
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name."}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        anthropic_tools = self.converter.convert_tool_spec(openai_tools)

        self.assertEqual(len(anthropic_tools), 1)
        self.assertEqual(anthropic_tools[0]["name"], "get_weather")
        self.assertEqual(anthropic_tools[0]["description"], "Get the weather.")
        self.assertEqual(
            anthropic_tools[0]["input_schema"],
            {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."}
                },
                "required": ["location"],
            },
        )

    def test_convert_request_with_tool_call_and_result(self):
        """Test converting a request with a tool call and its result."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check the weather.",
                "tool_calls": [
                    {
                        "id": "tool123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                        "type": "function",
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tool123",
                "content": "The weather in Paris is sunny.",
            },
        ]
        system_message, converted_messages = self.converter.convert_request(messages)

        self.assertEqual(system_message, [])
        self.assertEqual(len(converted_messages), 2)
        self.assertEqual(converted_messages[0]["role"], "assistant")
        self.assertEqual(converted_messages[1]["role"], "user")
        self.assertEqual(
            converted_messages[0]["content"],
            [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "tool123",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                },
            ],
        )
        self.assertEqual(
            converted_messages[1]["content"],
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool123",
                    "content": "The weather in Paris is sunny.",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
