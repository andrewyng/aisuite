import unittest
from aisuite.providers.azure_provider import AzureMessageConverter
from aisuite.framework.message import Message, ChatCompletionMessageToolCall
from aisuite.framework import ChatCompletionResponse


class TestAzureMessageConverter(unittest.TestCase):
    def setUp(self):
        self.converter = AzureMessageConverter()

    def test_convert_request_dict_message(self):
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        converted_messages = self.converter.convert_request(messages)

        self.assertEqual(
            converted_messages, [{"role": "user", "content": "Hello, how are you?"}]
        )

    def test_convert_request_message_object(self):
        message = Message(role="user", content="Hello", tool_calls=None, refusal=None)
        messages = [message]
        converted_messages = self.converter.convert_request(messages)

        expected_message = {
            "role": "user",
            "content": "Hello",
            "tool_calls": None,
            "refusal": None,
        }
        self.assertEqual(converted_messages, [expected_message])

    def test_convert_response_basic(self):
        azure_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    }
                }
            ]
        }

        response = self.converter.convert_response(azure_response)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(
            response.choices[0].message.content, "Hello! How can I help you?"
        )
        self.assertEqual(response.choices[0].message.role, "assistant")
        self.assertIsNone(response.choices[0].message.tool_calls)

    def test_convert_response_with_tool_calls(self):
        azure_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check the weather.",
                        "tool_calls": [
                            {
                                "id": "tool123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "London"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        response = self.converter.convert_response(azure_response)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(
            response.choices[0].message.content, "Let me check the weather."
        )
        self.assertEqual(len(response.choices[0].message.tool_calls), 1)

        tool_call = response.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.id, "tool123")
        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function.name, "get_weather")
        self.assertEqual(tool_call.function.arguments, '{"location": "London"}')


if __name__ == "__main__":
    unittest.main()
