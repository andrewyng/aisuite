import unittest
from unittest.mock import MagicMock
from aisuite.providers.aws_provider import BedrockMessageConverter
from aisuite.framework.message import Message, ChatCompletionMessageToolCall
from aisuite.framework import ChatCompletionResponse


class TestBedrockMessageConverter(unittest.TestCase):

    def setUp(self):
        self.converter = BedrockMessageConverter()

    def test_convert_request_user_message(self):
        messages = [
            {"role": "user", "content": "What is the most popular song on WZPZ?"}
        ]
        system_message, formatted_messages = self.converter.convert_request(messages)

        self.assertEqual(system_message, [])
        self.assertEqual(len(formatted_messages), 1)
        self.assertEqual(formatted_messages[0]["role"], "user")
        self.assertEqual(
            formatted_messages[0]["content"],
            [{"text": "What is the most popular song on WZPZ?"}],
        )

    def test_convert_request_tool_result(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "tool123",
                "content": '{"song": "Elemental Hotel", "artist": "8 Storey Hike"}',
            }
        ]
        system_message, formatted_messages = self.converter.convert_request(messages)

        self.assertEqual(system_message, [])
        self.assertEqual(len(formatted_messages), 1)
        self.assertEqual(formatted_messages[0]["role"], "user")
        self.assertEqual(
            formatted_messages[0]["content"],
            [
                {
                    "toolResult": {
                        "toolUseId": "tool123",
                        "content": [
                            {
                                "json": {
                                    "song": "Elemental Hotel",
                                    "artist": "8 Storey Hike",
                                }
                            }
                        ],
                    }
                }
            ],
        )

    def test_convert_response_tool_call(self):
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tool123",
                                "name": "top_song",
                                "input": {"sign": "WZPZ"},
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
        }

        normalized_response = self.converter.convert_response(response)

        self.assertIsInstance(normalized_response, ChatCompletionResponse)
        self.assertEqual(normalized_response.choices[0].finish_reason, "tool_calls")
        tool_call = normalized_response.choices[0].message.tool_calls[0]
        self.assertEqual(tool_call.function.name, "top_song")
        self.assertEqual(tool_call.function.arguments, '{"sign": "WZPZ"}')

    def test_convert_response_text(self):
        response = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "text": "The most popular song on WZPZ is Elemental Hotel by 8 Storey Hike."
                        }
                    ],
                }
            },
            "stopReason": "complete",
        }

        normalized_response = self.converter.convert_response(response)

        self.assertIsInstance(normalized_response, ChatCompletionResponse)
        self.assertEqual(normalized_response.choices[0].finish_reason, "stop")
        self.assertEqual(
            normalized_response.choices[0].message.content,
            "The most popular song on WZPZ is Elemental Hotel by 8 Storey Hike.",
        )


if __name__ == "__main__":
    unittest.main()
