import unittest
from unittest.mock import MagicMock, patch

import pytest

from aisuite import Client
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
            "stopReason": "end_turn",
        }

        normalized_response = self.converter.convert_response(response)

        self.assertIsInstance(normalized_response, ChatCompletionResponse)
        self.assertEqual(normalized_response.choices[0].finish_reason, "stop")
        self.assertEqual(
            normalized_response.choices[0].message.content,
            "The most popular song on WZPZ is Elemental Hotel by 8 Storey Hike.",
        )

    def test_convert_response_stop_sequence(self):
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "stop_sequence",
        }

        normalized_response = self.converter.convert_response(response)

        self.assertEqual(normalized_response.choices[0].finish_reason, "stop")

    def test_convert_response_max_tokens(self):
        response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "max_tokens",
        }

        normalized_response = self.converter.convert_response(response)

        self.assertEqual(normalized_response.choices[0].finish_reason, "length")


@pytest.mark.parametrize(
    "content, expected_text, stop_reason, finish_reason",
    [
        (
            [
                {"reasoningContent": {"reasoningText": {"text": "Internal reasoning"}}},
                {"text": "The answer is 42."},
            ],
            "The answer is 42.",
            "end_turn",
            "stop",
        ),
        (
            [
                {"text": "Hello, "},
                {"reasoningContent": {"redactedContent": b"redacted"}},
                {"text": "world!\n"},
            ],
            "Hello, world!\n",
            "stop_sequence",
            "stop",
        ),
        ([{"text": ""}], "", "end_turn", "stop"),
        ([], "", "max_tokens", "length"),
        (
            [{"reasoningContent": {"reasoningText": {"text": "Internal reasoning"}}}],
            "",
            "max_tokens",
            "length",
        ),
    ],
    ids=["leading-reasoning", "multiple-text", "empty-text", "empty", "no-text"],
)
def test_client_preserves_bedrock_response_text(
    content, expected_text, stop_reason, finish_reason
):
    with patch("aisuite.providers.aws_provider.boto3.client") as create_client:
        create_client.return_value.converse.return_value = {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop_reason,
            "usage": {"inputTokens": 17, "outputTokens": 5, "totalTokens": 22},
        }
        response = Client().chat.completions.create(
            model="aws:anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert response.choices[0].message.content == expected_text
    assert response.choices[0].finish_reason == finish_reason
    assert response.choices[0].message.tool_calls is None
    assert response.usage.prompt_tokens == 17
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 22


if __name__ == "__main__":
    unittest.main()
