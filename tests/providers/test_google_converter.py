import unittest
from unittest.mock import MagicMock
from aisuite.providers.google_provider import GoogleMessageConverter
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function
from aisuite.framework import ChatCompletionResponse


class TestGoogleMessageConverter(unittest.TestCase):

    def setUp(self):
        self.converter = GoogleMessageConverter()

    def test_convert_request_user_message(self):
        messages = [{"role": "user", "content": "What is the weather today?"}]
        converted_messages = self.converter.convert_request(messages)

        self.assertEqual(len(converted_messages), 1)
        self.assertEqual(converted_messages[0].role, "user")
        self.assertEqual(
            converted_messages[0].parts[0].text, "What is the weather today?"
        )

    def test_convert_request_tool_result_message(self):
        messages = [
            {
                "role": "tool",
                "name": "get_weather",
                "content": '{"temperature": "15", "unit": "Celsius"}',
            }
        ]
        converted_messages = self.converter.convert_request(messages)

        self.assertEqual(len(converted_messages), 1)
        self.assertEqual(converted_messages[0].function_response.name, "get_weather")
        self.assertEqual(
            converted_messages[0].function_response.response,
            {"temperature": "15", "unit": "Celsius"},
        )

    def test_convert_request_assistant_message(self):
        messages = [
            {
                "role": "assistant",
                "content": "The weather is sunny with a temperature of 25 degrees Celsius.",
            }
        ]
        converted_messages = self.converter.convert_request(messages)

        self.assertEqual(len(converted_messages), 1)
        self.assertEqual(converted_messages[0].role, "model")
        self.assertEqual(
            converted_messages[0].parts[0].text,
            "The weather is sunny with a temperature of 25 degrees Celsius.",
        )

    def test_convert_response_with_function_call(self):
        function_call_mock = MagicMock()
        function_call_mock.name = "get_exchange_rate"
        function_call_mock.args = {
            "currency_from": "AUD",
            "currency_to": "SEK",
            "currency_date": "latest",
        }

        response = MagicMock()
        response.candidates = [
            MagicMock(
                content=MagicMock(parts=[MagicMock(function_call=function_call_mock)]),
                finish_reason="function_call",
            )
        ]

        normalized_response = self.converter.convert_response(response)

        self.assertIsInstance(normalized_response, ChatCompletionResponse)
        self.assertEqual(normalized_response.choices[0].finish_reason, "tool_calls")
        self.assertEqual(
            normalized_response.choices[0].message.tool_calls[0].function.name,
            "get_exchange_rate",
        )
        self.assertEqual(
            normalized_response.choices[0].message.tool_calls[0].function.arguments,
            '{"currency_from": "AUD", "currency_to": "SEK", "currency_date": "latest"}',
        )

    def test_convert_response_with_text(self):
        response = MagicMock()
        text_content = "The current exchange rate is 7.50 SEK per AUD."

        mock_part = MagicMock()
        mock_part.text = text_content
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        response.candidates = [mock_candidate]

        normalized_response = self.converter.convert_response(response)

        self.assertIsInstance(normalized_response, ChatCompletionResponse)
        self.assertEqual(normalized_response.choices[0].finish_reason, "stop")
        self.assertEqual(normalized_response.choices[0].message.content, text_content)


if __name__ == "__main__":
    unittest.main()
