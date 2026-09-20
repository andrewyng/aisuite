"""Tests for the Eas_paiProvider and Eas_paiMessageConverter."""

import json
import unittest
from unittest.mock import patch, MagicMock

from aisuite.providers.eas_pai_provider import Eas_paiProvider, Eas_paiMessageConverter
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message


class TestEasPaiMessageConverter(unittest.TestCase):
    """Test suite for the Eas_paiMessageConverter class."""

    def setUp(self):
        """Set up the test case."""
        self.converter = Eas_paiMessageConverter()

    def test_convert_request_dict_messages(self):
        """Test converting dict messages."""
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        converted = self.converter.convert_request(messages)

        self.assertEqual(len(converted), 2)
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"], "Hello!")
        self.assertEqual(converted[1]["role"], "assistant")
        self.assertEqual(converted[1]["content"], "Hi there!")

    def test_convert_request_message_objects(self):
        """Test converting Message objects."""
        messages = [
            Message(role="user", content="Hello!"),
        ]
        converted = self.converter.convert_request(messages)

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0]["role"], "user")
        self.assertEqual(converted[0]["content"], "Hello!")

    def test_convert_response_normal_message(self):
        """Test converting a normal text response."""
        resp_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    }
                }
            ]
        }

        response = self.converter.convert_response(resp_json)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(
            response.choices[0].message.content, "Hello! How can I help you?"
        )
        self.assertEqual(response.choices[0].message.role, "assistant")

    def test_convert_response_with_tool_calls(self):
        """Test converting a response with tool calls."""
        resp_json = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Beijing"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        response = self.converter.convert_response(resp_json)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertIsNone(response.choices[0].message.content)
        self.assertEqual(len(response.choices[0].message.tool_calls), 1)
        self.assertEqual(response.choices[0].message.tool_calls[0].id, "call_123")
        self.assertEqual(response.choices[0].message.tool_calls[0].type, "function")
        self.assertEqual(
            response.choices[0].message.tool_calls[0].function.name, "get_weather"
        )
        self.assertEqual(
            response.choices[0].message.tool_calls[0].function.arguments,
            '{"location": "Beijing"}',
        )


class TestEasPaiProvider(unittest.TestCase):
    """Test suite for the Eas_paiProvider class."""

    def test_init_with_config(self):
        """Test initialization with config parameters."""
        provider = Eas_paiProvider(
            base_url="https://test.pai-eas.aliyuncs.com/api/predict/test",
            api_key="test-token",
        )

        self.assertEqual(
            provider.base_url, "https://test.pai-eas.aliyuncs.com/api/predict/test"
        )
        self.assertEqual(provider.api_key, "test-token")

    def test_init_with_env_vars(self):
        """Test initialization with environment variables."""
        with patch.dict(
            "os.environ",
            {
                "EAS_PAI_BASE_URL": "https://env.pai-eas.aliyuncs.com/api/predict/env",
                "EAS_PAI_API_KEY": "env-token",
            },
        ):
            provider = Eas_paiProvider()

            self.assertEqual(
                provider.base_url,
                "https://env.pai-eas.aliyuncs.com/api/predict/env",
            )
            self.assertEqual(provider.api_key, "env-token")

    def test_init_missing_api_key_raises_error(self):
        """Test that missing api_key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                Eas_paiProvider(
                    base_url="https://test.pai-eas.aliyuncs.com/api/predict/test"
                )

            self.assertIn("api_key is required", str(context.exception))

    def test_init_missing_base_url_raises_error(self):
        """Test that missing base_url raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                Eas_paiProvider(api_key="test-token")

            self.assertIn("base_url is required", str(context.exception))

    def test_chat_completions_create(self):
        """Test chat completions create request."""
        provider = Eas_paiProvider(
            base_url="https://test.pai-eas.aliyuncs.com/api/predict/test",
            api_key="test-token",
        )

        mock_response = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello from EAS!"}}
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            response = provider.chat_completions_create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7,
            )

            # Verify the request was made
            mock_urlopen.assert_called_once()
            request = mock_urlopen.call_args[0][0]

            # Verify request body
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["model"], "qwen-turbo")
            self.assertEqual(body["messages"], [{"role": "user", "content": "Hi"}])
            self.assertEqual(body["temperature"], 0.7)

            # Verify headers
            self.assertEqual(request.get_header("Content-type"), "application/json")
            self.assertEqual(request.get_header("Authorization"), "test-token")

            # Verify response
            self.assertEqual(
                response.choices[0].message.content, "Hello from EAS!"
            )

    def test_chat_completions_create_with_tools(self):
        """Test chat completions create request with tools."""
        provider = Eas_paiProvider(
            base_url="https://test.pai-eas.aliyuncs.com/api/predict/test",
            api_key="test-token",
        )

        mock_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Shanghai"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            response = provider.chat_completions_create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=tools,
            )

            # Verify tools were included in request
            request = mock_urlopen.call_args[0][0]
            body = json.loads(request.data.decode("utf-8"))
            self.assertEqual(body["tools"], tools)

            # Verify response has tool calls
            self.assertEqual(len(response.choices[0].message.tool_calls), 1)
            self.assertEqual(
                response.choices[0].message.tool_calls[0].function.name, "get_weather"
            )

    def test_chat_completions_url_with_chat_completions_endpoint(self):
        """Test URL handling when base_url already contains /chat/completions."""
        provider = Eas_paiProvider(
            base_url="https://test.pai-eas.aliyuncs.com/v1/chat/completions",
            api_key="test-token",
        )

        mock_response = {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            provider.chat_completions_create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": "Hi"}],
            )

            request = mock_urlopen.call_args[0][0]
            self.assertEqual(
                request.full_url,
                "https://test.pai-eas.aliyuncs.com/v1/chat/completions",
            )

    def test_stream_parameter_is_removed(self):
        """Test that stream parameter is removed from kwargs."""
        provider = Eas_paiProvider(
            base_url="https://test.pai-eas.aliyuncs.com/api/predict/test",
            api_key="test-token",
        )

        mock_response = {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}]
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            provider.chat_completions_create(
                model="qwen-turbo",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,  # This should be removed
            )

            request = mock_urlopen.call_args[0][0]
            body = json.loads(request.data.decode("utf-8"))
            self.assertNotIn("stream", body)


if __name__ == "__main__":
    unittest.main()
