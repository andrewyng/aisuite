import json
import os
import unittest
from unittest.mock import MagicMock, patch

from aisuite.providers.azure_provider import AzureMessageConverter, AzureProvider
from aisuite.framework.message import Message, ChatCompletionMessageToolCall
from aisuite.framework import ChatCompletionResponse

BASE_URL = "https://model.region.models.ai.azure.com"
AZURE_RESPONSE = {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]}


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
            "reasoning_content": None,
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


class TestAzureProviderRequest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in ("AZURE_API_KEY", "AZURE_BASE_URL", "AZURE_API_VERSION"):
            os.environ.pop(var, None)

    def _send(self, provider, **kwargs):
        """Run a completion against a mocked endpoint and return the sent request."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(AZURE_RESPONSE).encode("utf-8")
        mock_response.__enter__.return_value = mock_response

        with patch(
            "urllib.request.urlopen", return_value=mock_response
        ) as mock_urlopen:
            response = provider.chat_completions_create(
                "my-deployment", [{"role": "user", "content": "Hello"}], **kwargs
            )

        self.assertEqual(response.choices[0].message.content, "Hi!")
        return mock_urlopen.call_args[0][0]

    @staticmethod
    def _headers(request):
        return {name.lower(): value for name, value in request.header_items()}

    def test_api_key_is_sent_in_authorization_header(self):
        provider = AzureProvider(api_key="secret-key", base_url=BASE_URL)

        request = self._send(provider)

        self.assertEqual(self._headers(request)["authorization"], "secret-key")

    def test_requires_api_key_or_entra_id_token(self):
        with self.assertRaises(ValueError):
            AzureProvider(base_url=BASE_URL)

    def test_static_azure_ad_token_is_sent_as_bearer(self):
        provider = AzureProvider(azure_ad_token="entra-token", base_url=BASE_URL)

        request = self._send(provider)

        self.assertEqual(self._headers(request)["authorization"], "Bearer entra-token")

    def test_token_provider_is_called_for_every_request(self):
        token_provider = MagicMock(side_effect=["token-1", "token-2"])
        provider = AzureProvider(
            azure_ad_token_provider=token_provider, base_url=BASE_URL
        )

        first = self._send(provider)
        second = self._send(provider)

        self.assertEqual(self._headers(first)["authorization"], "Bearer token-1")
        self.assertEqual(self._headers(second)["authorization"], "Bearer token-2")
        self.assertEqual(token_provider.call_count, 2)

    def test_entra_id_token_takes_precedence_over_api_key(self):
        os.environ["AZURE_API_KEY"] = "key-from-env"
        provider = AzureProvider(
            azure_ad_token_provider=lambda: "entra-token", base_url=BASE_URL
        )

        request = self._send(provider)

        self.assertEqual(self._headers(request)["authorization"], "Bearer entra-token")

    def test_default_headers_are_sent_with_every_request(self):
        provider = AzureProvider(
            api_key="secret-key",
            base_url=BASE_URL,
            default_headers={"x-ms-user": "alice"},
        )

        request = self._send(provider)

        headers = self._headers(request)
        self.assertEqual(headers["x-ms-user"], "alice")
        self.assertEqual(headers["authorization"], "secret-key")
        self.assertEqual(headers["content-type"], "application/json")

    def test_extra_headers_are_sent_but_not_put_in_body(self):
        provider = AzureProvider(
            api_key="secret-key",
            base_url=BASE_URL,
            default_headers={"x-ms-user": "alice", "x-team": "search"},
        )

        request = self._send(
            provider, extra_headers={"x-ms-user": "bob"}, temperature=0.2
        )

        headers = self._headers(request)
        self.assertEqual(headers["x-ms-user"], "bob")
        self.assertEqual(headers["x-team"], "search")
        body = json.loads(request.data)
        self.assertNotIn("extra_headers", body)
        self.assertEqual(body["temperature"], 0.2)


if __name__ == "__main__":
    unittest.main()
