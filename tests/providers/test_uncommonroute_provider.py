import unittest
from unittest.mock import patch, MagicMock
from aisuite.providers.uncommonroute_provider import UncommonrouteProvider


class TestUncommonrouteProvider(unittest.TestCase):
    def setUp(self):
        self.provider = UncommonrouteProvider()

    def test_default_base_url(self):
        self.assertEqual(self.provider.base_url, "http://localhost:8403/v1")

    def test_custom_base_url(self):
        provider = UncommonrouteProvider(base_url="http://myhost:9000/v1")
        self.assertEqual(provider.base_url, "http://myhost:9000/v1")

    def test_resolve_model_auto(self):
        self.assertEqual(self.provider._resolve_model("auto"), "uncommon-route/auto")

    def test_resolve_model_fast(self):
        self.assertEqual(self.provider._resolve_model("fast"), "uncommon-route/fast")

    def test_resolve_model_best(self):
        self.assertEqual(self.provider._resolve_model("best"), "uncommon-route/best")

    def test_resolve_model_passthrough(self):
        self.assertEqual(self.provider._resolve_model("gpt-4o"), "gpt-4o")

    @patch("aisuite.providers.uncommonroute_provider.httpx.post")
    def test_chat_completions_create(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        result = self.provider.chat_completions_create("auto", messages)

        self.assertEqual(result.choices[0].message.content, "Hello!")

        call_args = mock_post.call_args
        self.assertIn("/chat/completions", call_args[0][0])
        self.assertEqual(call_args[1]["json"]["model"], "uncommon-route/auto")

    @patch("aisuite.providers.uncommonroute_provider.httpx.post")
    def test_connect_error(self, mock_post):
        from aisuite.provider import LLMError
        import httpx

        mock_post.side_effect = httpx.ConnectError("refused")

        with self.assertRaises(LLMError) as ctx:
            self.provider.chat_completions_create(
                "auto", [{"role": "user", "content": "Hi"}]
            )
        self.assertIn("uncommon-route serve", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
