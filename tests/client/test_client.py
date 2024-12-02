import unittest
from unittest.mock import patch
from aisuite import Client


class TestClient(unittest.TestCase):
    @patch("aisuite.providers.tongyi_provider.TongyiProvider.chat_completions_create")
    @patch("aisuite.providers.mistral_provider.MistralProvider.chat_completions_create")
    @patch("aisuite.providers.groq_provider.GroqProvider.chat_completions_create")
    @patch("aisuite.providers.openai_provider.OpenaiProvider.chat_completions_create")
    @patch("aisuite.providers.aws_provider.AwsProvider.chat_completions_create")
    @patch("aisuite.providers.azure_provider.AzureProvider.chat_completions_create")
    @patch(
        "aisuite.providers.anthropic_provider.AnthropicProvider.chat_completions_create"
    )
    @patch("aisuite.providers.google_provider.GoogleProvider.chat_completions_create")
    @patch(
        "aisuite.providers.fireworks_provider.FireworksProvider.chat_completions_create"
    )
    def test_client_chat_completions(
        self,
        mock_fireworks,
        mock_google,
        mock_anthropic,
        mock_azure,
        mock_bedrock,
        mock_openai,
        mock_groq,
        mock_mistral,
        mock_tongyi,
    ):
        # Mock responses from providers
        mock_openai.return_value = "OpenAI Response"
        mock_bedrock.return_value = "AWS Bedrock Response"
        mock_azure.return_value = "Azure Response"
        mock_anthropic.return_value = "Anthropic Response"
        mock_groq.return_value = "Groq Response"
        mock_mistral.return_value = "Mistral Response"
        mock_google.return_value = "Google Response"
        mock_fireworks.return_value = "Fireworks Response"
        mock_tongyi.return_value = "Tongyi Response"

        # Provider configurations
        provider_configs = {
            "openai": {"api_key": "test_openai_api_key"},
            "aws": {
                "aws_access_key": "test_aws_access_key",
                "aws_secret_key": "test_aws_secret_key",
                "aws_session_token": "test_aws_session_token",
                "aws_region": "us-west-2",
            },
            "azure": {
                "api_key": "azure-api-key",
                "base_url": "https://model.ai.azure.com",
            },
            "groq": {
                "api_key": "groq-api-key",
            },
            "mistral": {
                "api_key": "mistral-api-key",
            },
            "google": {
                "project_id": "test_google_project_id",
                "region": "us-west4",
                "application_credentials": "test_google_application_credentials",
            },
            "fireworks": {
                "api_key": "fireworks-api-key",
            },
            "tongyi": {
                "api_key": "tongyi-api-key",
            },
        }

        # Initialize the client
        client = Client()
        client.configure(provider_configs)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
        ]

        # Test OpenAI model
        open_ai_model = "openai" + ":" + "gpt-4o"
        openai_response = client.chat.completions.create(
            open_ai_model, messages=messages
        )
        self.assertEqual(openai_response, "OpenAI Response")
        mock_openai.assert_called_once()

        # Test AWS Bedrock model
        bedrock_model = "aws" + ":" + "claude-v3"
        bedrock_response = client.chat.completions.create(
            bedrock_model, messages=messages
        )
        self.assertEqual(bedrock_response, "AWS Bedrock Response")
        mock_bedrock.assert_called_once()

        # Test Azure model
        azure_model = "azure" + ":" + "azure-model"
        azure_response = client.chat.completions.create(azure_model, messages=messages)
        self.assertEqual(azure_response, "Azure Response")
        mock_azure.assert_called_once()

        # Test Anthropic model
        anthropic_model = "anthropic" + ":" + "anthropic-model"
        anthropic_response = client.chat.completions.create(
            anthropic_model, messages=messages
        )
        self.assertEqual(anthropic_response, "Anthropic Response")
        mock_anthropic.assert_called_once()

        # Test Groq model
        groq_model = "groq" + ":" + "groq-model"
        groq_response = client.chat.completions.create(groq_model, messages=messages)
        self.assertEqual(groq_response, "Groq Response")
        mock_groq.assert_called_once()

        # Test Mistral model
        mistral_model = "mistral" + ":" + "mistral-model"
        mistral_response = client.chat.completions.create(
            mistral_model, messages=messages
        )
        self.assertEqual(mistral_response, "Mistral Response")
        mock_mistral.assert_called_once()

        # Test Google model
        google_model = "google" + ":" + "google-model"
        google_response = client.chat.completions.create(
            google_model, messages=messages
        )
        self.assertEqual(google_response, "Google Response")
        mock_google.assert_called_once()

        # Test Fireworks model
        fireworks_model = "fireworks" + ":" + "fireworks-model"
        fireworks_response = client.chat.completions.create(
            fireworks_model, messages=messages
        )
        self.assertEqual(fireworks_response, "Fireworks Response")
        mock_fireworks.assert_called_once()

        # Test Tongyi model
        tongyi_model = "tongyi" + ":" + "qwen-plus"
        tongyi_response = client.chat.completions.create(
            tongyi_model, messages=messages
        )
        self.assertEqual(tongyi_response, "Tongyi Response")
        mock_tongyi.assert_called_once()

        # Test that new instances of Completion are not created each time we make an inference call.
        compl_instance = client.chat.completions
        next_compl_instance = client.chat.completions
        assert compl_instance is next_compl_instance

    def test_invalid_provider_in_client_config(self):
        # Testing an invalid provider name in the configuration
        invalid_provider_configs = {
            "invalid_provider": {"api_key": "invalid_api_key"},
        }

        # Expect ValueError when initializing Client with invalid provider
        with self.assertRaises(ValueError) as context:
            client = Client(invalid_provider_configs)

        # Verify the error message
        self.assertIn(
            "Invalid provider key 'invalid_provider'. Supported providers: ",
            str(context.exception),
        )

    @patch("aisuite.providers.openai_provider.OpenaiProvider.chat_completions_create")
    def test_invalid_model_format_in_create(self, mock_openai):
        # Valid provider configurations
        provider_configs = {
            "openai": {"api_key": "test_openai_api_key"},
        }

        # Initialize the client with valid provider
        client = Client()
        client.configure(provider_configs)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ]

        # Invalid model format
        invalid_model = "invalidmodel"

        # Expect ValueError when calling create with invalid model format
        with self.assertRaises(ValueError) as context:
            client.chat.completions.create(invalid_model, messages=messages)

        # Verify the error message
        self.assertIn(
            "Invalid model format. Expected 'provider:model'", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
