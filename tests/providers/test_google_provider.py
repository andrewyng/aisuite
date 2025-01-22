import pytest
from unittest.mock import patch, MagicMock
from aisuite.providers.google_provider import GoogleProvider
from vertexai.generative_models import Content, Part
import json


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "path-to-service-account-json")
    monkeypatch.setenv("GOOGLE_PROJECT_ID", "vertex-project-id")
    monkeypatch.setenv("GOOGLE_REGION", "us-central1")


def test_missing_env_vars():
    """Test that an error is raised if required environment variables are missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError) as exc_info:
            GoogleProvider()
        assert "Missing one or more required Google environment variables" in str(
            exc_info.value
        )


def test_vertex_interface():
    """High-level test that the interface is initialized and chat completions are requested successfully."""

    # Test case 1: Regular text response
    def test_text_response():
        user_greeting = "Hello!"
        message_history = [{"role": "user", "content": user_greeting}]
        selected_model = "our-favorite-model"
        response_text_content = "mocked-text-response-from-model"

        interface = GoogleProvider()
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = response_text_content
        # Ensure function_call attribute doesn't exist
        del mock_response.candidates[0].content.parts[0].function_call

        with patch(
            "aisuite.providers.google_provider.GenerativeModel"
        ) as mock_generative_model:
            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model
            mock_chat = MagicMock()
            mock_model.start_chat.return_value = mock_chat
            mock_chat.send_message.return_value = mock_response

            response = interface.chat_completions_create(
                messages=message_history,
                model=selected_model,
                temperature=0.7,
            )

            # Assert the response is in the correct format
            assert response.choices[0].message.content == response_text_content
            assert response.choices[0].finish_reason == "stop"

    # Test case 2: Function call response
    def test_function_call():
        user_greeting = "What's the weather?"
        message_history = [{"role": "user", "content": user_greeting}]
        selected_model = "our-favorite-model"

        interface = GoogleProvider()
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]

        # Mock the function call response
        function_call_mock = MagicMock()
        function_call_mock.name = "get_weather"
        function_call_mock.args = {"location": "San Francisco"}
        mock_response.candidates[0].content.parts[0].function_call = function_call_mock
        mock_response.candidates[0].content.parts[0].text = None

        with patch(
            "aisuite.providers.google_provider.GenerativeModel"
        ) as mock_generative_model:
            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model
            mock_chat = MagicMock()
            mock_model.start_chat.return_value = mock_chat
            mock_chat.send_message.return_value = mock_response

            response = interface.chat_completions_create(
                messages=message_history,
                model=selected_model,
                temperature=0.7,
            )

            # Assert the response contains the function call
            assert response.choices[0].message.content is None
            assert response.choices[0].message.tool_calls[0].type == "function"
            assert (
                response.choices[0].message.tool_calls[0].function.name == "get_weather"
            )
            assert json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            ) == {"location": "San Francisco"}
            assert response.choices[0].finish_reason == "tool_calls"

    # Run both test cases
    test_text_response()
    test_function_call()


def test_convert_openai_to_vertex_ai():
    """Test the message conversion from OpenAI format to Vertex AI format."""
    interface = GoogleProvider()
    message = {"role": "user", "content": "Hello!"}

    # Use the transformer to convert the message
    result = interface.transformer.convert_request([message])

    # Verify the conversion result
    assert len(result) == 1
    assert isinstance(result[0], Content)
    assert result[0].role == "user"
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], Part)
    assert result[0].parts[0].text == "Hello!"


def test_role_conversions():
    """Test that different message roles are converted correctly."""
    interface = GoogleProvider()

    messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]

    result = interface.transformer.convert_request(messages)

    # System and user messages should both be converted to "user" role in Vertex AI
    assert len(result) == 3
    assert result[0].role == "user"  # system converted to user
    assert result[0].parts[0].text == "System message"

    assert result[1].role == "user"
    assert result[1].parts[0].text == "User message"

    assert result[2].role == "model"  # assistant converted to model
    assert result[2].parts[0].text == "Assistant message"
