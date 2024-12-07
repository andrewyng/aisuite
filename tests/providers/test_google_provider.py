import pytest, json
from unittest.mock import patch, MagicMock
from aisuite.providers.google_provider import GoogleProvider
from vertexai.generative_models import Content, Part


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

    user_greeting = "Hello!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "our-favorite-model"
    response_text_content = "mocked-text-response-from-model"

    interface = GoogleProvider()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = response_text_content

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
        # Assert that GenerativeModel was called with correct arguments.
        mock_generative_model.assert_called_once()
        args, kwargs = mock_generative_model.call_args

        assert kwargs["model_name"] == selected_model
        assert "generation_config" in kwargs

        # Assert that start_chat was called with correct history.
        mock_model.start_chat.assert_called_once()
        _chat_args, chat_kwargs = mock_model.start_chat.call_args
        assert "history" in chat_kwargs
        assert isinstance(chat_kwargs["history"], list)

        # Assert that send_message was called with the last message.
        mock_chat.send_message.assert_called_once_with(user_greeting)

        # Assert that the response is in the correct format.
        assert response.choices[0].message.content == response_text_content


def test_convert_openai_to_vertex_ai():
    interface = GoogleProvider()
    message_history = [{"role": "user", "content": "Hello!"}]
    result = interface.convert_openai_to_vertex_ai(message_history)
    assert isinstance(result[0], Content)
    assert result[0].role == "user"
    assert len(result[0].parts) == 1
    assert isinstance(result[0].parts[0], Part)
    assert result[0].parts[0].text == "Hello!"


def test_transform_roles():
    interface = GoogleProvider()

    messages = [
        {"role": "system", "content": "Google: system message = 1st user message."},
        {"role": "user", "content": "User message 1."},
        {"role": "assistant", "content": "Assistant message 1."},
    ]

    expected_output = [
        {"role": "user", "content": "Google: system message = 1st user message."},
        {"role": "user", "content": "User message 1."},
        {"role": "model", "content": "Assistant message 1."},
    ]

    result = interface.transform_roles(messages)

    assert result == expected_output


def test_tool_calls():
    """Test that the tools parameter is correctly passed and tool calls are handled properly."""

    user_input = "Call the example_tool with param1=value1"
    message_history = [{"role": "user", "content": user_input}]
    selected_model = "our-favorite-model"
    response_text_content = "mocked-text-response-from-model"
    tools = [
        {
            "function": {
                "name": "example_tool",
                "description": "An example tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {
                            "type": "string",
                            "description": "An example parameter",
                        }
                    },
                },
            }
        }
    ]

    interface = GoogleProvider()
    mock_response = MagicMock()
    mock_response.candidates = [MagicMock()]
    mock_response.candidates[0].content.parts[0].text = response_text_content

    # Create a mock function call with the necessary attributes
    mock_function_call = MagicMock()
    mock_function_call.name = "example_tool"
    mock_function_call.args = {"param1": "value1"}

    mock_response.candidates[0].function_calls = [mock_function_call]

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
            tools=tools,
        )

        # Assert that the response is in the correct format.
        assert response.choices[0].message.content == ""
        assert response.choices[0].finish_reason == "tool_calls"
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].message.tool_calls[0].function.name == "example_tool"
        assert (
            response.choices[0].message.tool_calls[0].function.arguments
            == '{"param1": "value1"}'
        )
