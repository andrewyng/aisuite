import pytest
from unittest.mock import patch, MagicMock

from aisuite.providers.watsonx_provider import WatsonxProvider


@pytest.fixture(autouse=True)
def set_api_credentials_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("WATSONX_CLUSTER_URL", "https://eu-de.ml.cloud.ibm.com")
    monkeypatch.setenv("IBM_IAM_ACCESS_TOKEN", "test-access-token")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project-id")


def test_completion():
    """Test that completions request successfully."""

    user_greeting = "Howdy!"
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "best-model-ever"
    chosen_temperature = 0.77
    response_text_content = "mocked-text-response-from-watsonx-model"
    url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-10-25"
    headers = {
        "Authorization": f"Bearer test-access-token",
        "Content-Type": "application/json",
    }
    watsonx = WatsonxProvider()
    mock_response = {"choices": [{"message": {"content": response_text_content}}]}

    with patch(
        "httpx.post",
        return_value=MagicMock(status_code=200, json=lambda: mock_response),
    ) as mock_post:
        response = watsonx.chat_completions_create(
            model=selected_model,
            messages=message_history,
            temperature=chosen_temperature,
        )

        mock_post.assert_called_once_with(
            url,
            json={
                "model_id": selected_model,
                "project_id": "test-project-id",
                "messages": message_history,
                "temperature": chosen_temperature,
            },
            headers=headers,
            timeout=30,
        )

        assert response.choices[0].message.content == response_text_content
