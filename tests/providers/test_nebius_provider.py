import pytest
from unittest.mock import patch, MagicMock

from aisuite.providers.nebius_provider import NebiusProvider

def test_nebius_provider():
    """High-level test that the provider is initialized and chat completions are requested successfully."""

    user_greeting = "We are testing you. Please say 'One two three' and nothing more."
    message_history = [{"role": "user", "content": user_greeting}]
    selected_model = "Qwen/Qwen2.5-32B-Instruct-fast"
    chosen_top_p = 0.01
    response_text_content = "One two three"

    provider = NebiusProvider()
    print(provider.api_key)
    response = provider.chat_completions_create(model=selected_model, messages=message_history, top_p=chosen_top_p)

    assert response.choices[0].message.content == response_text_content
