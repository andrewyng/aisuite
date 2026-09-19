"""Opt-in live tests for the Parasail provider."""

import os

import pytest

import aisuite as ai


MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.skipif(
    not os.getenv("PARASAIL_API_KEY"),
    reason="PARASAIL_API_KEY is required for the live Parasail test",
)
def test_parasail_chat_completion():
    client = ai.Client()

    response = client.chat.completions.create(
        model=f"parasail:{MODEL}",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert response.choices
    assert response.choices[0].message.content
