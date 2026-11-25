import json

import httpx
import pytest

from aisuite.providers.together_provider import TogetherProvider


@pytest.mark.parametrize("count", [1, 2, 3])
def test_together_returns_every_requested_completion(monkeypatch, count):
    choices = [
        {
            "index": index,
            "message": {"role": "assistant", "content": f"Candidate {index}"},
            "finish_reason": "stop",
        }
        for index in range(count)
    ]

    def respond(request):
        assert json.loads(request.content)["n"] == count
        return httpx.Response(200, json={"choices": choices})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            "aisuite.providers.together_provider.httpx.post", client.post
        )
        provider = TogetherProvider(api_key="test-key")
        response = provider.chat_completions_create(
            model="test-model",
            messages=[{"role": "user", "content": "Propose several solutions"}],
            n=count,
        )

    assert [choice.message.content for choice in response.choices] == [
        f"Candidate {index}" for index in range(count)
    ]
