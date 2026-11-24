"""Preserve response fields through a real OpenAI-compatible SDK transport."""

import json

import httpx
import pytest

from aisuite.providers.deepseek_provider import DeepseekProvider


@pytest.mark.parametrize("finish_reason", ["length", "tool_calls", "content_filter"])
def test_deepseek_response_keeps_finish_reason_and_assistant_metadata(finish_reason):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-metadata",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-reasoner",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "I should inspect the weather before answering.",
                            "refusal": (
                                "Request cannot be fulfilled."
                                if finish_reason == "content_filter"
                                else None
                            ),
                            "tool_calls": (
                                [
                                    {
                                        "id": "call_weather",
                                        "type": "function",
                                        "function": {
                                            "name": "weather",
                                            "arguments": '{"city":"Paris"}',
                                        },
                                    }
                                ]
                                if finish_reason == "tool_calls"
                                else None
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        provider = DeepseekProvider(api_key="test-key", http_client=http_client)
        response = provider.chat_completions_create(
            "deepseek-reasoner", [{"role": "user", "content": "Weather in Paris?"}]
        )
        choice = response.choices[0]
        assert choice.finish_reason == finish_reason
        assert (
            choice.message.reasoning_content
            == "I should inspect the weather before answering."
        )
        assert choice.message.refusal == (
            "Request cannot be fulfilled."
            if finish_reason == "content_filter"
            else None
        )
        assert response.usage.total_tokens == 15
        if finish_reason == "tool_calls":
            assert choice.message.tool_calls[0].function.name == "weather"
            replay = choice.message.model_dump(exclude_none=True)
            provider.chat_completions_create(
                "deepseek-reasoner",
                [
                    replay,
                    {
                        "role": "tool",
                        "tool_call_id": "call_weather",
                        "content": "sunny",
                    },
                ],
            )
            assert (
                requests[1]["messages"][0]["reasoning_content"]
                == replay["reasoning_content"]
            )
