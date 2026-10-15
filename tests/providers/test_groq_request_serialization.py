"""Exercise tool-loop serialization through the real SDK's offline transport."""

import json

import httpx

from aisuite import Client


def test_tool_loop_omits_internal_reasoning_field():
    requests = []

    def weather(city: str) -> str:
        """Get weather for a city.

        Args:
            city: City name.
        """
        return f"Sunny in {city}"

    def respond(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": "Done"}
        finish_reason = "stop"
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": "offline-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "llama-3.3-70b-versatile",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = Client(
            provider_configs={
                "groq": {
                    "api_key": "offline-test-only",
                    "http_client": transport,
                    "max_retries": 0,
                }
            }
        )
        response = client.chat.completions.create(
            model="groq:llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[weather],
            max_turns=2,
        )

    assert response.choices[0].message.content == "Done"
    assert len(requests) == 2
    assert "reasoning_content" not in requests[1]["messages"][1]
    assert json.loads(requests[1]["messages"][2]["content"]) == "Sunny in Paris"
