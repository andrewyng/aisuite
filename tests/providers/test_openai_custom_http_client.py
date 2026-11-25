import json
import threading

import httpx
import openai
import pytest

from aisuite import Client
from aisuite.providers.openai_provider import OpenaiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_custom_sync_http_client(asynchronous, stream):
    requests = []
    request_threads = []
    caller_thread = threading.get_ident()

    def respond(request):
        request_threads.append(threading.get_ident())
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("stream"):
            chunk = {
                "id": "completion",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = Client(
            provider_configs={
                "openai": {"api_key": "test-only", "http_client": transport}
            }
        )
        kwargs = {
            "model": "openai:test-model",
            "messages": [{"role": "user", "content": "question"}],
            "stream": stream,
        }
        if asynchronous:
            response = await client.chat.completions.acreate(**kwargs)
            if stream:
                contents = [chunk.choices[0].delta.content async for chunk in response]
        else:
            response = client.chat.completions.create(**kwargs)
            if stream:
                contents = [chunk.choices[0].delta.content for chunk in response]
        if stream:
            assert contents == ["answer"]
        else:
            assert response.choices[0].message.content == "answer"
    assert len(requests) == 1
    assert (request_threads[0] != caller_thread) is asynchronous
    assert requests[0]["messages"] == [{"role": "user", "content": "question"}]


@pytest.mark.asyncio
async def test_default_transport_keeps_native_async_client():
    provider = OpenaiProvider(api_key="test-only")
    try:
        assert isinstance(provider.aclient, openai.AsyncOpenAI)
    finally:
        provider.client.close()
        await provider.aclient.close()
