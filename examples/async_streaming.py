"""
Example of using the async client with streaming responses.
"""
import asyncio
import aisuite as ai
from aisuite.async_client import AsyncClient

async def stream_responses():
    """Stream responses from multiple models concurrently."""
    async with AsyncClient() as client:
        models = ["openai:gpt-4", "anthropic:claude-3-sonnet-20240229"]
        messages = [
            {"role": "system", "content": "Respond in Pirate English."},
            {"role": "user", "content": "Tell me a story about a brave sailor."},
        ]

        # Create tasks for concurrent streaming
        async def process_stream(model):
            print(f"\nStarting stream from {model}:")
            async for chunk in client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.75,
                stream=True
            ):
                print(chunk.choices[0].message.content, end="", flush=True)
            print("\n")

        # Execute all streams concurrently
        tasks = [process_stream(model) for model in models]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    # Run the async streaming example
    asyncio.run(stream_responses())
