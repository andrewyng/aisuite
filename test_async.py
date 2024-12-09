"""
Test script for async functionality.
"""
import os
import asyncio
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aisuite.async_client import AsyncClient
from aisuite.providers.async_openai_provider import OpenAIError

async def test_async():
    """Test async functionality with OpenAI."""
    client = None
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running the tests."
            )

        # Configure with OpenAI API key
        client = AsyncClient({
            "openai": {"api_key": os.getenv("OPENAI_API_KEY")}
        })

        # Test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"}
        ]

        # Test regular async completion
        print("\nTesting single async completion:")
        try:
            response = await client.chat.completions.create(
                model="openai:gpt-3.5-turbo",
                messages=messages
            )
            print(f"Response: {response.choices[0].message.content}")
        except OpenAIError as e:
            print(f"Error during single completion: {str(e)}")
            return

        # Test concurrent completions
        print("\nTesting concurrent completions:")
        try:
            models = ["openai:gpt-3.5-turbo", "openai:gpt-4"]
            tasks = [
                client.chat.completions.create(model=model, messages=messages)
                for model in models
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for model, response in zip(models, responses):
                if isinstance(response, Exception):
                    print(f"\n{model} error: {str(response)}")
                else:
                    print(f"\n{model}: {response.choices[0].message.content}")
        except Exception as e:
            print(f"Error during concurrent completions: {str(e)}")

        # Test streaming
        print("\nTesting streaming completion:")
        try:
            async for chunk in client.chat.completions.create(
                model="openai:gpt-3.5-turbo",
                messages=messages,
                stream=True
            ):
                if chunk.choices[0].delta:
                    print(chunk.choices[0].delta.content or "", end="", flush=True)
            print("\n")
        except OpenAIError as e:
            print(f"Error during streaming: {str(e)}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        if client:
            await client.close()

if __name__ == "__main__":
    asyncio.run(test_async())
