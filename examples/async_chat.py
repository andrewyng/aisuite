"""
Example of using the async client to interact with multiple LLMs concurrently.
"""
import asyncio
import aisuite as ai
from aisuite.async_client import AsyncClient

async def main():
    # Initialize the async client
    client = AsyncClient()

    # Define models and messages
    models = ["openai:gpt-4", "anthropic:claude-3-sonnet-20240229"]
    messages = [
        {"role": "system", "content": "Respond in Pirate English."},
        {"role": "user", "content": "Tell me a joke."},
    ]

    # Create tasks for concurrent execution
    tasks = [
        client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.75
        )
        for model in models
    ]

    # Execute all tasks concurrently
    responses = await asyncio.gather(*tasks)

    # Print responses
    for model, response in zip(models, responses):
        print(f"\nResponse from {model}:")
        print(response.choices[0].message.content)

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
