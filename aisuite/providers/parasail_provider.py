"""Parasail provider for its OpenAI-compatible Chat Completions API."""

import os
from typing import Any

import openai

from aisuite.provider import LLMError, Provider
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


DEFAULT_BASE_URL = "https://api.parasail.io/v1"


class ParasailProvider(Provider):
    """Call Parasail through its OpenAI-compatible API."""

    def __init__(self, **config: Any):
        super().__init__()

        config = dict(config)
        api_key = config.get("api_key") or os.getenv("PARASAIL_API_KEY")
        if not api_key:
            raise ValueError(
                "Parasail API key is missing. Provide it in the provider "
                "configuration or set PARASAIL_API_KEY."
            )

        config["api_key"] = api_key
        config.setdefault("base_url", DEFAULT_BASE_URL)

        self.client = openai.OpenAI(**config)
        self.aclient = openai.AsyncOpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model: str, messages: list, **kwargs: Any):
        """Create a synchronous Parasail chat completion."""
        try:
            return self.client.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"Parasail chat completion failed: {exc}") from exc

    async def achat_completions_create(self, model: str, messages: list, **kwargs: Any):
        """Create an asynchronous Parasail chat completion."""
        try:
            return await self.aclient.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"Parasail chat completion failed: {exc}") from exc

    def chat_completions_create_stream(self, model: str, messages: list, **kwargs: Any):
        """Yield synchronous Parasail chat completion chunks."""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                stream=True,
                **kwargs,
            )
            yield from stream
        except Exception as exc:
            raise LLMError(f"Parasail chat completion stream failed: {exc}") from exc

    async def achat_completions_create_stream(
        self, model: str, messages: list, **kwargs: Any
    ):
        """Yield asynchronous Parasail chat completion chunks."""
        try:
            stream = await self.aclient.chat.completions.create(
                model=model,
                messages=self.transformer.convert_request(messages),
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                yield chunk
        except Exception as exc:
            raise LLMError(f"Parasail chat completion stream failed: {exc}") from exc
