"""
Base class for async provider implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncGenerator
import aiohttp
import asyncio

from .chat_completion_response import ChatCompletionResponse

class AsyncProvider(ABC):
    """Base class for async provider implementations."""

    def __init__(self, **kwargs):
        """Initialize async provider."""
        self._session = None
        self.api_key = kwargs.get("api_key")

    @property
    async def session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @abstractmethod
    async def async_chat_completions_create(
        self, model: str, messages: List[Dict], **kwargs
    ) -> ChatCompletionResponse:
        """Create an async chat completion.

        Args:
            model: The model to use for completion
            messages: The messages to complete
            **kwargs: Additional arguments for the completion

        Returns:
            ChatCompletionResponse: The completion response
        """
        raise NotImplementedError

    @abstractmethod
    async def async_chat_completions_create_stream(
        self, model: str, messages: List[Dict], **kwargs
    ) -> AsyncGenerator[ChatCompletionResponse, None]:
        """Create a streaming async chat completion.

        Args:
            model: The model to use for completion
            messages: The messages to complete
            **kwargs: Additional arguments for the completion

        Yields:
            ChatCompletionResponse: Stream of completion responses
        """
        raise NotImplementedError

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
