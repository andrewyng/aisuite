"""
Async OpenAI provider implementation.
"""
import os
from typing import Dict, List, Optional, AsyncGenerator
import openai
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

from ..framework.async_provider import AsyncProvider
from ..framework.chat_completion_response import ChatCompletionResponse

class OpenAIError(Exception):
    """Base class for OpenAI-related errors."""
    pass

class AsyncOpenAIProvider(AsyncProvider):
    """Async OpenAI provider implementation."""

    def __init__(self, **kwargs):
        """Initialize async OpenAI provider."""
        super().__init__(**kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=60.0,  # Increased timeout
            max_retries=2  # Add retries
        )

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
        
        Raises:
            OpenAIError: If there's an error with the OpenAI API
            ValueError: If the model or messages are invalid
        """
        try:
            # Remove provider prefix if present (e.g., "openai:gpt-4" -> "gpt-4")
            if ":" in model:
                model = model.split(":")[-1]

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return ChatCompletionResponse.from_openai_response(response)
            
        except APIConnectionError as e:
            raise OpenAIError(
                "Failed to connect to OpenAI API. "
                "Please check your internet connection and try again."
            ) from e
        except RateLimitError as e:
            raise OpenAIError(
                "OpenAI API rate limit exceeded. "
                "Please wait a moment before trying again."
            ) from e
        except APIError as e:
            raise OpenAIError(
                "OpenAI API error. "
                "Please check your API key and model name."
            ) from e
        except Exception as e:
            raise OpenAIError(f"Unexpected error during OpenAI request: {str(e)}") from e

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
        
        Raises:
            OpenAIError: If there's an error with the OpenAI API
            ValueError: If the model or messages are invalid
        """
        try:
            # Remove provider prefix if present
            if ":" in model:
                model = model.split(":")[-1]

            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                yield ChatCompletionResponse.from_openai_response(chunk)
                
        except APIConnectionError as e:
            raise OpenAIError(
                "Failed to connect to OpenAI API. "
                "Please check your internet connection and try again."
            ) from e
        except RateLimitError as e:
            raise OpenAIError(
                "OpenAI API rate limit exceeded. "
                "Please wait a moment before trying again."
            ) from e
        except APIError as e:
            raise OpenAIError(
                "OpenAI API error. "
                "Please check your API key and model name."
            ) from e
        except Exception as e:
            raise OpenAIError(f"Unexpected error during OpenAI streaming request: {str(e)}") from e

    async def close(self):
        """Close the client and any open sessions."""
        await super().close()
        # The OpenAI client handles its own cleanup
