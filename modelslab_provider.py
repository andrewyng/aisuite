"""
ModelsLab provider for aisuite - Andrew Ng's unified AI interface.

Provides access to ModelsLab's uncensored language models and TTS capabilities
through the aisuite unified interface framework.
"""

import os
import requests
import time
from typing import Dict, Any, Optional, Union, BinaryIO, AsyncGenerator
from aisuite.provider import Provider, LLMError, ASRError, Audio
from aisuite.providers.message_converter import OpenAICompliantMessageConverter
from aisuite.framework.message import (
    TranscriptionResult,
    Segment,
    Word,
    StreamingTranscriptionChunk,
)


class ModelsLabProvider(Provider):
    """ModelsLab provider implementation for aisuite."""
    
    def __init__(self, **config):
        """
        Initialize the ModelsLab provider with the given configuration.
        
        Args:
            **config: Configuration dictionary containing:
                - api_key: ModelsLab API key (or set MODELSLAB_API_KEY env var)
                - base_url: Base URL for ModelsLab API (optional)
                - timeout: Request timeout in seconds (default: 60)
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MODELSLAB_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "ModelsLab API key is missing. Please provide it in the config or set the MODELSLAB_API_KEY environment variable."
            )
        
        # Set up configuration
        self.api_key = config["api_key"]
        self.base_url = config.get("base_url", "https://modelslab.com/api/")
        self.timeout = config.get("timeout", 60)
        
        # Initialize HTTP session for efficient connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "aisuite-modelslab/1.0",
        })
        
        # Initialize message transformer for OpenAI compatibility
        self.transformer = OpenAICompliantMessageConverter()
        
        # Initialize audio functionality
        super().__init__()
        self.audio = ModelsLabAudio(self.api_key, self.base_url, self.session, self.timeout)
    
    def chat_completions_create(self, model, messages, **kwargs):
        """
        Create chat completion using ModelsLab's uncensored language models.
        
        Args:
            model: Model identifier (e.g., "modelslab:llama-3.1-8b-uncensored")
            messages: List of messages in OpenAI format
            **kwargs: Additional parameters (temperature, max_tokens, stream, etc.)
        
        Returns:
            OpenAI-compatible chat completion response
        """
        try:
            # Transform messages to OpenAI format
            transformed_messages = self.transformer.convert_request(messages)
            
            # Extract model name (remove provider prefix if present)
            model_name = model.replace("modelslab:", "") if ":" in model else model
            
            # Map common model aliases to ModelsLab model identifiers
            model_mapping = {
                "llama-3.1-8b-uncensored": "ModelsLab/Llama-3.1-8b-Uncensored-Dare",
                "llama-3.1-70b-uncensored": "ModelsLab/Llama-3.1-70b-Uncensored-Dare",
                "llama-3.1-8b": "ModelsLab/Llama-3.1-8b-Uncensored-Dare",
                "llama-3.1-70b": "ModelsLab/Llama-3.1-70b-Uncensored-Dare",
            }
            
            actual_model = model_mapping.get(model_name, model_name)
            
            # Prepare request data
            request_data = {
                "model": actual_model,
                "messages": transformed_messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 2048),
                "stream": kwargs.get("stream", False),
            }
            
            # Add optional parameters if provided
            if "top_p" in kwargs:
                request_data["top_p"] = kwargs["top_p"]
            if "frequency_penalty" in kwargs:
                request_data["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs:
                request_data["presence_penalty"] = kwargs["presence_penalty"]
            if "stop" in kwargs:
                request_data["stop"] = kwargs["stop"]
            if "functions" in kwargs:
                request_data["functions"] = kwargs["functions"]
            if "function_call" in kwargs:
                request_data["function_call"] = kwargs["function_call"]
            if "tools" in kwargs:
                request_data["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                request_data["tool_choice"] = kwargs["tool_choice"]
            
            # Make API request to uncensored chat endpoint (OpenAI-compatible)
            response = self._make_request("uncensored-chat/v1/chat/completions", request_data)
            
            return response
            
        except Exception as e:
            raise LLMError(f"ModelsLab API error: {e}")
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        # Add API key to request data
        data["key"] = self.api_key
        
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise LLMError("Request timeout - ModelsLab API did not respond in time")
        except requests.exceptions.RequestException as e:
            raise LLMError(f"Network error: {e}")
        except ValueError as e:
            raise LLMError(f"Invalid JSON response: {e}")


class ModelsLabAudio(Audio):
    """ModelsLab Audio functionality container."""
    
    def __init__(self, api_key: str, base_url: str, session: requests.Session, timeout: int):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.session = session
        self.timeout = timeout
        self.transcriptions = self.Transcriptions(api_key, base_url, session, timeout)
    
    class Transcriptions(Audio.Transcription):
        """ModelsLab Audio Transcriptions functionality."""
        
        def __init__(self, api_key: str, base_url: str, session: requests.Session, timeout: int):
            self.api_key = api_key
            self.base_url = base_url
            self.session = session
            self.timeout = timeout
        
        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> TranscriptionResult:
            """
            Create audio transcription using ModelsLab TTS API.
            
            Note: This is a placeholder implementation for future ModelsLab STT capabilities.
            Currently returns a basic transcription result.
            """
            try:
                # Placeholder implementation - ModelsLab STT not yet available
                # This would be implemented when ModelsLab adds speech-to-text capabilities
                
                # For now, return a basic result indicating the service is not available
                return TranscriptionResult(
                    text="ModelsLab Speech-to-Text not yet available. This is a placeholder for future STT capabilities.",
                    language="en",
                    confidence=None,
                    segments=[],
                )
                
            except Exception as e:
                raise ASRError(f"ModelsLab transcription error: {e}") from e
        
        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """
            Create streaming audio transcription.
            
            Note: Placeholder for future ModelsLab streaming STT capabilities.
            """
            # Placeholder implementation
            yield StreamingTranscriptionChunk(
                text="ModelsLab streaming transcription not yet available.",
                is_final=True,
                confidence=None,
            )
    
    def text_to_speech(self, text: str, voice_id: str = "default", **kwargs) -> Dict[str, Any]:
        """
        Generate speech using ModelsLab Text-to-Speech API.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice identifier (default, professional, narrative, conversational)
            **kwargs: Additional parameters (language, emotion, speed, etc.)
        
        Returns:
            Dictionary containing audio URL and metadata
        """
        try:
            # Prepare TTS request data
            request_data = {
                "text": text,
                "voice_id": voice_id,
                "language": kwargs.get("language", "en"),
                "emotion": kwargs.get("emotion", "neutral"),
                "speed": kwargs.get("speed", 1.0),
                "key": self.api_key,
            }
            
            # Make API request to TTS endpoint
            url = f"{self.base_url}v6/voice/text_to_speech"
            response = self.session.post(url, json=request_data, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            # Handle async processing if needed
            if result.get("status") == "processing":
                return self._poll_async_task(result["id"])
            
            return result
            
        except Exception as e:
            raise ASRError(f"ModelsLab TTS error: {e}")
    
    def _poll_async_task(self, task_id: str) -> Dict[str, Any]:
        """Poll async task until completion."""
        max_attempts = 30  # Maximum polling attempts
        attempt = 0
        
        while attempt < max_attempts:
            try:
                url = f"{self.base_url}v6/fetch/{task_id}"
                data = {"key": self.api_key}
                
                response = self.session.post(url, json=data, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "success":
                    return result
                elif result.get("status") == "failed":
                    raise ASRError(f"ModelsLab task failed: {result.get('message')}")
                
                # Wait before next poll
                time.sleep(2)
                attempt += 1
                
            except Exception as e:
                raise ASRError(f"ModelsLab polling error: {e}")
        
        raise ASRError("ModelsLab task polling timeout - task did not complete in time")


# Model configurations for common ModelsLab models
MODELSLAB_MODELS = {
    "modelslab:llama-3.1-8b-uncensored": {
        "name": "Llama 3.1 8B Uncensored",
        "description": "Uncensored Llama 3.1 8B model for creative and unrestricted content",
        "max_tokens": 32768,
        "supports_functions": True,
        "supports_streaming": True,
    },
    "modelslab:llama-3.1-70b-uncensored": {
        "name": "Llama 3.1 70B Uncensored", 
        "description": "Uncensored Llama 3.1 70B model for high-quality unrestricted content",
        "max_tokens": 32768,
        "supports_functions": True,
        "supports_streaming": True,
    },
}


def get_available_models():
    """Get list of available ModelsLab models."""
    return list(MODELSLAB_MODELS.keys())


def get_model_info(model: str) -> Dict[str, Any]:
    """Get information about a specific ModelsLab model."""
    return MODELSLAB_MODELS.get(model, {})