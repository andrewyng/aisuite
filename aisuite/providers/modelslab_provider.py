import os
import requests
from typing import Union, List, Dict, Any, Optional
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class ModelslabProvider(Provider):
    """
    ModelsLab provider for unified LLM access.
    
    ModelsLab provides access to 10,000+ AI models including LLMs, image generation,
    video creation, voice cloning, and more via a unified API.
    
    API Documentation: https://docs.modelslab.com
    """
    
    def __init__(self, **config):
        """
        Initialize the ModelsLab provider with the given configuration.
        
        Configuration options:
        - api_key: ModelsLab API key (required, or set MODELSLAB_API_KEY env var)
        - base_url: Custom API base URL (optional)
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("MODELSLAB_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "ModelsLab API key is missing. Please provide it in the config "
                "or set the MODELSLAB_API_KEY environment variable."
            )
        
        self.api_key = config["api_key"]
        self.base_url = config.get("base_url", "https://api.modelslab.com/v1")
        
        # Use OpenAI-compatible message converter
        self.transformer = OpenAICompliantMessageConverter()
        
        # Initialize base provider
        super().__init__()

    def chat_completions_create(self, model: str, messages: List[Dict], **kwargs):
        """
        Create a chat completion using ModelsLab API.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'claude-3-opus', etc.)
            messages: List of message dictionaries
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            OpenAI-compatible chat completion response
        """
        try:
            transformed_messages = self.transformer.convert_request(messages)
            
            # Build request payload
            payload = {
                "model": model,
                "messages": transformed_messages,
            }
            
            # Add optional parameters
            if "temperature" in kwargs:
                payload["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                payload["max_tokens"] = kwargs["max_tokens"]
            if "top_p" in kwargs:
                payload["top_p"] = kwargs["top_p"]
            if "stream" in kwargs:
                payload["stream"] = kwargs["stream"]
            if "stop" in kwargs:
                payload["stop"] = kwargs["stop"]
            if "presence_penalty" in kwargs:
                payload["presence_penalty"] = kwargs["presence_penalty"]
            if "frequency_penalty" in kwargs:
                payload["frequency_penalty"] = kwargs["frequency_penalty"]
            if "n" in kwargs:
                payload["n"] = kwargs["n"]
            if "response_format" in kwargs:
                payload["response_format"] = kwargs["response_format"]
            if "seed" in kwargs:
                payload["seed"] = kwargs["seed"]
            if "tools" in kwargs:
                payload["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                payload["tool_choice"] = kwargs["tool_choice"]
            if "parallel_tool_calls" in kwargs:
                payload["parallel_tool_calls"] = kwargs["parallel_tool_calls"]
            
            # Make API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Use chat/completions endpoint
            url = f"{self.base_url}/chat/completions"
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code != 200:
                error_msg = f"ModelsLab API error: {response.status_code} - {response.text}"
                raise LLMError(error_msg)
            
            return response.json()
            
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"ModelsLab API error: {str(e)}")