import os
import groq
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter

# Implementation of Groq provider.
# Groq's message format is same as OpenAI's.
# Tool calling specification is also exactly the same as OpenAI's.
# Links:
# https://console.groq.com/docs/tool-use
# Groq supports tool calling for the following models, as of 16th Nov 2024:
#   llama3-groq-70b-8192-tool-use-preview
#   llama3-groq-8b-8192-tool-use-preview
#   llama-3.1-70b-versatile
#   llama-3.1-8b-instant
#   llama3-70b-8192
#   llama3-8b-8192
#   mixtral-8x7b-32768 (parallel tool use not supported)
#   gemma-7b-it (parallel tool use not supported)
#   gemma2-9b-it (parallel tool use not supported)


class GroqMessageConverter(OpenAICompliantMessageConverter):
    """
    Groq-specific message converter if needed
    """

    pass


class GroqProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Groq provider with the given configuration.
        Pass the entire configuration dictionary to the Groq client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        self.api_key = config.get("api_key", os.getenv("GROQ_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "Groq API key is missing. Please provide it in the config or set the GROQ_API_KEY environment variable."
            )
        config["api_key"] = self.api_key
        self.client = groq.Groq(**config)
        self.transformer = GroqMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Groq chat completions endpoint using the official client.
        """
        try:
            # Transform messages using converter
            transformed_messages = self.transformer.convert_request(messages)

            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the Groq API
            )
            return self.transformer.convert_response(response.model_dump())
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
