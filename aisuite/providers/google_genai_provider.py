import os
from google import genai
from google.genai import types
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class GoogleGenaiProvider(Provider):
    def __init__(self, **config):
        self.api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is missing. Please provide it in the config or set the GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=self.api_key)

    def chat_completions_create(self, model, messages, **kwargs):
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=[message["content"] for message in messages],
                **kwargs
            )
            return self.normalize_response(response)
        except Exception as e:
            raise LLMError(f"Error in chat_completions_create: {str(e)}")

    def generate_content(self, model, contents, **kwargs):
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                **kwargs
            )
            return self.normalize_response(response)
        except Exception as e:
            raise LLMError(f"Error in generate_content: {str(e)}")

    def list_models(self):
        try:
            response = self.client.models.list()
            return [model.name for model in response]
        except Exception as e:
            raise LLMError(f"Error in list_models: {str(e)}")

    def normalize_response(self, response):
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response.text
        return normalized_response
