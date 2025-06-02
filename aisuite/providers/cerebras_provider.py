import os
import cerebras.cloud.sdk as cerebras
from aisuite.provider import Provider, LLMError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter


class CerebrasMessageConverter(OpenAICompliantMessageConverter):
    """
    Cerebras-specific message converter if needed.
    """

    pass


class CerebrasProvider(Provider):
    def __init__(self, **config):
        self.client = cerebras.Cerebras(**config)
        self.transformer = CerebrasMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Cerebras chat completions endpoint using the official client.
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,  # Pass any additional arguments to the Cerebras API.
            )
            return self.transformer.convert_response(response.model_dump())

        # Re-raise Cerebras API-specific exceptions.
        except cerebras.cloud.sdk.PermissionDeniedError as e:
            raise
        except cerebras.cloud.sdk.AuthenticationError as e:
            raise
        except cerebras.cloud.sdk.RateLimitError as e:
            raise

        # Wrap all other exceptions in LLMError.
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
