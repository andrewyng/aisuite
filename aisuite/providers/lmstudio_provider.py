import lmstudio

from aisuite.framework import ChatCompletionResponse
from aisuite.provider import LLMError, Provider


class LmstudioProvider(Provider):
    def __init__(self, **config):
        self.client = lmstudio.Client(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Cerebras chat completions endpoint using the official client.
        """
        try:
            model = self.client.llm.model(model)
            result = model.respond({"messages": messages})

            # Return the normalized response
            normalized_response = self._normalize_response(result)
            return normalized_response

        # Wrap all other exceptions in LLMError.
        except Exception as e:
            raise LLMError("An error occurred.") from e

    def _normalize_response(self, response_data: lmstudio.PredictionResult):
        """
        Normalize the lmstudio response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data.content
        return normalized_response
