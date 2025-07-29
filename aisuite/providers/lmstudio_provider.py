import lmstudio

from aisuite.framework import ChatCompletionResponse
from aisuite.provider import LLMError, Provider


class LmstudioProvider(Provider):
    def __init__(self, **config):
        self.client = lmstudio.Client(**config)
        self.model: lmstudio.LLM | None = None

    def _chat(self, model: str, messages, **kwargs) -> lmstudio.PredictionResult[str]:
        """
        Makes a request to the lmstudio chat completions endpoint using the official client
        """
        # get an handle of the specified model (load it if necessary)
        model = self.client.llm.model(model)
        # send the request to the model
        result = model.respond({"messages": messages}, **kwargs)

        return result

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the lmstudio chat completions endpoint using the official client
        and convert output to be conform to openAI model response
        """
        try:
            # --- send request to lmstudio endpoint
            result = self._chat(model=model, messages=messages, **kwargs)

            # --- Return the normalized response
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
