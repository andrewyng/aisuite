import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse


class WatsonxProvider(Provider):
    """
    WatsonX AI Provider using httpx for direct API calls.
    """

    _CHAT_COMPLETION_ENDPOINT = (
        f"ml/v1/text/chat?version={os.getenv('WATSONX_API_VERSION', '2023-10-25')}"
    )

    def __init__(self, **config):
        """
        Initialize the WatsonX provider with the given configuration.
        The Access Token, Project ID and Cluster URL are fetched from the config or environment variables.
        """
        self.access_token = config.get(
            "access_token", os.getenv("IBM_IAM_ACCESS_TOKEN")
        )
        self.project_id = config.get("project_id", os.getenv("WATSONX_PROJECT_ID"))
        self.cluster_url = config.get("cluster_url", os.getenv("WATSONX_CLUSTER_URL"))

        if not self.access_token:
            raise ValueError(
                "WatsonX ACCESS TOKEN is missing. Please provide it in the config or set the WATSONX_ACCESS_TOKEN environment variable."
            )

        if not self.project_id:
            raise ValueError(
                "WatsonX Project ID is missing. Please provide it in the config or set the WATSONX_PROJECT_ID environment variable."
            )

        if not self.cluster_url:
            raise ValueError(
                "WatsonX Cluster URL is missing. Please provide it in the config or set the WATSONX_CLUSTER_URL environment variable."
            )

        self._url = f"{self.cluster_url}/{self._CHAT_COMPLETION_ENDPOINT}"

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the WatsonX AI chat completions endpoint using httpx.
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        data = {
            "model_id": model,
            "project_id": self.project_id,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }

        try:
            # Make the request to WatsonX AI endpoint.
            response = httpx.post(
                self._url, json=data, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"WatsonX AI request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        print(response.json())
        return self._normalize_response(response.json())

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["choices"][0][
            "message"
        ]["content"]
        return normalized_response
