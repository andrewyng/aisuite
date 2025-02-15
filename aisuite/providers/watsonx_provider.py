from aisuite.provider import Provider
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from aisuite.framework import ChatCompletionResponse


class WatsonxProvider(Provider):
    """A provider for interacting with the WatsonX AI service.

    This class facilitates communication with WatsonX by setting up
    the required credentials and service URL. It provides methods to
    create chat completions using specified models.

    Attributes:
        service_url (str): The URL of the WatsonX service.
        api_key (str): The API key for authenticating with WatsonX.
        project_id (str): The project ID associated with WatsonX.

    Raises:
        EnvironmentError: If any required configuration is missing."""
    def __init__(self, **config):
        """Initializes a WatsonX client with the necessary configuration.

    This constructor sets up a client for the WatsonX service by 
    retrieving configuration details from the provided `config` dictionary 
    or from environment variables. The required configuration parameters 
    are `service_url`, `api_key`, and `project_id`. If any of these 
    parameters are missing, an EnvironmentError is raised.

    Args:
        **config: Arbitrary keyword arguments that may include:
            - service_url (str): The URL of the WatsonX service.
            - api_key (str): The API key for authentication.
            - project_id (str): The project ID to be used.

    Raises:
        EnvironmentError: If any of `service_url`, `api_key`, or `project_id` 
        are not provided either in `config` or as environment variables 
        `WATSONX_SERVICE_URL`, `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`."""
        self.service_url = config.get("service_url") or os.getenv("WATSONX_SERVICE_URL")
        self.api_key = config.get("api_key") or os.getenv("WATSONX_API_KEY")
        self.project_id = config.get("project_id") or os.getenv("WATSONX_PROJECT_ID")

        if not self.service_url or not self.api_key or not self.project_id:
            raise EnvironmentError(
                "Missing one or more required WatsonX environment variables: "
                "WATSONX_SERVICE_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID. "
                "Please refer to the setup guide: /guides/watsonx.md."
            )

    def chat_completions_create(self, model, messages, **kwargs):
        """Generates chat completions using a specified model.

    This method sends a series of messages to a designated model and returns the model's response after normalization.

    Args:
        model (str): The identifier of the model to be used for inference.
        messages (list): A list of message objects to be sent to the model.
        **kwargs: Additional parameters to customize the model's behavior.

    Returns:
        dict: A normalized response from the model."""
        model = ModelInference(
            model_id=model,
            credentials=Credentials(
                api_key=self.api_key,
                url=self.service_url,
            ),
            project_id=self.project_id,
        )

        res = model.chat(messages=messages, params=kwargs)
        return self.normalize_response(res)

    def normalize_response(self, response):
        openai_response = ChatCompletionResponse()
        openai_response.choices[0].message.content = response["choices"][0]["message"][
            "content"
        ]
        return openai_response
