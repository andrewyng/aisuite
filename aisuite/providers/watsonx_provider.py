from aisuite.provider import Provider
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

DEFAULT_TEMPERATURE = 0.7


class WatsonxProvider(Provider):
    def __init__(self, **config):
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
        model = ModelInference(
            model_id=model,
            params={
                GenParams.TEMPERATURE: kwargs.get("temperature", DEFAULT_TEMPERATURE),
            },
            credentials=Credentials(api_key=self.api_key, url=self.service_url),
            project_id=self.project_id,
        )

        return model.chat(prompt=messages, **kwargs)
