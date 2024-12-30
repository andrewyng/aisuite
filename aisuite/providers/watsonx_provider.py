from aisuite.provider import Provider
import os
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from aisuite.framework import ChatCompletionResponse


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
