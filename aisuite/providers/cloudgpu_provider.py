import os
from aisuite.provider import Provider
from openai import Client

BASE_URL = "https://cloudgpu.app/v1"


# The class name follows aisuite's loader convention: "cloudgpu".capitalize() + "Provider".
class CloudgpuProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the CloudGPU provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("CLOUDGPU_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "CloudGPU API key is missing. Please provide it in the config or set the CLOUDGPU_API_KEY environment variable. You can create an API key at https://cloudgpu.app"
            )

        config["base_url"] = BASE_URL
        # Pass the entire config to the OpenAI client constructor
        self.client = Client(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the CloudGPU API
        )
