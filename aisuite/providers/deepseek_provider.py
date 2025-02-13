import openai
import os
from aisuite.provider import Provider, LLMError


class DeepseekProvider(Provider):
    """A provider class for interfacing with the DeepSeek API, built on top of the OpenAI client.

    This class initializes the DeepSeek provider using a configuration dictionary, ensuring that
    an API key is provided either through the configuration or as an environment variable. It 
    facilitates interaction with the DeepSeek API, specifically for creating chat completions.

    Attributes:
        client (openai.OpenAI): An instance of the OpenAI client initialized with the provided configuration.

    Methods:
        chat_completions_create(model, messages, **kwargs): Creates chat completions using the specified model and messages."""
    def __init__(self, **config):
        """
        Initialize the DeepSeek provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("DEEPSEEK_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "DeepSeek API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )
        config["base_url"] = "https://api.deepseek.com"

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID. Except for OPEN_AI_BASE_URL which has to be the deepseek url

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)

    def chat_completions_create(self, model, messages, **kwargs):
        """Create a chat completion using the specified model and messages.

    This method interfaces with the OpenAI API to generate chat completions based on the provided model and messages. Additional parameters can be passed through kwargs to customize the API request.

    Args:
        model (str): The model name to use for generating chat completions.
        messages (list): A list of message dictionaries representing the conversation history.
        **kwargs: Optional additional parameters to customize the API request.

    Returns:
        dict: The response from the OpenAI API containing the chat completion."""
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )
