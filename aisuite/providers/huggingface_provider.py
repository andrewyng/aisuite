import os
import json
from huggingface_hub import InferenceClient
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message


class HuggingfaceProvider(Provider):
    """
    HuggingFace Provider using the official InferenceClient.
    This provider supports calls to HF serverless Inference Endpoints
    which use Text Generation Inference (TGI) as the backend.
    TGI is OpenAI protocol compliant.
    https://huggingface.co/inference-endpoints/
    """

    def __init__(self, **config):
        """
        Initialize the provider with the given configuration.
        The token is fetched from the config or environment variables.
        """
        # Ensure API key is provided either in config or via environment variable
        self.token = config.get("token") or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "Hugging Face token is missing. Please provide it in the config or set the HF_TOKEN environment variable."
            )

        # Initialize the InferenceClient with the specified model and timeout if provided
        self.model = config.get("model")
        self.timeout = config.get("timeout", 30)
        self.client = InferenceClient(
            token=self.token, model=self.model, timeout=self.timeout
        )

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the Inference API endpoint using InferenceClient.
        """
        # Validate and transform messages
        transformed_messages = []
        for message in messages:
            if isinstance(message, Message):
                transformed_message = self.transform_from_message(message)
            elif isinstance(message, dict):
                transformed_message = message
            else:
                raise ValueError(f"Invalid message format: {message}")

            # Ensure 'content' is a non-empty string
            if (
                "content" not in transformed_message
                or transformed_message["content"] is None
            ):
                transformed_message["content"] = ""

            transformed_messages.append(transformed_message)

        try:
            # Prepare the payload
            payload = {
                "messages": transformed_messages,
                **kwargs,  # Include other parameters like temperature, max_tokens, etc.
            }

            # Make the API call using the client
            response = self.client.chat_completion(model=model, **payload)

            return self._normalize_response(response)

        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    def transform_from_message(self, message: Message):
        """Transform framework Message to a format that HuggingFace understands."""
        # Ensure content is a string
        content = message.content if message.content is not None else ""

        # Transform the message
        transformed_message = {
            "role": message.role,
            "content": content,
        }

        # Include tool_calls if present
        if message.tool_calls:
            transformed_message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "type": tool_call.type,
                }
                for tool_call in message.tool_calls
            ]

        return transformed_message

    def transform_to_message(self, message_dict: dict):
        """Transform HuggingFace message (dict) to a format that the framework Message understands."""
        # Ensure required fields are present
        message_dict.setdefault("content", "")  # Set empty string if content is missing
        message_dict.setdefault("refusal", None)  # Set None if refusal is missing
        message_dict.setdefault("tool_calls", None)  # Set None if tool_calls is missing

        # Handle tool calls if present and not None
        if message_dict.get("tool_calls"):
            for tool_call in message_dict["tool_calls"]:
                if "function" in tool_call:
                    # Ensure function arguments are stringified
                    if isinstance(tool_call["function"].get("arguments"), dict):
                        tool_call["function"]["arguments"] = json.dumps(
                            tool_call["function"]["arguments"]
                        )

        return Message(**message_dict)

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        message_data = response_data["choices"][0]["message"]
        normalized_response.choices[0].message = self.transform_to_message(message_data)
        return normalized_response
