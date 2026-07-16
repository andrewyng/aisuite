import os
import json
import time
from typing import Union, BinaryIO
import requests
from huggingface_hub import InferenceClient
from aisuite.provider import Provider, LLMError, ASRError, Audio
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, TranscriptionResult, Word


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
        self.token = (
            config.get("token")
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
        )
        if not self.token:
            raise ValueError(
                "Hugging Face token is missing. Please provide it in the config or set the HF_TOKEN or HUGGINGFACE_API_KEY environment variable."
            )

        # Initialize the InferenceClient with the specified model and timeout if provided
        self.model = config.get("model")
        self.timeout = config.get("timeout", 30)
        self.client = InferenceClient(
            token=self.token, model=self.model, timeout=self.timeout
        )

        # Initialize audio functionality
        super().__init__()
        self.audio = HuggingfaceAudio(self.token, self.timeout)

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


# Audio Classes
class HuggingfaceAudio(Audio):
    """Hugging Face Audio functionality container."""

    def __init__(self, token, timeout=120):
        super().__init__()
        self.transcriptions = self.Transcriptions(token, timeout)

    class Transcriptions(Audio.Transcription):
        """Hugging Face Audio Transcriptions functionality."""

        def __init__(self, token, timeout=120):
            self.token = token
            self.timeout = timeout

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> TranscriptionResult:
            """
            Create audio transcription using Hugging Face Inference API.

            All parameters are already validated and mapped by the Client layer.
            This makes an HTTP POST request to the Hugging Face Inference API.

            Note: Whisper-based models have a 30-second processing window.
            For longer audio, users should deploy custom Inference Endpoints.
            """
            try:
                # Extract model ID from format "huggingface:model-id"
                model_id = model.split(":", 1)[1] if ":" in model else model

                # Prepare API endpoint
                url = f"https://api-inference.huggingface.co/models/{model_id}"

                # Prepare audio data
                if isinstance(file, str):
                    with open(file, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    content_type = self._detect_content_type(file)
                else:
                    audio_bytes = file.read()
                    # Default to wav for file-like objects
                    content_type = "audio/wav"

                # Prepare headers
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": content_type,
                }

                # First attempt without wait_for_model
                try:
                    response = requests.post(
                        url,
                        headers=headers,
                        data=audio_bytes,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    # If 503 (model loading), retry with x-wait-for-model header
                    if e.response.status_code == 503:
                        headers["x-wait-for-model"] = "true"
                        response = requests.post(
                            url,
                            headers=headers,
                            data=audio_bytes,
                            timeout=self.timeout,
                        )
                        response.raise_for_status()
                    else:
                        raise

                # Parse response
                response_data = response.json()
                return self._parse_huggingface_response(response_data, model_id)

            except requests.exceptions.RequestException as e:
                raise ASRError(f"Hugging Face transcription error: {e}") from e
            except Exception as e:
                raise ASRError(f"Hugging Face transcription error: {e}") from e

        def _detect_content_type(self, file_path: str) -> str:
            """Detect audio content type from file extension."""
            if file_path.lower().endswith(".wav"):
                return "audio/wav"
            elif file_path.lower().endswith(".mp3"):
                return "audio/mpeg"  # HF API requires audio/mpeg for MP3
            elif file_path.lower().endswith(".flac"):
                return "audio/flac"
            else:
                # Default to wav if unknown
                return "audio/wav"

        def _parse_huggingface_response(
            self, response_data, model_id: str
        ) -> TranscriptionResult:
            """
            Parse Hugging Face API response into TranscriptionResult.

            Response format can vary:
            - Standard: {"text": "...", "chunks": [...]}
            - Text only: {"text": "..."}
            - Some models may use different keys
            """
            try:
                # Extract text
                if isinstance(response_data, dict):
                    text = response_data.get("text", "")
                elif isinstance(response_data, str):
                    # Some models return plain string
                    text = response_data
                else:
                    text = str(response_data)

                # Extract words from chunks if available
                words = None
                if isinstance(response_data, dict) and "chunks" in response_data:
                    chunks = response_data["chunks"]
                    if chunks:
                        words = []
                        for chunk in chunks:
                            if isinstance(chunk, dict):
                                word_text = chunk.get("text", "")
                                timestamp = chunk.get("timestamp")

                                # timestamp can be [start, end] or (start, end)
                                start, end = None, None
                                if timestamp and len(timestamp) >= 2:
                                    start, end = timestamp[0], timestamp[1]

                                words.append(
                                    Word(
                                        word=word_text,
                                        start=start,
                                        end=end,
                                        confidence=None,  # HF doesn't provide confidence
                                    )
                                )

                return TranscriptionResult(
                    text=text,
                    language=None,  # HF API doesn't return language
                    confidence=None,  # HF API doesn't return confidence
                    words=words,
                    task="transcribe",
                )

            except (KeyError, TypeError, IndexError) as e:
                raise ASRError(f"Error parsing Hugging Face response: {e}")
