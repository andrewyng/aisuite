"""The interface to Google's Vertex AI and Gemini via modern GenAI SDK."""

import os
import json
from typing import List, Dict, Any, Optional, Union, BinaryIO, AsyncGenerator

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

import pprint

from aisuite.framework import ChatCompletionResponse, Message
from aisuite.framework.message import (
    TranscriptionResult,
    Word,
    Segment,
    Alternative,
    StreamingTranscriptionChunk,
)
from aisuite.provider import Provider, ASRError, Audio

DEFAULT_TEMPERATURE = 0.7
ENABLE_DEBUG_MESSAGES = False

# Links.
# https://codelabs.developers.google.com/codelabs/gemini-function-calling#6
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#chat-samples


class GoogleMessageConverter:
    @staticmethod
    def convert_user_role_message(message: Dict[str, Any]) -> types.Content:
        """Convert user or system messages to Google GenAI format."""
        parts = [types.Part.from_text(text=message.get("content", ""))]
        return types.Content(role="user", parts=parts)

    @staticmethod
    def convert_assistant_role_message(message: Dict[str, Any]) -> types.Content:
        """Convert assistant messages to Google GenAI format."""
        if "tool_calls" in message and message["tool_calls"]:
            # Handle function calls
            tool_call = message["tool_calls"][
                0
            ]  # Assuming single function call for now
            function_call = tool_call["function"]

            # Create a Part from the function call
            parts = [
                types.Part.from_function_call(
                    name=function_call["name"],
                    args=json.loads(function_call["arguments"]),
                )
            ]
        else:
            # Handle regular text messages
            parts = [types.Part.from_text(text=message.get("content", ""))]

        return types.Content(role="model", parts=parts)

    @staticmethod
    def convert_tool_role_message(message: Dict[str, Any]) -> types.Content:
        """Convert tool messages to Google GenAI format."""
        if "content" not in message:
            raise ValueError("Tool result message must have a content field")

        try:
            content_json = json.loads(message["content"])
            part = types.Part.from_function_response(
                name=message["name"], response=content_json
            )
            return types.Content(role="user", parts=[part])
        except json.JSONDecodeError:
            raise ValueError("Tool result message must be valid JSON")

    @staticmethod
    def convert_request(messages: List[Dict[str, Any]]) -> List[types.Content]:
        """Convert messages to Google GenAI format."""
        # Convert all messages to dicts if they're Message objects
        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message
            for message in messages
        ]

        formatted_messages = []
        for message in messages:
            if message["role"] == "tool":
                vertex_message = GoogleMessageConverter.convert_tool_role_message(
                    message
                )
                if vertex_message:
                    formatted_messages.append(vertex_message)
            elif message["role"] == "assistant":
                formatted_messages.append(
                    GoogleMessageConverter.convert_assistant_role_message(message)
                )
            else:  # user or system role
                formatted_messages.append(
                    GoogleMessageConverter.convert_user_role_message(message)
                )

        return formatted_messages

    @staticmethod
    def convert_response(response) -> ChatCompletionResponse:
        """Normalize the response from Google GenAI to match OpenAI's response format."""
        openai_response = ChatCompletionResponse()

        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the response")
            pprint.pprint(response)

        if not response.candidates:
            return openai_response

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return openai_response

        first_part = candidate.content.parts[0]

        # Check if the response contains function calls
        if hasattr(first_part, "function_call") and first_part.function_call:
            function_call = first_part.function_call

            args_dict = {}
            if hasattr(function_call, "args") and function_call.args:
                for key, value in function_call.args.items():
                    args_dict[key] = value

            if ENABLE_DEBUG_MESSAGES:
                print("Dumping the args_dict")
                pprint.pprint(args_dict)

            openai_response.choices[0].message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"call_{hash(function_call.name)}",  # Generate a unique ID
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(args_dict),
                        },
                    }
                ],
                "refusal": None,
            }
            openai_response.choices[0].message = Message(
                **openai_response.choices[0].message
            )
            openai_response.choices[0].finish_reason = "tool_calls"
        else:
            # Handle regular text response
            openai_response.choices[0].message.content = first_part.text
            openai_response.choices[0].finish_reason = "stop"

        return openai_response


class GoogleProvider(Provider):
    """Implements the ProviderInterface for interacting with Google's GenAI."""

    def __init__(self, **config):
        """Set up the Google GenAI client."""
        super().__init__()

        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
        self.location = config.get("region") or os.getenv("GOOGLE_REGION")

        if not self.api_key and not self.project_id:
            raise EnvironmentError(
                "Missing required Google environment variable: "
                "GOOGLE_API_KEY (for Gemini) or GOOGLE_PROJECT_ID (for Vertex AI). "
                "Please refer to the setup guide: /guides/google.md."
            )

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = genai.Client(
                vertexai=True, project=self.project_id, location=self.location
            )

        self.transformer = GoogleMessageConverter()

        # Initialize Speech client lazily
        self._speech_client = None

        # Initialize audio functionality
        self.audio = GoogleAudio(self)

    def chat_completions_create(self, model, messages, **kwargs):
        """Request chat completions from the Google GenAI API.

        Args:
        ----
            model (str): Identifies the specific provider/model to use.
            messages (list of dict): A list of message objects in chat history.
            kwargs (dict): Optional arguments for the Google GenAI API.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.

        """

        # Set the temperature if provided, otherwise use the default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)

        # Convert messages to GenAI format
        message_history = self.transformer.convert_request(messages)

        # Handle tools if provided
        tools = None
        if "tools" in kwargs:
            tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=tool["function"]["name"],
                            description=tool["function"].get("description", ""),
                            parameters=tool["function"].get("parameters", {}),
                        )
                        for tool in kwargs["tools"]
                    ]
                )
            ]

        # Use new model generate content API
        config = types.GenerateContentConfig(temperature=temperature, tools=tools)

        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the message_history")
            pprint.pprint(message_history)

        # Start chat and get response
        response = self.client.models.generate_content(
            model=model,
            contents=message_history,
            config=config,
        )

        # Convert and return the response
        return self.transformer.convert_response(response)

    @property
    def speech_client(self):
        """Lazy initialization of Google Cloud Speech client."""
        if self._speech_client is None:
            try:
                from google.cloud import speech

                self._speech_client = speech.SpeechClient()
            except ImportError:
                raise ImportError(
                    "google-cloud-speech is required for ASR functionality. "
                    "Install it with: pip install google-cloud-speech"
                )
        return self._speech_client


# Audio Classes
class GoogleAudio(Audio):
    """Google Audio functionality container."""

    def __init__(self, provider):
        super().__init__()
        self.provider = provider
        self.transcriptions = self.Transcriptions(provider)

    class Transcriptions(Audio.Transcription):
        """Google Audio Transcriptions functionality."""

        def __init__(self, provider):
            self.provider = provider

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> TranscriptionResult:
            """
            Create audio transcription using Google Cloud Speech-to-Text API.

            All parameters are already validated and mapped by the Client layer.
            This is a simple pass-through to the Google API.
            """
            try:
                from google.cloud import speech

                # Set defaults
                kwargs["model"] = model if model != "default" else "latest_long"
                kwargs.setdefault("sample_rate_hertz", 16000)
                kwargs.setdefault("enable_automatic_punctuation", True)

                audio_data = self._read_audio_data(file)
                audio = speech.RecognitionAudio(content=audio_data)
                config = self._build_recognition_config(kwargs, speech, file)

                response = self.provider.speech_client.recognize(
                    config=config, audio=audio
                )
                return self._parse_google_response(response)

            except ImportError:
                raise ASRError(
                    "google-cloud-speech is required for ASR functionality. "
                    "Install it with: pip install google-cloud-speech"
                )
            except Exception as e:
                raise ASRError(f"Google Speech-to-Text error: {e}") from e

        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """
            Create streaming audio transcription using Google Cloud Speech-to-Text API.

            All parameters are already validated and mapped by the Client layer.
            This implementation handles streaming with Google's API.
            """
            try:
                from google.cloud import speech

                # Set defaults
                kwargs["model"] = model if model != "default" else "latest_long"
                kwargs.setdefault("sample_rate_hertz", 16000)
                kwargs.setdefault("enable_automatic_punctuation", True)

                config = self._build_recognition_config(kwargs, speech, file)
                streaming_config = speech.StreamingRecognitionConfig(
                    config=config, interim_results=True, single_utterance=False
                )

                audio_data = self._read_audio_data(file)
                request_generator = self._create_streaming_requests(
                    speech, streaming_config, audio_data
                )

                responses = self.provider.speech_client.streaming_recognize(
                    config=streaming_config, requests=request_generator
                )

                for response in responses:
                    for result in response.results:
                        if result.alternatives:
                            alternative = result.alternatives[0]
                            yield StreamingTranscriptionChunk(
                                text=alternative.transcript,
                                is_final=result.is_final,
                                confidence=getattr(alternative, "confidence", None),
                            )

            except ImportError:
                raise ASRError(
                    "google-cloud-speech is required for ASR functionality. "
                    "Install it with: pip install google-cloud-speech"
                )
            except Exception as e:
                raise ASRError(f"Google Speech-to-Text streaming error: {e}") from e

        def _read_audio_data(self, file: Union[str, BinaryIO]) -> bytes:
            """Read audio data from file or file-like object."""
            if isinstance(file, str):
                with open(file, "rb") as audio_file:
                    return audio_file.read()
            else:
                return file.read()

        def _detect_audio_encoding(self, file: Union[str, BinaryIO], speech):
            """Detect audio encoding based on file extension or content."""
            if isinstance(file, str):
                # File path - detect by extension
                file_lower = file.lower()
                if file_lower.endswith(".mp3"):
                    return speech.RecognitionConfig.AudioEncoding.MP3
                elif file_lower.endswith(".flac"):
                    return speech.RecognitionConfig.AudioEncoding.FLAC
                elif file_lower.endswith(".wav"):
                    return speech.RecognitionConfig.AudioEncoding.LINEAR16
                elif file_lower.endswith(".ogg"):
                    return speech.RecognitionConfig.AudioEncoding.OGG_OPUS
                elif file_lower.endswith(".webm"):
                    return speech.RecognitionConfig.AudioEncoding.WEBM_OPUS

            # Default to LINEAR16 for unknown formats
            return speech.RecognitionConfig.AudioEncoding.LINEAR16

        def _build_recognition_config(
            self, params: dict, speech, file: Union[str, BinaryIO]
        ):
            """Build Google Speech RecognitionConfig from parameters."""
            # Auto-detect encoding if not specified
            encoding = params.get("encoding")
            if encoding is None:
                encoding = self._detect_audio_encoding(file, speech)

            config_params = {
                "encoding": encoding,
                "sample_rate_hertz": params.get("sample_rate_hertz", 16000),
                "language_code": params.get("language_code", "en-US"),
                "enable_word_time_offsets": True,
                "enable_word_confidence": True,
                "enable_automatic_punctuation": params.get(
                    "enable_automatic_punctuation", True
                ),
                "model": params["model"],
            }

            for param in ["max_alternatives", "profanity_filter", "speech_contexts"]:
                if param in params:
                    config_params[param] = params[param]

            return speech.RecognitionConfig(**config_params)

        def _create_streaming_requests(
            self, speech, streaming_config, audio_data: bytes
        ):
            """Create streaming requests generator for Google Speech API."""

            def request_generator():
                chunk_size = 8192
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i : i + chunk_size]
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)

            return request_generator()

        def _parse_google_response(self, response) -> TranscriptionResult:
            """Convert Google Speech-to-Text response to unified TranscriptionResult."""
            if not response.results or not response.results[0].alternatives:
                return TranscriptionResult(
                    text="", language=None, confidence=None, task="transcribe"
                )

            best_result = response.results[0]
            best_alternative = best_result.alternatives[0]
            text = best_alternative.transcript
            confidence = getattr(best_alternative, "confidence", None)

            words = []
            if hasattr(best_alternative, "words") and best_alternative.words:
                words = [
                    Word(
                        word=word.word,
                        start=(
                            word.start_time.total_seconds()
                            if hasattr(word, "start_time")
                            else 0.0
                        ),
                        end=(
                            word.end_time.total_seconds()
                            if hasattr(word, "end_time")
                            else 0.0
                        ),
                        confidence=getattr(word, "confidence", None),
                    )
                    for word in best_alternative.words
                ]

            alternatives = [
                Alternative(
                    transcript=alt.transcript,
                    confidence=getattr(alt, "confidence", None),
                )
                for alt in best_result.alternatives
            ]

            segments = []
            if words:
                segments = [
                    Segment(
                        id=0,
                        seek=0,
                        start=words[0].start,
                        end=words[-1].end,
                        text=text,
                        tokens=[],
                        temperature=0.0,
                        avg_logprob=0.0,
                        compression_ratio=0.0,
                        no_speech_prob=0.0,
                    )
                ]

            return TranscriptionResult(
                text=text,
                language=None,
                confidence=confidence,
                task="transcribe",
                words=words or None,
                alternatives=alternatives or None,
                segments=segments or None,
            )
