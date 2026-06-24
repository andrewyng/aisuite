import openai
import os
from typing import Union, BinaryIO, AsyncGenerator
from aisuite.provider import Provider, LLMError, ASRError, Audio
from aisuite.providers.message_converter import OpenAICompliantMessageConverter
from aisuite.framework.message import (
    TranscriptionResult,
    Segment,
    Word,
    StreamingTranscriptionChunk,
)


_MAX_TOKENS_PARAM = "max_tokens"
_MAX_COMPLETION_TOKENS_PARAM = "max_completion_tokens"
_MAX_COMPLETION_TOKEN_MODEL_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)


def _prefers_max_completion_tokens(model):
    """Return True for OpenAI model families that reject max_tokens."""
    normalized_model = str(model).strip().lower()
    return normalized_model.startswith(_MAX_COMPLETION_TOKEN_MODEL_PREFIXES)


def _normalize_chat_completion_kwargs(model, kwargs):
    """Map aisuite's max_tokens abstraction to OpenAI's current parameter names."""
    normalized_kwargs = dict(kwargs)
    if _prefers_max_completion_tokens(model) and _MAX_TOKENS_PARAM in normalized_kwargs:
        max_tokens = normalized_kwargs.pop(_MAX_TOKENS_PARAM)
        normalized_kwargs.setdefault(_MAX_COMPLETION_TOKENS_PARAM, max_tokens)
    return normalized_kwargs


def _error_body(error):
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return {}
    nested_error = body.get("error")
    if isinstance(nested_error, dict):
        return nested_error
    return body


def _is_unsupported_param_error(error, param):
    body = _error_body(error)
    message = str(body.get("message") or error)
    normalized_message = message.lower()
    reported_param = body.get("param")
    code = body.get("code")
    param_matches = reported_param == param or param in message
    unsupported = (
        code == "unsupported_parameter"
        or "unsupported" in normalized_message
        or "not supported" in normalized_message
    )
    return param_matches and unsupported


def _alternate_token_limit_param(param):
    if param == _MAX_TOKENS_PARAM:
        return _MAX_COMPLETION_TOKENS_PARAM
    return _MAX_TOKENS_PARAM


def _retry_kwargs_for_unsupported_token_param(error, request_kwargs):
    for param in (_MAX_TOKENS_PARAM, _MAX_COMPLETION_TOKENS_PARAM):
        if param not in request_kwargs:
            continue
        if not _is_unsupported_param_error(error, param):
            continue
        retry_kwargs = dict(request_kwargs)
        token_limit = retry_kwargs.pop(param)
        retry_kwargs.setdefault(_alternate_token_limit_param(param), token_limit)
        return retry_kwargs
    return None


class OpenaiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_BASE_URL, etc.

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)
        # Async client shares the same config for true non-blocking I/O.
        self.aclient = openai.AsyncOpenAI(**config)
        self.transformer = OpenAICompliantMessageConverter()

        # Initialize audio functionality
        super().__init__()
        self.audio = OpenAIAudio(self.client)

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        transformed_messages = None
        request_kwargs = None
        try:
            transformed_messages = self.transformer.convert_request(messages)
            request_kwargs = _normalize_chat_completion_kwargs(model, kwargs)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **request_kwargs,  # Pass any additional arguments to the OpenAI API
            )
            return response
        except openai.BadRequestError as e:
            if request_kwargs is None:
                raise LLMError(f"An error occurred: {e}")
            retry_kwargs = _retry_kwargs_for_unsupported_token_param(e, request_kwargs)
            if retry_kwargs is None:
                raise LLMError(f"An error occurred: {e}")
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=transformed_messages,
                    **retry_kwargs,
                )
            except Exception as retry_error:
                raise LLMError(f"An error occurred: {retry_error}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    async def achat_completions_create(self, model, messages, **kwargs):
        # Native async path via openai.AsyncOpenAI.
        transformed_messages = None
        request_kwargs = None
        try:
            transformed_messages = self.transformer.convert_request(messages)
            request_kwargs = _normalize_chat_completion_kwargs(model, kwargs)
            response = await self.aclient.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **request_kwargs,  # Pass any additional arguments to the OpenAI API
            )
            return response
        except openai.BadRequestError as e:
            if request_kwargs is None:
                raise LLMError(f"An error occurred: {e}")
            retry_kwargs = _retry_kwargs_for_unsupported_token_param(e, request_kwargs)
            if retry_kwargs is None:
                raise LLMError(f"An error occurred: {e}")
            try:
                return await self.aclient.chat.completions.create(
                    model=model,
                    messages=transformed_messages,
                    **retry_kwargs,
                )
            except Exception as retry_error:
                raise LLMError(f"An error occurred: {retry_error}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    def chat_completions_create_stream(self, model, messages, **kwargs):
        # OpenAI's own chunks already have the unified shape — pass them through.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            stream = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
        yield from stream

    async def achat_completions_create_stream(self, model, messages, **kwargs):
        # Native async streaming via openai.AsyncOpenAI.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            stream = await self.aclient.chat.completions.create(
                model=model,
                messages=transformed_messages,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")
        async for chunk in stream:
            yield chunk


# Audio Classes
class OpenAIAudio(Audio):
    """OpenAI Audio functionality container."""

    def __init__(self, client):
        super().__init__()
        self.transcriptions = self.Transcriptions(client)

    class Transcriptions(Audio.Transcription):
        """OpenAI Audio Transcriptions functionality."""

        def __init__(self, client):
            self.client = client

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> TranscriptionResult:
            """
            Create audio transcription using OpenAI Whisper API.

            All parameters are already validated and mapped by the Client layer.
            This is a simple pass-through to the OpenAI API.
            """
            try:
                # Handle TranscriptionOptions object if passed
                if "options" in kwargs:
                    options = kwargs.pop("options")
                    # Extract all non-None attributes from options object
                    if hasattr(options, "__dict__"):
                        for key, value in options.__dict__.items():
                            if value is not None and key not in kwargs:
                                kwargs[key] = value

                # Handle timestamp_granularities requirement
                if "timestamp_granularities" in kwargs:
                    # OpenAI requires verbose_json format for timestamp_granularities
                    kwargs["response_format"] = "verbose_json"

                # Handle file input
                if isinstance(file, str):
                    with open(file, "rb") as audio_file:
                        response = self.client.audio.transcriptions.create(
                            file=audio_file, model=model, **kwargs
                        )
                else:
                    response = self.client.audio.transcriptions.create(
                        file=file, model=model, **kwargs
                    )

                return self._parse_openai_response(response)

            except Exception as e:
                raise ASRError(f"OpenAI transcription error: {e}") from e

        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            **kwargs,
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """
            Create streaming audio transcription using OpenAI Whisper API.

            All parameters are already validated and mapped by the Client layer.
            This is a simple pass-through to the OpenAI API with streaming enabled.
            """
            try:
                # Handle TranscriptionOptions object if passed
                if "options" in kwargs:
                    options = kwargs.pop("options")
                    # Extract all non-None attributes from options object
                    if hasattr(options, "__dict__"):
                        for key, value in options.__dict__.items():
                            if value is not None and key not in kwargs:
                                kwargs[key] = value

                # Enable streaming
                kwargs["stream"] = True

                # Handle timestamp_granularities requirement
                if "timestamp_granularities" in kwargs:
                    # OpenAI requires verbose_json format for timestamp_granularities
                    if (
                        "response_format" in kwargs
                        and kwargs["response_format"] != "verbose_json"
                    ):
                        raise ASRError(
                            f"OpenAI timestamp_granularities requires response_format='verbose_json', "
                            f"but got '{kwargs['response_format']}'. "
                            f"Either remove timestamp_granularities or use response_format='verbose_json'."
                        )
                    else:
                        kwargs["response_format"] = "verbose_json"

                try:
                    if isinstance(file, str):
                        with open(file, "rb") as audio_file:
                            response_stream = self.client.audio.transcriptions.create(
                                file=audio_file, model=model, **kwargs
                            )
                    else:
                        response_stream = self.client.audio.transcriptions.create(
                            file=file, model=model, **kwargs
                        )

                    # Process streaming response - handle event types
                    for event in response_stream:
                        # Handle TranscriptionTextDeltaEvent (incremental text)
                        if (
                            hasattr(event, "type")
                            and event.type == "transcript.text.delta"
                        ):
                            if hasattr(event, "delta") and event.delta:
                                yield StreamingTranscriptionChunk(
                                    text=event.delta,
                                    is_final=False,  # Delta events are interim
                                    confidence=getattr(event, "confidence", None),
                                )
                        # Handle TranscriptionTextDoneEvent (final complete text)
                        elif (
                            hasattr(event, "type")
                            and event.type == "transcript.text.done"
                        ):
                            if hasattr(event, "text") and event.text:
                                yield StreamingTranscriptionChunk(
                                    text=event.text,
                                    is_final=True,  # Done event is final
                                    confidence=getattr(event, "confidence", None),
                                )

                except Exception as stream_error:
                    raise ASRError(
                        f"OpenAI streaming transcription error: {stream_error}"
                    ) from stream_error

            except Exception as e:
                raise ASRError(f"OpenAI streaming transcription error: {e}") from e

        def _parse_openai_response(self, response) -> TranscriptionResult:
            """Parse OpenAI API response into TranscriptionResult."""
            text = response.text if hasattr(response, "text") else ""
            language = getattr(response, "language", "unknown")

            # Parse segments if available
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    words = []
                    if hasattr(seg, "words") and seg.words:
                        for word in seg.words:
                            words.append(
                                Word(
                                    word=word.word,
                                    start=word.start,
                                    end=word.end,
                                    confidence=getattr(word, "confidence", None),
                                )
                            )

                    segments.append(
                        Segment(
                            id=getattr(seg, "id", 0),
                            seek=getattr(seg, "seek", 0),
                            text=seg.text,
                            start=seg.start,
                            end=seg.end,
                            words=words,
                            confidence=getattr(seg, "avg_logprob", None),
                        )
                    )

            return TranscriptionResult(
                text=text,
                language=language,
                confidence=getattr(response, "confidence", None),
                segments=segments,
            )
