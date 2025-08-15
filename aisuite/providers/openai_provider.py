import openai
import os
from typing import Union, BinaryIO
from aisuite.provider import Provider, LLMError, ASRError
from aisuite.providers.message_converter import OpenAICompliantMessageConverter
from aisuite.framework.message import TranscriptionResult, Segment, Word


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
        self.transformer = OpenAICompliantMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        try:
            transformed_messages = self.transformer.convert_request(messages)
            response = self.client.chat.completions.create(
                model=model,
                messages=transformed_messages,
                **kwargs,  # Pass any additional arguments to the OpenAI API
            )
            return response
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    def audio_transcriptions_create(
        self, model: str, file: Union[str, BinaryIO], **kwargs
    ) -> TranscriptionResult:
        """Create audio transcription using OpenAI Whisper API."""
        try:
            # Handle file input - if it's a string, assume it's a file path
            if isinstance(file, str):
                with open(file, "rb") as audio_file:
                    response = self.client.audio.transcriptions.create(
                        model=model, file=audio_file, **kwargs
                    )
            else:
                # Assume it's a file-like object
                response = self.client.audio.transcriptions.create(
                    model=model, file=file, **kwargs
                )

            return self._parse_openai_response(response)

        except Exception as e:
            raise ASRError(f"OpenAI transcription error: {e}")

    def _parse_openai_response(self, response) -> TranscriptionResult:
        """Convert OpenAI API response to unified TranscriptionResult."""
        # Extract basic fields
        text = response.text
        task = getattr(response, "task", None)
        language = getattr(response, "language", None)
        duration = getattr(response, "duration", None)

        # Handle segments if present (when timestamp_granularities includes 'segment')
        segments = []
        if hasattr(response, "segments") and response.segments:
            segments = [
                Segment(
                    id=seg.id,
                    seek=seg.seek,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    tokens=getattr(seg, "tokens", None),
                    temperature=getattr(seg, "temperature", None),
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    compression_ratio=getattr(seg, "compression_ratio", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                )
                for seg in response.segments
            ]

        # Handle words if present (when timestamp_granularities includes 'word')
        words = []
        if hasattr(response, "words") and response.words:
            words = [
                Word(word=word.word, start=word.start, end=word.end)
                for word in response.words
            ]

        return TranscriptionResult(
            text=text,
            task=task,
            language=language,
            duration=duration,
            confidence=None,  # OpenAI doesn't provide overall confidence
            segments=segments,
            words=words,
        )
