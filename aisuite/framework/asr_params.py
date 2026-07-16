"""
ASR parameter registry and validation.

This module provides a unified parameter validation system for audio transcription
across different providers. It supports:
- Common parameters (OpenAI-style) that are auto-mapped to provider equivalents
- Provider-specific parameters that are passed through directly
- Three validation modes: strict, warn, and permissive
"""

from typing import Dict, Set, Any, Optional, Literal
import logging

logger = logging.getLogger(__name__)


# Common parameters that get auto-mapped across providers
# These follow OpenAI's API conventions for maximum portability
COMMON_PARAMS: Dict[str, Dict[str, Optional[str]]] = {
    "language": {
        "openai": "language",
        "deepgram": "language",
        "google": "language_code",
        "huggingface": None,  # Not supported by Inference API
    },
    "prompt": {
        "openai": "prompt",
        "deepgram": "keywords",
        "google": "speech_contexts",
        "huggingface": None,  # Not supported
    },
    "temperature": {
        "openai": "temperature",
        "deepgram": None,  # Not supported
        "google": None,  # Not supported
        "huggingface": "temperature",  # Supported as generation param
    },
}


# Valid provider-specific parameters
# Each provider has its own set of supported parameters
PROVIDER_PARAMS: Dict[str, Set[str]] = {
    "openai": {
        # Basic parameters
        "language",
        "prompt",
        "temperature",
        # Output format
        "response_format",  # "json" | "text" | "srt" | "verbose_json" | "vtt"
        "timestamp_granularities",  # ["word"] | ["segment"] | ["word", "segment"]
        # Streaming
        "stream",  # Boolean
    },
    "deepgram": {
        # Basic parameters
        "language",
        "model",
        # Text enhancement
        "punctuate",  # Auto-add punctuation
        "diarize",  # Speaker diarization
        "utterances",  # Sentence-level timestamps
        "paragraphs",  # Paragraph segmentation
        "smart_format",  # Format numbers, dates, etc.
        "profanity_filter",  # Filter profanity
        # Advanced features
        "search",  # Search for keywords: ["keyword1", "keyword2"]
        "replace",  # Replace words: {"um": "", "uh": ""}
        "keywords",  # Boost keywords: ["important", "technical"]
        "numerals",  # Format numerals
        "measurements",  # Format measurements
        # AI features
        "sentiment",  # Sentiment analysis
        "topics",  # Topic detection
        "intents",  # Intent recognition
        "summarize",  # Auto-summarization
        # Audio format
        "encoding",  # "linear16" | "mp3" | "flac"
        "sample_rate",  # Integer (Hz)
        "channels",  # Integer
        # Quality and alternatives
        "confidence",  # Include confidence scores
        "alternatives",  # Number of alternative transcripts
        # Streaming
        "interim_results",  # Get interim results while streaming
    },
    "google": {
        # Basic parameters
        "language_code",  # BCP-47 code like "en-US"
        "model",  # "latest_long" | "latest_short" | "default"
        # Audio format
        "encoding",  # "LINEAR16" | "FLAC" | "MP3"
        "sample_rate_hertz",  # Integer
        "audio_channel_count",  # Integer
        # Text enhancement
        "enable_automatic_punctuation",  # Boolean
        "profanity_filter",  # Boolean
        "enable_spoken_punctuation",  # Boolean
        "enable_spoken_emojis",  # Boolean
        # Speaker features
        "enable_speaker_diarization",  # Boolean
        "diarization_speaker_count",  # Integer (max speakers)
        "min_speaker_count",  # Integer
        # Metadata
        "enable_word_time_offsets",  # Word-level timestamps
        "enable_word_confidence",  # Word-level confidence
        "max_alternatives",  # Number of alternatives
        # Context
        "speech_contexts",  # [{"phrases": [...], "boost": float}]
        "boost",  # Float (phraseHint boost)
        # Streaming
        "interim_results",  # Boolean
        "single_utterance",  # Boolean (stop after one utterance)
    },
    "huggingface": {
        # Basic parameters
        "model",  # Model ID on Hugging Face Hub
        "temperature",  # Generation temperature
        # API options
        "return_timestamps",  # Boolean or "word" or "char"
        "use_cache",  # Boolean: use cached inference
        "wait_for_model",  # Boolean: wait if model is loading
        # Generation parameters
        "top_k",  # Integer: top-k sampling
        "top_p",  # Float: nucleus sampling
        "max_length",  # Integer: maximum output length
        "do_sample",  # Boolean: enable sampling
    },
}


# Language code expansion for Google (2-letter to locale codes)
GOOGLE_LANGUAGE_MAP = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "zh": "zh-CN",
    "ar": "ar-SA",
    "hi": "hi-IN",
    "ru": "ru-RU",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "da": "da-DK",
    "no": "nb-NO",
    "fi": "fi-FI",
    "tr": "tr-TR",
    "th": "th-TH",
    "vi": "vi-VN",
}


class ParamValidator:
    """
    Validates and maps ASR parameters for different providers.

    This class handles three types of parameters:
    1. Common parameters (OpenAI-style) - auto-mapped to provider equivalents
    2. Provider-specific parameters - passed through with validation
    3. Unknown parameters - handled based on extra_param_mode
    """

    def __init__(self, extra_param_mode: Literal["strict", "warn", "permissive"]):
        """
        Initialize the parameter validator.

        Args:
            extra_param_mode: How to handle unknown parameters
                - "strict": Raise ValueError on unknown params
                - "warn": Log warning on unknown params (default)
                - "permissive": Allow all params without validation
        """
        self.extra_param_mode = extra_param_mode

    def validate_and_map(
        self, provider_key: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and map parameters for the given provider.

        This method:
        1. Maps common parameters to provider-specific equivalents
        2. Validates provider-specific parameters
        3. Handles unknown parameters based on extra_param_mode

        Args:
            provider_key: Provider identifier (e.g., "openai", "deepgram")
            params: Raw parameters from user

        Returns:
            Validated and mapped parameters ready for provider API

        Raises:
            ValueError: If extra_param_mode="strict" and unknown params found
        """
        result = {}
        unknown_params = []
        provider_params = PROVIDER_PARAMS.get(provider_key, set())

        for key, value in params.items():
            # Check if it's a common param that needs mapping
            if key in COMMON_PARAMS:
                mapped_key = COMMON_PARAMS[key].get(provider_key)

                # Provider doesn't support this common param
                if mapped_key is None:
                    logger.debug(
                        f"Parameter '{key}' not supported by {provider_key}, ignoring"
                    )
                    continue

                # Transform value if needed (e.g., "en" -> "en-US" for Google)
                mapped_value = self._transform_value(provider_key, key, value)
                result[mapped_key] = mapped_value

            # Check if it's a valid provider-specific param
            elif key in provider_params:
                result[key] = value

            # Unknown parameter
            else:
                unknown_params.append(key)

        # Handle unknown parameters based on mode
        if unknown_params:
            self._handle_unknown(provider_key, unknown_params)

            # In permissive mode, still pass them through
            if self.extra_param_mode == "permissive":
                for key in unknown_params:
                    result[key] = params[key]

        return result

    def _transform_value(self, provider_key: str, param_key: str, value: Any) -> Any:
        """
        Transform parameter values during mapping.

        This handles provider-specific transformations like:
        - Google: Expanding "en" to "en-US"
        - Google: Wrapping prompt in speech_contexts structure
        - Deepgram: Converting prompt string to keywords list

        Args:
            provider_key: Provider identifier
            param_key: Parameter name (from COMMON_PARAMS)
            value: Parameter value to transform

        Returns:
            Transformed parameter value
        """
        # Google: Expand 2-letter language codes to locale codes
        if provider_key == "google" and param_key == "language":
            if isinstance(value, str) and len(value) == 2:
                return GOOGLE_LANGUAGE_MAP.get(value, f"{value}-US")

        # Google: Wrap prompt in speech_contexts structure
        if provider_key == "google" and param_key == "prompt":
            return [{"phrases": [value]}]

        # Deepgram: Split prompt into keywords list
        if provider_key == "deepgram" and param_key == "prompt":
            if isinstance(value, str):
                return value.split()
            return value

        return value

    def _handle_unknown(self, provider_key: str, unknown_params: list):
        """
        Handle unknown parameters based on extra_param_mode.

        Args:
            provider_key: Provider identifier
            unknown_params: List of unknown parameter names

        Raises:
            ValueError: If extra_param_mode="strict"
        """
        msg = (
            f"Unknown parameters for {provider_key}: {unknown_params}. "
            f"See {provider_key} documentation for valid parameters."
        )

        if self.extra_param_mode == "strict":
            raise ValueError(msg)
        elif self.extra_param_mode == "warn":
            import warnings

            warnings.warn(msg, UserWarning)
        # permissive mode: do nothing
