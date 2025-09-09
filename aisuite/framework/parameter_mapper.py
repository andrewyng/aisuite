"""
Parameter mapping utilities for ASR providers.
Maps unified TranscriptionOptions to provider-specific parameters.
"""

from typing import Dict, Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .message import TranscriptionOptions


class ParameterMapper:
    """Maps unified TranscriptionOptions to provider-specific parameters."""

    # OpenAI Whisper API parameter mapping
    OPENAI_MAPPING = {
        "language": "language",
        "response_format": "response_format",
        "temperature": "temperature",
        "prompt": "prompt",
        "stream": "stream",
        "timestamp_granularities": "timestamp_granularities",
    }

    # Deepgram API parameter mapping
    DEEPGRAM_MAPPING = {
        "language": "language",
        "enable_automatic_punctuation": "punctuate",
        "enable_smart_formatting": "smart_format",
        "enable_speaker_diarization": "diarize",
        "include_word_timestamps": "utterances",
        "include_segment_timestamps": "paragraphs",
        "context_phrases": "keywords",
        "enable_profanity_filter": "profanity_filter",
        "enable_sentiment_analysis": "sentiment",
        "enable_topic_detection": "topics",
        "enable_intent_recognition": "intents",
        "enable_summarization": "summarize",
        "interim_results": "interim_results",
        "channels": "channels",
        "sample_rate": "sample_rate",
        "include_confidence_scores": "confidence",
        "enable_word_confidence": "confidence",
        "max_alternatives": "alternatives",
        "stream": "interim_results",
        "encoding": "encoding",
        # timestamp_granularities is handled specially for Deepgram
    }

    # Google API parameter mapping
    GOOGLE_MAPPING = {
        "language": "language_code",
        "sample_rate": "sample_rate_hertz",
        "channels": "audio_channel_count",
        "enable_automatic_punctuation": "enable_automatic_punctuation",
        "enable_speaker_diarization": "enable_speaker_diarization",
        "max_speakers": "diarization_speaker_count",
        "min_speakers": "min_speaker_count",
        "include_word_timestamps": "enable_word_time_offsets",
        "include_confidence_scores": "enable_word_confidence",
        "enable_word_confidence": "enable_word_confidence",
        "context_phrases": "speech_contexts",
        "enable_profanity_filter": "profanity_filter",
        "max_alternatives": "max_alternatives",
        "boost_phrases": "speech_contexts",
        "audio_format": "encoding",
        "encoding": "encoding",
        "interim_results": "interim_results",
        "stream": "interim_results",
        "enable_spoken_punctuation": "enable_spoken_punctuation",
        "enable_spoken_emojis": "enable_spoken_emojis",
    }

    @classmethod
    def map_to_openai(cls, options: "TranscriptionOptions") -> Dict[str, Any]:
        """Map TranscriptionOptions to OpenAI Whisper API parameters."""
        params = {}

        # Handle timestamp granularities
        timestamp_granularities = []
        if options.include_word_timestamps:
            timestamp_granularities.append("word")
        if options.include_segment_timestamps:
            timestamp_granularities.append("segment")
        if timestamp_granularities:
            params["timestamp_granularities"] = timestamp_granularities

        # Map other parameters
        for opt_key, api_key in cls.OPENAI_MAPPING.items():
            if hasattr(options, opt_key):
                value = getattr(options, opt_key)
                if value is not None and not opt_key.startswith("include_"):
                    params[api_key] = value

        # Handle custom parameters
        cls._apply_custom_parameters(params, options.custom_parameters, "openai")

        return params

    @classmethod
    def map_to_deepgram(cls, options: "TranscriptionOptions") -> Dict[str, Any]:
        """Map TranscriptionOptions to Deepgram API parameters."""
        params = {}

        for opt_key, api_key in cls.DEEPGRAM_MAPPING.items():
            if hasattr(options, opt_key):
                value = getattr(options, opt_key)
                if value is not None:
                    params[api_key] = value

        # Handle special cases
        if options.context_phrases:
            params["keywords"] = options.context_phrases

        # Handle timestamp_granularities conversion for Deepgram
        if (
            hasattr(options, "timestamp_granularities")
            and options.timestamp_granularities
        ):
            if "word" in options.timestamp_granularities:
                params["utterances"] = True
            if "segment" in options.timestamp_granularities:
                params["paragraphs"] = True

        # Handle custom parameters
        cls._apply_custom_parameters(params, options.custom_parameters, "deepgram")

        return params

    @classmethod
    def map_to_google(cls, options: "TranscriptionOptions") -> Dict[str, Any]:
        """Map TranscriptionOptions to Google Speech-to-Text API parameters."""
        params = {}

        for opt_key, api_key in cls.GOOGLE_MAPPING.items():
            if hasattr(options, opt_key):
                value = getattr(options, opt_key)
                if value is not None:
                    if opt_key == "context_phrases" or opt_key == "boost_phrases":
                        if "speech_contexts" not in params:
                            params["speech_contexts"] = []
                        params["speech_contexts"].append({"phrases": value})
                    elif opt_key == "language":
                        # Handle language code conversion for Google
                        # Google expects BCP-47 locale codes like "en-US", not just "en"
                        if len(value) == 2:  # Convert "en" to "en-US"
                            language_map = {
                                "en": "en-US",
                                "es": "es-ES",
                                "fr": "fr-FR",
                                "de": "de-DE",
                                "it": "it-IT",
                                "pt": "pt-BR",  # Portuguese -> Brazilian Portuguese
                                "ja": "ja-JP",
                                "ko": "ko-KR",
                                "zh": "zh-CN",  # Chinese -> Simplified Chinese
                                "ar": "ar-SA",  # Arabic -> Saudi Arabia
                                "hi": "hi-IN",  # Hindi -> India
                                "ru": "ru-RU",  # Russian -> Russia
                                "nl": "nl-NL",  # Dutch -> Netherlands
                                "pl": "pl-PL",  # Polish -> Poland
                                "sv": "sv-SE",  # Swedish -> Sweden
                                "da": "da-DK",  # Danish -> Denmark
                                "no": "nb-NO",  # Norwegian -> Norway
                                "fi": "fi-FI",  # Finnish -> Finland
                                "tr": "tr-TR",  # Turkish -> Turkey
                                "th": "th-TH",  # Thai -> Thailand
                                "vi": "vi-VN",  # Vietnamese -> Vietnam
                            }
                            params[api_key] = language_map.get(value, f"{value}-US")
                        else:
                            params[api_key] = value
                    else:
                        params[api_key] = value

        # Handle audio encoding mapping
        if options.audio_format:
            encoding_map = {
                "wav": "LINEAR16",
                "flac": "FLAC",
                "mp3": "MP3",
                "ogg": "OGG_OPUS",
                "webm": "WEBM_OPUS",
            }
            params["encoding"] = encoding_map.get(
                options.audio_format.lower(), "LINEAR16"
            )

        # Handle timestamp_granularities conversion for Google
        if (
            hasattr(options, "timestamp_granularities")
            and options.timestamp_granularities
        ):
            if "word" in options.timestamp_granularities:
                params["enable_word_time_offsets"] = True

        # Handle custom parameters
        cls._apply_custom_parameters(params, options.custom_parameters, "google")

        return params

    @classmethod
    def _apply_custom_parameters(
        cls, params: Dict[str, Any], custom_params: Dict[str, Any], provider: str
    ):
        """
        Apply custom parameters for the specific provider.

        Only provider-namespaced parameters are supported.
        Parameters not under a provider key are IGNORED.
        """
        if not custom_params:
            return

        # Provider-specific namespacing ONLY
        # Users MUST structure custom_parameters like:
        # {
        #   "openai": {"response_format": "srt", "temperature": 0.2},
        #   "deepgram": {"search": ["keyword"], "numerals": True},
        #   "google": {"use_enhanced": True, "adaptation": {...}}
        # }
        if provider in custom_params:
            params.update(custom_params[provider])
        # Note: Any parameters not under a provider key are ignored
