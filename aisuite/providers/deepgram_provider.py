import os
import requests
import json
from typing import Union, BinaryIO
from aisuite.provider import Provider, ASRError
from aisuite.framework.message import (
    TranscriptionResult,
    Segment,
    Word,
    Alternative,
    Channel,
)


class DeepgramProvider(Provider):
    """Deepgram ASR provider."""

    def __init__(self, **config):
        """Initialize the Deepgram provider with the given configuration."""
        # Ensure API key is provided either in config or via environment variable
        self.api_key = config.get("api_key") or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key is missing. Please provide it in the config or set the DEEPGRAM_API_KEY environment variable."
            )

        self.base_url = config.get("base_url", "https://api.deepgram.com/v1/listen")

    def chat_completions_create(self, model, messages):
        """Deepgram does not support chat completions."""
        raise NotImplementedError(
            "Deepgram provider only supports audio transcription, not chat completions."
        )

    def audio_transcriptions_create(
        self, model: str, file: Union[str, BinaryIO], **kwargs
    ) -> TranscriptionResult:
        """Perform transcription using Deepgram API."""
        try:
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/*",
            }

            # Build query parameters
            params = {"model": model}
            params.update(kwargs)

            # Handle file input
            if isinstance(file, str):
                with open(file, "rb") as audio_file:
                    audio_data = audio_file.read()
            else:
                # Assume it's a file-like object
                audio_data = file.read()

            # Make API request
            response = requests.post(
                self.base_url,
                headers=headers,
                params=params,
                data=audio_data,
                timeout=60,
            )

            if response.status_code != 200:
                raise ASRError(
                    f"Deepgram API error: {response.status_code} - {response.text}"
                )

            response_data = response.json()
            return self._parse_deepgram_response(response_data)

        except requests.RequestException as e:
            raise ASRError(f"Deepgram API request error: {e}")
        except Exception as e:
            raise ASRError(f"Deepgram transcription error: {e}")

    def _parse_deepgram_response(self, response_data: dict) -> TranscriptionResult:
        """Convert Deepgram API response to unified TranscriptionResult."""
        try:
            # Extract the main transcript
            results = response_data.get("results", {})
            channels_data = results.get("channels", [])

            if not channels_data:
                return TranscriptionResult(text="", language=None)

            # Get the first channel
            channel_data = channels_data[0]
            alternatives_data = channel_data.get("alternatives", [])

            if not alternatives_data:
                return TranscriptionResult(text="", language=None)

            # Get the best alternative
            best_alternative = alternatives_data[0]
            text = best_alternative.get("transcript", "")
            overall_confidence = best_alternative.get("confidence", None)

            # Extract language if available
            language = results.get("language", None)

            # Parse segments/paragraphs if available
            segments = []
            words = []

            # Handle paragraphs (segments)
            paragraphs = best_alternative.get("paragraphs", {})
            if paragraphs and "paragraphs" in paragraphs:
                for i, paragraph in enumerate(paragraphs["paragraphs"]):
                    segments.append(
                        Segment(
                            id=i,
                            seek=0,  # Deepgram doesn't provide seek
                            start=paragraph.get("start", 0.0),
                            end=paragraph.get("end", 0.0),
                            text=paragraph.get("text", ""),
                            confidence=paragraph.get("confidence", None),
                        )
                    )

            # Handle words if available with enhanced fields
            deepgram_words = best_alternative.get("words", [])
            for word_data in deepgram_words:
                words.append(
                    Word(
                        word=word_data.get("word", ""),
                        start=word_data.get("start", 0.0),
                        end=word_data.get("end", 0.0),
                        confidence=word_data.get("confidence", None),
                        speaker=word_data.get("speaker", None),
                        speaker_confidence=word_data.get("speaker_confidence", None),
                        punctuated_word=word_data.get("punctuated_word", None),
                    )
                )

            # Build alternatives list for full Deepgram compatibility
            alternatives = []
            for alt_data in alternatives_data:
                alt_words = []
                for word_data in alt_data.get("words", []):
                    alt_words.append(
                        Word(
                            word=word_data.get("word", ""),
                            start=word_data.get("start", 0.0),
                            end=word_data.get("end", 0.0),
                            confidence=word_data.get("confidence", None),
                            speaker=word_data.get("speaker", None),
                        )
                    )

                alternatives.append(
                    Alternative(
                        transcript=alt_data.get("transcript", ""),
                        confidence=alt_data.get("confidence", None),
                        words=alt_words if alt_words else None,
                    )
                )

            # Build channels list
            channels = []
            for chan_data in channels_data:
                chan_alternatives = []
                for alt_data in chan_data.get("alternatives", []):
                    chan_alternatives.append(
                        Alternative(
                            transcript=alt_data.get("transcript", ""),
                            confidence=alt_data.get("confidence", None),
                        )
                    )

                channels.append(
                    Channel(
                        alternatives=chan_alternatives,
                        search=chan_data.get("search", None),
                    )
                )

            # Extract advanced features if available
            metadata = response_data.get("metadata", None)
            utterances = best_alternative.get("utterances", None)
            # Convert paragraphs dict to list format for TranscriptionResult
            paragraphs_data = paragraphs.get("paragraphs", []) if paragraphs else None
            topics = results.get("topics", None)
            intents = results.get("intents", None)
            sentiment = results.get("sentiment", None)
            summary = results.get("summary", None)

            return TranscriptionResult(
                text=text,
                language=language,
                confidence=overall_confidence,
                task="transcribe",  # Deepgram is always transcription
                duration=metadata.get("duration", None) if metadata else None,
                segments=segments if segments else None,
                words=words if words else None,
                channels=channels if channels else None,
                alternatives=alternatives if alternatives else None,
                utterances=utterances,
                paragraphs=paragraphs_data,
                topics=topics,
                intents=intents,
                sentiment=sentiment,
                summary=summary,
                metadata=metadata,
            )

        except (KeyError, TypeError, IndexError) as e:
            raise ASRError(f"Error parsing Deepgram response: {e}")
