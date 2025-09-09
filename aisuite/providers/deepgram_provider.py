import os
import json
import numpy as np
import queue
import threading
import time
from typing import Union, BinaryIO, Optional, AsyncGenerator

from aisuite.provider import Provider, ASRError, Audio
from aisuite.framework.message import (
    TranscriptionResult,
    Segment,
    Word,
    Alternative,
    Channel,
    TranscriptionOptions,
    StreamingTranscriptionChunk,
)
from aisuite.framework.parameter_mapper import ParameterMapper


class DeepgramProvider(Provider):
    """Deepgram ASR provider."""

    def __init__(self, **config):
        """Initialize the Deepgram provider with the given configuration."""
        super().__init__()

        # Ensure API key is provided either in config or via environment variable
        self.api_key = config.get("api_key") or os.getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Deepgram API key is missing. Please provide it in the config or set the DEEPGRAM_API_KEY environment variable."
            )

        # Initialize Deepgram client
        try:
            from deepgram import DeepgramClient, DeepgramClientOptions

            client_options = DeepgramClientOptions(
                api_key=self.api_key, options={"keepalive": "true"}
            )
            self.client = DeepgramClient(self.api_key, client_options)
        except ImportError:
            raise ImportError(
                "Deepgram SDK is required. Install it with: pip install deepgram-sdk"
            )

        # Initialize audio functionality
        self.audio = DeepgramAudio(self.client)

    def chat_completions_create(self, model, messages):
        """Deepgram does not support chat completions."""
        raise NotImplementedError(
            "Deepgram provider only supports audio transcription, not chat completions."
        )


# Audio Classes
class DeepgramAudio(Audio):
    """Deepgram Audio functionality container."""

    def __init__(self, client):
        super().__init__()
        self.transcriptions = self.Transcriptions(client)

    class Transcriptions(Audio.Transcription):
        """Deepgram Audio Transcriptions functionality."""

        def __init__(self, client):
            self.client = client

        def create(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options: Optional[TranscriptionOptions] = None,
            **kwargs,
        ) -> TranscriptionResult:
            """Create audio transcription using Deepgram SDK."""
            try:
                from deepgram import PrerecordedOptions

                api_params = self._prepare_api_params(model, options, kwargs)
                deepgram_options = PrerecordedOptions(**api_params)
                payload = self._prepare_audio_payload(file)

                response = self.client.listen.rest.v("1").transcribe_file(
                    payload, deepgram_options
                )

                response_dict = (
                    response.to_dict() if hasattr(response, "to_dict") else response
                )

                return self._parse_deepgram_response(response_dict)

            except Exception as e:
                raise ASRError(f"Deepgram transcription error: {e}")

        async def create_stream_output(
            self,
            model: str,
            file: Union[str, BinaryIO],
            options: Optional[TranscriptionOptions] = None,
            chunk_size_minutes: float = 3.0,
            **kwargs,
        ) -> AsyncGenerator[StreamingTranscriptionChunk, None]:
            """Create streaming audio transcription using Deepgram SDK with chunked processing."""
            try:
                from deepgram import LiveOptions
                from deepgram.clients.listen import LiveTranscriptionEvents

                # Load and prepare audio
                audio_data, sample_rate = await self._load_and_prepare_audio(file)

                # Calculate chunking strategy
                duration_seconds = len(audio_data) / sample_rate
                chunk_duration_seconds = chunk_size_minutes * 60

                if duration_seconds <= chunk_duration_seconds:
                    chunks = [audio_data]
                else:
                    chunk_size_samples = int(chunk_duration_seconds * sample_rate)
                    chunks = []
                    num_chunks = int(np.ceil(duration_seconds / chunk_duration_seconds))
                    for i in range(num_chunks):
                        start_sample = i * chunk_size_samples
                        end_sample = min(
                            start_sample + chunk_size_samples, len(audio_data)
                        )
                        chunks.append(audio_data[start_sample:end_sample])

                # Setup API parameters
                api_params = self._prepare_api_params(
                    model, options, kwargs, is_streaming=True
                )
                api_params["interim_results"] = (
                    True  # Enable interim results for streaming
                )

                # Add critical audio format parameters (matching reference)
                api_params["encoding"] = "linear16"  # PCM16 format
                api_params["sample_rate"] = 16000  # Match our target sample rate
                api_params["channels"] = 1  # Mono audio

                live_options = LiveOptions(**api_params)

                # Create single connection for all chunks
                connection = self.client.listen.websocket.v("1")

                # Use thread-safe queue instead of asyncio.Queue for cross-thread communication
                transcript_queue = queue.Queue()
                connection_closed = threading.Event()

                def on_message(*args, **kwargs):
                    """Handle transcript events"""
                    # Extract result from args or kwargs (following reference pattern)
                    result = None
                    if len(args) >= 2:
                        result = args[1]
                    elif "result" in kwargs:
                        result = kwargs["result"]
                    else:
                        return

                    if hasattr(result, "channel") and result.channel.alternatives:
                        alt = result.channel.alternatives[0]
                        if alt.transcript:
                            chunk = StreamingTranscriptionChunk(
                                text=alt.transcript,
                                is_final=getattr(result, "is_final", False),
                                confidence=getattr(alt, "confidence", None),
                            )
                            transcript_queue.put(chunk)  # Thread-safe put

                def on_error(*args, **kwargs):
                    """Handle error events"""
                    # Extract error from args or kwargs
                    error = None
                    if len(args) >= 2:
                        error = args[1]
                    elif "error" in kwargs:
                        error = kwargs["error"]

                    if error:
                        transcript_queue.put(
                            ASRError(f"Deepgram streaming error: {error}")
                        )  # Thread-safe put

                def on_close(*args, **kwargs):
                    """Handle connection close events"""
                    connection_closed.set()

                # Register event handlers
                connection.on(LiveTranscriptionEvents.Transcript, on_message)
                connection.on(LiveTranscriptionEvents.Error, on_error)
                connection.on(LiveTranscriptionEvents.Close, on_close)

                # Start connection
                if not connection.start(live_options):
                    raise ASRError("Failed to start Deepgram streaming connection")

                # Send all chunks through single connection
                try:
                    for audio_chunk in chunks:
                        self._send_audio_chunk(connection, audio_chunk)

                    # Send CloseStream message to signal end of all chunks
                    close_stream_message = json.dumps({"type": "CloseStream"})
                    connection.send(close_stream_message)

                    # Yield results until connection closes naturally
                    while not connection_closed.is_set():
                        try:
                            # Use thread-safe queue with timeout
                            chunk = transcript_queue.get(timeout=0.1)
                            if isinstance(chunk, Exception):
                                raise chunk
                            yield chunk
                        except queue.Empty:
                            continue

                    # Get any remaining results
                    while not transcript_queue.empty():
                        try:
                            chunk = transcript_queue.get_nowait()
                            if isinstance(chunk, Exception):
                                raise chunk
                            yield chunk
                        except queue.Empty:
                            break

                except Exception as e:
                    raise ASRError(f"Error during audio streaming: {e}")

            except Exception as e:
                raise ASRError(f"Deepgram streaming transcription error: {e}")

        def _extract_model_name(self, model: str) -> str:
            """Use model name directly (client already extracted it)."""
            return model

        def _prepare_api_params(
            self,
            model: str,
            options: Optional[TranscriptionOptions],
            kwargs: dict,
            is_streaming: bool = False,
        ) -> dict:
            """Prepare API parameters for Deepgram."""
            if options is not None:
                api_params = ParameterMapper.map_to_deepgram(options)
            else:
                api_params = self._map_openai_to_deepgram_params(kwargs)

            # Remove parameters not supported by LiveOptions (streaming)
            if is_streaming:
                # utterances is only supported in batch/prerecorded, not streaming
                api_params.pop("utterances", None)

            model_name = self._extract_model_name(model)
            api_params.setdefault("smart_format", True)
            api_params.setdefault("punctuate", True)
            api_params.setdefault("language", "en")
            api_params["model"] = model_name
            return api_params

        def _prepare_audio_payload(self, file: Union[str, BinaryIO]) -> dict:
            """Prepare audio payload for Deepgram API."""
            if isinstance(file, str):
                with open(file, "rb") as audio_file:
                    buffer_data = audio_file.read()
            else:
                if hasattr(file, "read"):
                    buffer_data = file.read()
                else:
                    raise ValueError(
                        "File must be a file path string or file-like object"
                    )
            return {"buffer": buffer_data}

        async def _load_and_prepare_audio(
            self, file: Union[str, BinaryIO]
        ) -> tuple[np.ndarray, int]:
            """Load and prepare audio file for streaming.

            Conversions performed only when necessary:
            - Stereo to mono: Required for multi-channel audio
            - Sample rate conversion: Required when input != 16kHz
            - Other formats: Error out as unsupported
            """
            try:
                try:
                    import soundfile as sf
                except ImportError:
                    raise ASRError(
                        "soundfile is required for audio processing. Install with: pip install soundfile"
                    )

                if isinstance(file, str):
                    audio_data, original_sample_rate = sf.read(file)
                else:
                    audio_data, original_sample_rate = sf.read(file)

                audio_data = np.asarray(audio_data, dtype=np.float32)

                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    if audio_data.shape[1] == 2:
                        audio_data = np.mean(audio_data, axis=1)
                    else:
                        raise ASRError(
                            f"Unsupported audio format: {audio_data.shape[1]} channels. Only mono and stereo are supported."
                        )

                # Resample to 16kHz if needed
                target_sample_rate = 16000
                if original_sample_rate != target_sample_rate:
                    try:
                        from scipy import signal

                        num_samples = int(
                            len(audio_data) * target_sample_rate / original_sample_rate
                        )
                        audio_data = signal.resample(audio_data, num_samples)
                    except ImportError:
                        raise ASRError(
                            f"Audio resampling required but scipy not available. "
                            f"Input is {original_sample_rate}Hz, need {target_sample_rate}Hz. "
                            f"Install scipy or provide audio at {target_sample_rate}Hz."
                        )

                return np.asarray(audio_data, dtype=np.float32), target_sample_rate

            except Exception as e:
                if isinstance(e, ASRError):
                    raise
                raise ASRError(f"Error loading audio file: {e}")

        def _send_audio_chunk(self, connection, audio_chunk: np.ndarray) -> None:
            """Send audio chunk data through the connection."""
            streaming_chunk_size = 8000  # Match reference BLOCKSIZE (~0.5s @16kHz mono)
            send_delay = 0.01

            for i in range(0, len(audio_chunk), streaming_chunk_size):
                piece = audio_chunk[i : i + streaming_chunk_size]

                if len(piece) < streaming_chunk_size:
                    piece = np.pad(
                        piece, (0, streaming_chunk_size - len(piece)), mode="constant"
                    )

                pcm16 = (piece * 32767).astype(np.int16).tobytes()
                connection.send(pcm16)
                time.sleep(send_delay)  # Use synchronous sleep like reference

        def _map_openai_to_deepgram_params(self, openai_params: dict) -> dict:
            """Map OpenAI-style parameters to Deepgram parameters."""
            deepgram_params = {}

            if "language" in openai_params:
                deepgram_params["language"] = openai_params["language"]
            if "prompt" in openai_params:
                deepgram_params["keywords"] = openai_params["prompt"]
            if "timestamp_granularities" in openai_params:
                granularities = openai_params["timestamp_granularities"]
                if "word" in granularities:
                    deepgram_params["punctuate"] = True
                    # Note: utterances is only for batch/prerecorded, not streaming

            # Essential for streaming - map interim_results
            if "interim_results" in openai_params:
                deepgram_params["interim_results"] = openai_params["interim_results"]

            return deepgram_params

        def _parse_deepgram_response(self, response_dict: dict) -> TranscriptionResult:
            """Convert Deepgram API response to unified TranscriptionResult."""
            try:
                results = response_dict.get("results", {})
                channels = results.get("channels", [])

                if not channels or not channels[0].get("alternatives"):
                    return TranscriptionResult(
                        text="", language=None, confidence=None, task="transcribe"
                    )

                best_alternative = channels[0]["alternatives"][0]
                text = best_alternative.get("transcript", "")
                confidence = best_alternative.get("confidence", None)

                words = [
                    Word(
                        word=word_data.get("word", ""),
                        start=word_data.get("start", None),
                        end=word_data.get("end", None),
                        confidence=word_data.get("confidence", None),
                    )
                    for word_data in best_alternative.get("words", [])
                ]

                segments = []
                paragraphs = results.get("paragraphs", {}).get("paragraphs", [])
                for para in paragraphs:
                    for sentence in para.get("sentences", []):
                        segments.append(
                            Segment(
                                id=len(segments),
                                seek=0,
                                start=sentence.get("start", None),
                                end=sentence.get("end", None),
                                text=sentence.get("text", ""),
                                tokens=[],
                                temperature=0.0,
                                avg_logprob=0.0,
                                compression_ratio=0.0,
                                no_speech_prob=0.0,
                            )
                        )

                alternatives_list = [
                    Alternative(
                        transcript=alt.get("transcript", ""),
                        confidence=alt.get("confidence", None),
                    )
                    for alt in channels[0]["alternatives"][1:]
                ]

                channels_list = [
                    Channel(
                        alternatives=[
                            Alternative(
                                transcript=alt.get("transcript", ""),
                                confidence=alt.get("confidence", None),
                            )
                            for alt in channel.get("alternatives", [])
                        ]
                    )
                    for channel in channels
                ]

                metadata = response_dict.get("metadata", {})

                return TranscriptionResult(
                    text=text,
                    language=results.get("language", None),
                    confidence=confidence,
                    task="transcribe",
                    duration=metadata.get("duration", None) if metadata else None,
                    segments=segments or None,
                    words=words or None,
                    channels=channels_list or None,
                    alternatives=alternatives_list or None,
                    utterances=results.get("utterances", []),
                    paragraphs=results.get("paragraphs", None),
                    topics=results.get("topics", []),
                    intents=results.get("intents", []),
                    sentiment=results.get("sentiment", None),
                    summary=results.get("summary", None),
                    metadata=metadata,
                )

            except (KeyError, TypeError, IndexError) as e:
                raise ASRError(f"Error parsing Deepgram response: {e}")
