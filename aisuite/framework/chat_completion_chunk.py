"""OpenAI `chat.completion.chunk`-shaped models for unified chat streaming.

Providers that stream natively in OpenAI's wire format (e.g. OpenAI, Ollama)
yield the SDK's own chunk objects; providers with a different event stream
(e.g. Anthropic) are normalized into these models. Both are duck-type
compatible: `chunk.choices[0].delta.content`, `delta.tool_calls`, and
`choices[0].finish_reason` read the same either way.
"""

from typing import List, Optional

from pydantic import BaseModel

from aisuite.framework.message import CompletionUsage


class ChoiceDeltaFunction(BaseModel):
    """Function fragment inside a streamed tool call: the name arrives on the
    first fragment, the JSON arguments arrive as string pieces to concatenate."""

    name: Optional[str] = None
    arguments: Optional[str] = None


class ChoiceDeltaToolCall(BaseModel):
    """One streamed tool-call fragment. `index` identifies which tool call the
    fragment belongs to (0-based across the assistant turn's tool calls)."""

    index: int
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[ChoiceDeltaFunction] = None


class ChoiceDelta(BaseModel):
    """The incremental payload of one chunk."""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None


class StreamChoice(BaseModel):
    """One choice within a chunk. `finish_reason` is set on the final chunk,
    using the same normalized values as non-streaming responses."""

    index: int = 0
    delta: ChoiceDelta = ChoiceDelta()
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """One streamed piece of a chat completion. `usage`, when the provider
    reports it, rides on the final chunk."""

    choices: List[StreamChoice]
    usage: Optional[CompletionUsage] = None
