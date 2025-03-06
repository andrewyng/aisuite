# aisuite/framework/chat_completion_stream_response.py

from typing import Optional

class ChatCompletionStreamResponseDelta:
    """
    Mimics the 'delta' object returned by OpenAI streaming chunks.
    Example usage in code:
        chunk.choices[0].delta.content
    """
    def __init__(self, role: Optional[str] = None, content: Optional[str] = None):
        self.role = role
        self.content = content


class ChatCompletionStreamResponseChoice:
    """
    Holds the 'delta' for a single chunk choice.
    Example usage in code:
        chunk.choices[0].delta
    """
    def __init__(self, delta: ChatCompletionStreamResponseDelta):
        self.delta = delta


class ChatCompletionStreamResponse:
    """
    Container for streaming response chunks.
    Each chunk has a 'choices' list, each with a 'delta'.
    Example usage in code:
        chunk.choices[0].delta.content
    """
    def __init__(self, choices: list[ChatCompletionStreamResponseChoice]):
        self.choices = choices



