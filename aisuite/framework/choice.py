from aisuite.framework.message import Message
from typing import Literal, Optional, List


class Choice:
    """Represents a choice with a finish reason and a message.

    This class encapsulates the state of a choice, which includes the reason
    for its completion and an associated message. The message contains content,
    potential tool calls, the role of the message sender, and optional fields
    for refusal and reasoning content.

    Attributes:
        finish_reason (Optional[Literal["stop", "tool_calls"]]): The reason
            for the choice's completion. It can be either "stop" or 
            "tool_calls", or None if not set.
        message (Message): An instance of the Message class with attributes
            for content, tool calls, role, refusal, and reasoning content."""
    def __init__(self):
        """Initializes an assistant response object with default values.

    Attributes:
        finish_reason (Optional[Literal["stop", "tool_calls"]]): Indicates the reason for finishing the process. 
            Defaults to None.
        message (Message): Represents the assistant's message with default attributes set to None or 
            "assistant" for role."""
        self.finish_reason: Optional[Literal["stop", "tool_calls"]] = None
        self.message = Message(
            content=None,
            tool_calls=None,
            role="assistant",
            refusal=None,
            reasoning_content=None,
        )
        self.intermediate_messages: List[Message] = []
