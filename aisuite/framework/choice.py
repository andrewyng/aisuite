from aisuite.framework.message import Message
from typing import Literal, Optional, List


class Choice:
    def __init__(self):
        self.finish_reason: Optional[Literal["stop", "tool_calls"]] = None
        self.message = Message(
            content=None,
            tool_calls=None,
            role="assistant",
            refusal=None,
            reasoning_content=None,
        )
        self.intermediate_messages: List[Message] = []
