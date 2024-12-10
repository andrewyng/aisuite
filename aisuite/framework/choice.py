from aisuite.framework.message import Message
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict


class Choice(BaseModel):
    finish_reason: Optional[Literal["stop", "tool_calls"]] = None
    message: Message = Message(
        content=None, tool_calls=None, role="assistant", refusal=None
    )

    model_config = ConfigDict(
        extra="allow",
    ) # Should we allow extra fields for Choice ?
