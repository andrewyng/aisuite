"""Interface to hold contents of api responses when they do not confirm to the OpenAI style response"""

from pydantic import BaseModel
from typing import Literal, Optional, List


class Function(BaseModel):
    arguments: str
    name: str


class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"]


class Message(BaseModel):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    role: Optional[Literal["user", "assistant", "system", "tool"]] = None
    refusal: Optional[str] = None
