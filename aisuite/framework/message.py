"""Interface to hold contents of api responses when they do not confirm to the OpenAI style response"""

from pydantic import BaseModel
from typing import Literal, Optional, List


class Function(BaseModel):
    """Represents a function with its name and arguments.

    This class is a model for storing and managing information about a function,
    specifically its name and the arguments it takes.

    Attributes:
        arguments (str): A string representing the arguments of the function.
        name (str): The name of the function."""
    arguments: str
    name: str


class ChatCompletionMessageToolCall(BaseModel):
    """Represents a tool call message within a chat completion process.

    This class models a message that triggers a specific function call within
    a chat application. It contains the unique identifier of the message, the
    function to be executed, and the type of the message, which is always 
    'function'.

    Attributes:
        id (str): Unique identifier for the message.
        function (Function): The function to be executed when this message is processed.
        type (Literal["function"]): Indicates that the message is a function call."""
    id: str
    function: Function
    type: Literal["function"]


class Message(BaseModel):
    """Represents a message entity with optional content and metadata.

    Attributes:
        content (Optional[str]): The main content of the message.
        reasoning_content (Optional[str]): Additional reasoning or explanation related to the message.
        tool_calls (Optional[list[ChatCompletionMessageToolCall]]): A list of tool calls associated with the message.
        role (Optional[Literal["user", "assistant", "system"]]): The role of the message sender.
        refusal (Optional[str]): A message indicating refusal or inability to fulfill a request.

    Note:
        This class inherits from BaseModel, which provides additional functionalities for data validation and parsing."""
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
    role: Optional[Literal["user", "assistant", "system", "tool"]] = None
    refusal: Optional[str] = None
