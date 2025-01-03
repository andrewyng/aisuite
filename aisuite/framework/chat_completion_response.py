"""
Chat completion response class.
"""
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Message:
    """Message in a chat completion."""
    role: str = ""
    content: str = ""
    function_call: Optional[dict] = None
    tool_calls: Optional[List[dict]] = None

@dataclass
class Choice:
    """Choice in a chat completion."""
    message: Message = field(default_factory=Message)
    finish_reason: Optional[str] = None
    index: int = 0
    delta: Optional[Message] = None

@dataclass
class ChatCompletionResponse:
    """Response from a chat completion."""
    choices: List[Choice] = field(default_factory=lambda: [Choice()])
    model: str = ""
    created: int = 0
    usage: Optional[dict] = None

    @classmethod
    def from_openai_response(cls, response):
        """Create a ChatCompletionResponse from an OpenAI response."""
        completion = cls()
        
        # Handle model and created time if available
        if hasattr(response, 'model'):
            completion.model = response.model
        if hasattr(response, 'created'):
            completion.created = response.created

        # Handle both streaming and non-streaming responses
        if hasattr(response, 'choices'):
            completion.choices = []
            for choice in response.choices:
                new_choice = Choice()
                new_choice.index = choice.index if hasattr(choice, 'index') else 0
                new_choice.finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else None

                # Handle streaming delta
                if hasattr(choice, 'delta'):
                    delta = Message()
                    if hasattr(choice.delta, 'content') and choice.delta.content is not None:
                        delta.content = choice.delta.content
                    if hasattr(choice.delta, 'role') and choice.delta.role is not None:
                        delta.role = choice.delta.role
                    if hasattr(choice.delta, 'function_call'):
                        delta.function_call = choice.delta.function_call
                    if hasattr(choice.delta, 'tool_calls'):
                        delta.tool_calls = choice.delta.tool_calls
                    new_choice.delta = delta
                    new_choice.message = delta
                # Handle regular message
                elif hasattr(choice, 'message'):
                    message = Message()
                    if hasattr(choice.message, 'content') and choice.message.content is not None:
                        message.content = choice.message.content
                    if hasattr(choice.message, 'role'):
                        message.role = choice.message.role
                    if hasattr(choice.message, 'function_call'):
                        message.function_call = choice.message.function_call
                    if hasattr(choice.message, 'tool_calls'):
                        message.tool_calls = choice.message.tool_calls
                    new_choice.message = message

                completion.choices.append(new_choice)

        # Handle usage information if available (only for non-streaming responses)
        if hasattr(response, 'usage') and response.usage is not None:
            completion.usage = {
                'prompt_tokens': response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0,
                'completion_tokens': response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0,
                'total_tokens': response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else 0
            }

        return completion
