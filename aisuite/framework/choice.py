from aisuite.framework.message import Message
from typing import Literal, Optional, List


class Choice():

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

    def __repr_name__(self) -> str:
        """Name of the instance's class, used in __repr__."""
        return self.__class__.__name__

    def __repr_args__(self):
        attrs_names = self.__slots__ if hasattr(self, '__slots__') else None
        if not attrs_names and hasattr(self, '__dict__'):
            attrs_names = self.__dict__.keys()
        attrs = ((s, getattr(self, s)) for s in attrs_names)
        return [(a, v) for a, v in attrs]

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join(repr(v) if a is None else f'{a}={v!r}' for a, v in self.__repr_args__())

    def __repr__(self):
        return f'{self.__repr_name__()}({self.__repr_str__(", ")})'
