"""Defines the ChatCompletionResponse class."""

from typing import Optional

from aisuite.framework.choice import Choice
from aisuite.framework.message import CompletionUsage


# pylint: disable=too-few-public-methods
class ChatCompletionResponse:
    """Used to conform to the response model of OpenAI."""

    def __init__(self):
        """Initializes the ChatCompletionResponse."""
        self.choices = [Choice()]  # Adjust the range as needed for more choices
        self.usage: Optional[CompletionUsage] = None
