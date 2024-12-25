from aisuite.framework.choice import Choice
from aisuite.framework.function_call import FunctionCall


class ChatCompletionResponse:
    """Used to conform to the response model of OpenAI"""

    def __init__(self):
        self.choices = [Choice()]  # Adjust the range as needed for more choices
        self.function_calls = (
            []
        )  # Initialize as empty list instead of with an empty FunctionCall
