from aisuite.framework.choice import Choice


class ChatCompletionResponse:
    """Used to conform to the response model of OpenAI"""

    def __init__(self):
        self.choices = [Choice()]  # Adjust the range as needed for more choices
