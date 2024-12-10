from aisuite.framework.choice import Choice
from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional, List


class ChatCompletionResponse(BaseModel):
    """Used to conform to the response model of OpenAI"""

    choices: List[Choice] = [Choice()]  # Adjust the range as needed for more choices

    model_config = ConfigDict(
        extra="allow",
    ) # Some providers return some extra fields in reponse, like Antrophic, wich returns `usage`
