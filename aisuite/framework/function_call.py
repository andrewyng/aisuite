from typing import Dict, Any
from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str = Field(..., description="The name of the function to call")
    arguments: Dict[str, Any] = Field(
        ..., description="The arguments to pass to the function"
    )
