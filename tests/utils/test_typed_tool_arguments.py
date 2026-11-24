import json

import pytest
from pydantic import BaseModel

from aisuite.utils.tools import Tools


class Destination(BaseModel):
    city: str


def describe_destination(destination: Destination) -> str:
    """Describe a destination."""
    return destination.city.upper()


async def describe_destination_async(destination: Destination) -> str:
    """Describe a destination asynchronously."""
    return destination.city.upper()


def tool_call(name: str) -> dict:
    return {
        "id": "call_1",
        "function": {
            "name": name,
            "arguments": json.dumps({"destination": {"city": "Paris"}}),
        },
    }


def test_execute_preserves_validated_nested_models():
    tools = Tools([describe_destination])

    assert tools.execute(tool_call("describe_destination")) == ["PARIS"]


def test_execute_tool_preserves_models_and_serialized_policy_arguments():
    tools = Tools([describe_destination])
    policy_arguments = []

    def policy(context):
        policy_arguments.append(context.arguments)
        return True

    results, messages = tools.execute_tool(
        tool_call("describe_destination"), tool_policy=policy
    )

    assert results == ["PARIS"]
    assert json.loads(messages[0]["content"]) == "PARIS"
    assert policy_arguments == [{"destination": {"city": "Paris"}}]
    assert tools.last_tool_events[0]["arguments"] == {"destination": {"city": "Paris"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [describe_destination, describe_destination_async])
async def test_async_execute_tool_preserves_validated_nested_models(tool):
    tools = Tools([tool])

    results, messages = await tools.aexecute_tool(tool_call(tool.__name__))

    assert results == ["PARIS"]
    assert json.loads(messages[0]["content"]) == "PARIS"
