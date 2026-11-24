from __future__ import annotations

from enum import Enum
from typing import Annotated

import pytest
from pydantic import Field

from aisuite.utils.tools import Tools


class TravelMode(str, Enum):
    WALK = "walk"
    BIKE = "bike"


def choose_route(mode: TravelMode) -> str:
    """Choose a route using a postponed enum annotation."""
    return mode.value


def search_city(city: Annotated[str, Field(min_length=2)]) -> str:
    """Search using a postponed Annotated parameter."""
    return city


def test_postponed_enum_annotation_is_resolved_in_the_tool_module():
    tools = Tools([choose_route])
    schema = tools.tools()[0]["function"]["parameters"]

    assert schema["properties"]["mode"]["enum"] == ["walk", "bike"]
    assert tools.execute(
        {"function": {"name": "choose_route", "arguments": {"mode": "bike"}}}
    ) == ["bike"]


def test_postponed_annotated_parameter_keeps_its_constraints():
    tools = Tools([search_city])
    schema = tools.tools()[0]["function"]["parameters"]

    assert schema["properties"]["city"]["minLength"] == 2
    with pytest.raises(ValueError, match="parameters"):
        tools.execute({"function": {"name": "search_city", "arguments": {"city": "x"}}})
