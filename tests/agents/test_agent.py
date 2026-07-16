from aisuite import Agent


def test_agent_stores_definition_fields():
    def tool(city: str) -> str:
        """Lookup city."""
        return city

    agent = Agent(
        name="weather",
        model="openai:gpt-4o",
        instructions="Answer briefly.",
        tools=[tool],
        model_settings={"temperature": 0.2},
        tags=["prod"],
        metadata={"team": "growth"},
    )

    assert agent.name == "weather"
    assert agent.model == "openai:gpt-4o"
    assert agent.instructions == "Answer briefly."
    assert agent.tools == [tool]
    assert agent.model_settings == {"temperature": 0.2}
    assert agent.tags == ["prod"]
    assert agent.metadata == {"team": "growth"}


def test_agent_mutable_defaults_are_independent():
    first = Agent(name="first", model="openai:gpt-4o")
    second = Agent(name="second", model="openai:gpt-4o")

    first.tools.append(lambda value: value)
    first.tags.append("one")
    first.metadata["team"] = "one"

    assert second.tools == []
    assert second.tags == []
    assert second.metadata == {}
