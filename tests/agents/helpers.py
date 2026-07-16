from types import SimpleNamespace

from aisuite.framework.message import Message


def chat_response(
    content="done", intermediate_messages=None, intermediate_responses=None
):
    message = Message(role="assistant", content=content)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    if intermediate_messages is not None:
        response.choices[0].intermediate_messages = intermediate_messages
    if intermediate_responses is not None:
        response.intermediate_responses = intermediate_responses
    return response
