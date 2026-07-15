# Anthropic provider
# Links:
# Tool calling docs - https://docs.anthropic.com/en/docs/build-with-claude/tool-use

import anthropic
import json
import re
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaFunction,
    ChoiceDeltaToolCall,
    StreamChoice,
)
from aisuite.framework.message import (
    Message,
    ChatCompletionMessageToolCall,
    Function,
    CompletionUsage,
    PromptTokensDetails,
)

# Define a constant for the default max_tokens value
DEFAULT_MAX_TOKENS = 4096

# OpenAI-style image_url parts carry either a data URL or a plain http(s) URL.
DATA_URL_RE = re.compile(
    r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.IGNORECASE | re.DOTALL
)


class AnthropicMessageConverter:
    # Role constants
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_TOOL = "tool"
    ROLE_SYSTEM = "system"

    # Finish reason mapping
    FINISH_REASON_MAPPING = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }

    def convert_request(self, messages):
        """Convert framework messages to Anthropic format."""
        system_message = self._extract_system_message(messages)
        converted_messages = [self._convert_single_message(msg) for msg in messages]
        return system_message, converted_messages

    def convert_response(self, response):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].finish_reason = self._get_finish_reason(response)
        normalized_response.usage = self._get_completion_usage(response)
        normalized_response.choices[0].message = self._get_message(response)
        return normalized_response

    def convert_stream_event(self, event, state):
        """Normalize one Anthropic stream event into an OpenAI-shaped chunk.

        Returns ``None`` for events with nothing to surface (block stops,
        pings, thinking deltas). ``state`` is a plain dict owned by the caller,
        carried across one message's events: it maps Anthropic content-block
        indices to OpenAI tool-call indices (Anthropic counts every content
        block, OpenAI counts only tool calls) and holds the prompt-token counts
        reported at message start until usage is emitted on the final chunk.
        """
        event_type = getattr(event, "type", None)

        if event_type == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            state["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
            state["cached_tokens"] = getattr(usage, "cache_read_input_tokens", 0) or 0
            return self._delta_chunk(ChoiceDelta(role=self.ROLE_ASSISTANT))

        if event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) != "tool_use":
                return None
            positions = state.setdefault("tool_positions", {})
            position = positions.setdefault(getattr(event, "index", 0), len(positions))
            return self._delta_chunk(
                ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=position,
                            id=block.id,
                            type="function",
                            function=ChoiceDeltaFunction(name=block.name, arguments=""),
                        )
                    ]
                )
            )

        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                text = getattr(delta, "text", "") or ""
                return self._delta_chunk(ChoiceDelta(content=text)) if text else None
            if delta_type == "input_json_delta":
                position = state.get("tool_positions", {}).get(
                    getattr(event, "index", None)
                )
                partial = getattr(delta, "partial_json", "") or ""
                if position is None or not partial:
                    return None
                return self._delta_chunk(
                    ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=position,
                                function=ChoiceDeltaFunction(arguments=partial),
                            )
                        ]
                    )
                )
            return None

        if event_type == "message_delta":
            stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
            output_tokens = getattr(
                getattr(event, "usage", None), "output_tokens", None
            )
            if stop_reason is None and output_tokens is None:
                return None
            usage = None
            if output_tokens is not None:
                prompt_tokens = state.get("input_tokens", 0)
                usage = CompletionUsage(
                    completion_tokens=output_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens + output_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=state.get("cached_tokens", 0),
                    ),
                )
            finish_reason = (
                self.FINISH_REASON_MAPPING.get(stop_reason, "stop")
                if stop_reason
                else None
            )
            return ChatCompletionChunk(
                choices=[
                    StreamChoice(delta=ChoiceDelta(), finish_reason=finish_reason)
                ],
                usage=usage,
            )

        return None

    @staticmethod
    def _delta_chunk(delta):
        """Wrap a delta in a single-choice chunk with no finish reason."""
        return ChatCompletionChunk(choices=[StreamChoice(delta=delta)])

    def _convert_single_message(self, msg):
        """Convert a single message to Anthropic format."""
        if isinstance(msg, dict):
            return self._convert_dict_message(msg)
        return self._convert_message_object(msg)

    def _convert_dict_message(self, msg):
        """Convert a dictionary message to Anthropic format."""
        if msg["role"] == self.ROLE_TOOL:
            return self._create_tool_result_message(msg["tool_call_id"], msg["content"])
        elif msg["role"] == self.ROLE_ASSISTANT and "tool_calls" in msg:
            return self._create_assistant_tool_message(
                msg["content"], msg["tool_calls"]
            )
        return {"role": msg["role"], "content": self._convert_content(msg["content"])}

    def _convert_message_object(self, msg):
        """Convert a Message object to Anthropic format."""
        if msg.role == self.ROLE_TOOL:
            return self._create_tool_result_message(msg.tool_call_id, msg.content)
        elif msg.role == self.ROLE_ASSISTANT and msg.tool_calls:
            return self._create_assistant_tool_message(msg.content, msg.tool_calls)
        return {"role": msg.role, "content": self._convert_content(msg.content)}

    def _convert_content(self, content):
        """String content passes through unchanged; an OpenAI-style parts list
        (``{"type": "text"}`` / ``{"type": "image_url"}``) becomes Anthropic
        content blocks. Empty text parts are dropped — Anthropic rejects them."""
        if not isinstance(content, list):
            return content
        blocks = []
        for part in content:
            block = self._convert_content_part(part)
            if block is not None:
                blocks.append(block)
        return blocks

    def _convert_content_part(self, part):
        """One OpenAI-style content part → an Anthropic content block (or None)."""
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            text = part.get("text") or ""
            return {"type": "text", "text": text} if text else None
        if kind == "image_url":
            url = (part.get("image_url") or {}).get("url") or ""
            match = DATA_URL_RE.match(url)
            if match:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": match.group(1).lower(),
                        "data": match.group(2),
                    },
                }
            if url.startswith(("http://", "https://")):
                return {"type": "image", "source": {"type": "url", "url": url}}
            raise ValueError(
                "Unsupported image_url for Anthropic: expected a base64 data URL "
                f"(data:image/...;base64,...) or an http(s) URL, got {url[:80]!r}"
            )
        raise ValueError(f"Unsupported content part type for Anthropic: {kind!r}")

    def _create_tool_result_message(self, tool_call_id, content):
        """Create a tool result message in Anthropic format."""
        return {
            "role": self.ROLE_USER,
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content,
                }
            ],
        }

    def _create_assistant_tool_message(self, content, tool_calls):
        """Create an assistant message with tool calls in Anthropic format."""
        message_content = []
        if content:
            message_content.append({"type": "text", "text": content})

        for tool_call in tool_calls:
            tool_input = (
                tool_call["function"]["arguments"]
                if isinstance(tool_call, dict)
                else tool_call.function.arguments
            )
            message_content.append(
                {
                    "type": "tool_use",
                    "id": (
                        tool_call["id"] if isinstance(tool_call, dict) else tool_call.id
                    ),
                    "name": (
                        tool_call["function"]["name"]
                        if isinstance(tool_call, dict)
                        else tool_call.function.name
                    ),
                    "input": json.loads(tool_input),
                }
            )

        return {"role": self.ROLE_ASSISTANT, "content": message_content}

    def _extract_system_message(self, messages):
        """Extract system message if present, otherwise return empty list."""
        # TODO: This is a temporary solution to extract the system message.
        # User can pass multiple system messages, which can mingled with other messages.
        # This needs to be fixed to handle this case.
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages.pop(0)
            return system_message
        return []

    def _get_finish_reason(self, response):
        """Get the normalized finish reason."""
        return self.FINISH_REASON_MAPPING.get(response.stop_reason, "stop")

    def _get_completion_usage(self, response):
        """Get the usage statistics."""
        return CompletionUsage(
            completion_tokens=response.usage.output_tokens,
            prompt_tokens=response.usage.input_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            prompt_tokens_details=PromptTokensDetails(
                cached_tokens=response.usage.cache_read_input_tokens,
            ),
        )

    def _get_message(self, response):
        """Get the appropriate message based on response type."""
        # Check if response contains any tool use blocks (regardless of stop_reason)
        has_tool_use = any(content.type == "tool_use" for content in response.content)

        if has_tool_use:
            tool_message = self.convert_response_with_tool_use(response)
            if tool_message:
                return tool_message

        # Safely extract text content from any position in content blocks
        text_content = next(
            (content.text for content in response.content if content.type == "text"),
            "",
        )

        return Message(
            content=text_content or None,
            role="assistant",
            tool_calls=None,
            refusal=None,
        )

    def convert_response_with_tool_use(self, response):
        """Convert Anthropic tool use response to the framework's format."""
        tool_call = next(
            (content for content in response.content if content.type == "tool_use"),
            None,
        )

        if tool_call:
            function = Function(
                name=tool_call.name, arguments=json.dumps(tool_call.input)
            )
            tool_call_obj = ChatCompletionMessageToolCall(
                id=tool_call.id, function=function, type="function"
            )
            text_content = next(
                (
                    content.text
                    for content in response.content
                    if content.type == "text"
                ),
                "",
            )

            return Message(
                content=text_content or None,
                tool_calls=[tool_call_obj] if tool_call else None,
                role="assistant",
                refusal=None,
            )
        return None

    def convert_tool_spec(self, openai_tools):
        """Convert OpenAI tool specification to Anthropic format."""
        anthropic_tools = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                continue

            function = tool["function"]
            anthropic_tool = {
                "name": function["name"],
                "description": function["description"],
                "input_schema": {
                    "type": "object",
                    "properties": function["parameters"]["properties"],
                    "required": function["parameters"].get("required", []),
                },
            }
            anthropic_tools.append(anthropic_tool)

        return anthropic_tools


class AnthropicProvider(Provider):
    def __init__(self, **config):
        """Initialize the Anthropic provider with the given configuration."""
        self.client = anthropic.Anthropic(**config)
        # Async client shares the same config for native async streaming.
        self.async_client = anthropic.AsyncAnthropic(**config)
        self.converter = AnthropicMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """Create a chat completion using the Anthropic API."""
        kwargs = self._prepare_kwargs(kwargs)
        system_message, converted_messages = self.converter.convert_request(messages)

        response = self.client.messages.create(
            model=model, system=system_message, messages=converted_messages, **kwargs
        )
        return self.converter.convert_response(response)

    def chat_completions_create_stream(self, model, messages, **kwargs):
        """Stream a chat completion as OpenAI-shaped chunks."""
        kwargs = self._prepare_kwargs(kwargs)
        system_message, converted_messages = self.converter.convert_request(messages)

        events = self.client.messages.create(
            model=model,
            system=system_message,
            messages=converted_messages,
            stream=True,
            **kwargs,
        )
        state = {}
        for event in events:
            chunk = self.converter.convert_stream_event(event, state)
            if chunk is not None:
                yield chunk

    async def achat_completions_create_stream(self, model, messages, **kwargs):
        """Stream a chat completion natively async, as OpenAI-shaped chunks."""
        kwargs = self._prepare_kwargs(kwargs)
        system_message, converted_messages = self.converter.convert_request(messages)

        events = await self.async_client.messages.create(
            model=model,
            system=system_message,
            messages=converted_messages,
            stream=True,
            **kwargs,
        )
        state = {}
        async for event in events:
            chunk = self.converter.convert_stream_event(event, state)
            if chunk is not None:
                yield chunk

    def _prepare_kwargs(self, kwargs):
        """Prepare kwargs for the API call."""
        kwargs = kwargs.copy()
        kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)

        if "tools" in kwargs:
            kwargs["tools"] = self.converter.convert_tool_spec(kwargs["tools"])

        return kwargs
