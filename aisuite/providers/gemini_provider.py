# Gemini provider — Google's Gemini Developer API (API key), via the google-genai SDK.
# Distinct from the `google` provider, which targets Vertex AI (GCP project auth).
#
# Mostly a pair of converters between the OpenAI-shaped interface and Gemini's
# generateContent format. The differences they absorb:
# - The system prompt is `system_instruction` in the request config, not a message role.
# - Roles are `user`/`model`; tool results ride as `function_response` parts in a user
#   message, matched by function NAME (Gemini calls carry no ids, so ids are synthesized
#   as `call_<n>` and mapped back during conversion).
# - Tool parameter schemas are an OpenAPI 3.0 subset: unsupported JSON Schema keys
#   (`additionalProperties`, `$schema`, ...) must be stripped or the API rejects them.
# - Images arrive as OpenAI-style `image_url` parts; data URLs become `inline_data`
#   (the API cannot fetch plain http(s) URLs).

import json
import os
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

DATA_URL_RE = re.compile(
    r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.IGNORECASE | re.DOTALL
)

# Gemini finishReason → the OpenAI-shaped vocabulary. STOP maps to "tool_calls"
# instead when the turn contains function calls (Gemini has no distinct reason).
FINISH_REASON_MAPPING = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "stop",
    "RECITATION": "stop",
    "MALFORMED_FUNCTION_CALL": "stop",
}

# GenerateContentConfig keys passed through from kwargs; everything else is dropped.
CONFIG_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "stop_sequences",
}

# The OpenAPI-subset schema keys Gemini function declarations accept.
SCHEMA_KEYS = {
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "items",
    "properties",
    "required",
    "anyOf",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
    "example",
    "default",
    "title",
}


class GeminiMessageConverter:
    """Convert between OpenAI-shaped messages and Gemini contents."""

    def convert_request(self, messages):
        """OpenAI-shaped history → (system_instruction, Gemini contents).

        Consecutive same-role entries fold into one content (Gemini dislikes
        non-alternating roles), and a leading model turn gets a placeholder
        user message inserted before it.
        """
        messages = [self._as_dict(message) for message in messages]

        system_parts = []
        index = 0
        while index < len(messages) and messages[index].get("role") == "system":
            content = messages[index].get("content")
            if isinstance(content, str) and content:
                system_parts.append(content)
            index += 1

        call_names = {}
        converted = []
        for message in messages[index:]:
            role = message.get("role")
            if role == "system":
                # Defensive: a stray mid-thread system message rides as marked user text.
                text = message.get("content") or ""
                if text:
                    converted.append(
                        {
                            "role": "user",
                            "parts": [{"text": f"<system>\n{text}\n</system>"}],
                        }
                    )
            elif role == "user":
                parts = self._user_parts(message.get("content"))
                if parts:
                    converted.append({"role": "user", "parts": parts})
            elif role == "assistant":
                parts = []
                text = message.get("content")
                if isinstance(text, str) and text:
                    parts.append({"text": text})
                for call in message.get("tool_calls") or []:
                    function = call.get("function") or {}
                    name = function.get("name") or ""
                    call_names[call.get("id") or ""] = name
                    parts.append(
                        {
                            "function_call": {
                                "name": name,
                                "args": self._parse_args(function.get("arguments")),
                            }
                        }
                    )
                if parts:
                    converted.append({"role": "model", "parts": parts})
            elif role == "tool":
                call_id = message.get("tool_call_id") or ""
                converted.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": call_names.get(call_id) or call_id,
                                    "response": self._result_payload(
                                        message.get("content")
                                    ),
                                }
                            }
                        ],
                    }
                )

        folded = []
        for message in converted:
            if folded and folded[-1]["role"] == message["role"]:
                folded[-1]["parts"].extend(message["parts"])
            else:
                folded.append(message)

        if not folded:
            raise ValueError("No convertible messages for the Gemini API.")
        if folded[0]["role"] != "user":
            folded.insert(0, {"role": "user", "parts": [{"text": "(continued)"}]})

        return ("\n\n".join(system_parts) or None), folded

    @staticmethod
    def _as_dict(message):
        """Accept both dict messages and framework Message objects."""
        if isinstance(message, dict):
            return message
        converted = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
            "tool_call_id": getattr(message, "tool_call_id", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            converted["tool_calls"] = [
                call if isinstance(call, dict) else call.model_dump()
                for call in tool_calls
            ]
        return converted

    def _user_parts(self, content):
        """User content (str or OpenAI parts list) → Gemini parts."""
        if not isinstance(content, list):
            return [{"text": content}] if content else []
        parts = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind == "text":
                text = part.get("text") or ""
                if text:
                    parts.append({"text": text})
            elif kind == "image_url":
                url = (part.get("image_url") or {}).get("url") or ""
                match = DATA_URL_RE.match(url)
                if not match:
                    raise ValueError(
                        "Unsupported image_url for Gemini: the API cannot fetch "
                        "plain URLs, so images must be base64 data URLs "
                        f"(data:image/...;base64,...), got {url[:80]!r}"
                    )
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": match.group(1).lower(),
                            "data": match.group(2),
                        }
                    }
                )
            else:
                raise ValueError(f"Unsupported content part type for Gemini: {kind!r}")
        return parts

    @staticmethod
    def _parse_args(raw):
        """Tool-call arguments: dict passthrough, JSON string parse, raw fallback."""
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except (TypeError, json.JSONDecodeError):
            return {"_raw": raw}

    @staticmethod
    def _result_payload(content):
        """A tool result → the JSON object Gemini requires as a function response."""
        if isinstance(content, dict):
            return content
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"result": parsed}
        except (TypeError, json.JSONDecodeError):
            return {"result": str(content or "")}

    def convert_tool_spec(self, openai_tools):
        """OpenAI function schemas → Gemini tool declarations."""
        declarations = []
        for tool in openai_tools or []:
            if tool.get("type") != "function":
                continue
            function = tool.get("function") or {}
            entry = {"name": function.get("name") or ""}
            if function.get("description"):
                entry["description"] = function["description"]
            parameters = function.get("parameters")
            if isinstance(parameters, dict) and parameters.get("properties"):
                entry["parameters"] = self._sanitize_schema(parameters)
            # Parameter-less functions omit `parameters` (Gemini rejects empty objects).
            declarations.append(entry)
        return [{"function_declarations": declarations}] if declarations else []

    def _sanitize_schema(self, schema):
        """Strip JSON Schema keys Gemini's OpenAPI subset rejects (recursively)."""
        if not isinstance(schema, dict):
            return schema
        cleaned = {}
        for key, value in schema.items():
            if key not in SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                cleaned[key] = {
                    name: self._sanitize_schema(sub) for name, sub in value.items()
                }
            elif key == "items":
                cleaned[key] = self._sanitize_schema(value)
            elif key == "anyOf" and isinstance(value, list):
                cleaned[key] = [self._sanitize_schema(sub) for sub in value]
            else:
                cleaned[key] = value
        return cleaned

    # -- response side -------------------------------------------------------------
    @staticmethod
    def parse_candidate(response):
        """Pull (texts, function calls, finish reason) from a response or stream chunk."""
        texts, calls, finish = [], [], None
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return texts, calls, finish
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                calls.append(
                    {
                        "name": getattr(function_call, "name", "") or "",
                        "args": dict(getattr(function_call, "args", None) or {}),
                    }
                )
        raw_finish = getattr(candidate, "finish_reason", None)
        if raw_finish is not None:
            finish = getattr(raw_finish, "name", None) or str(raw_finish)
        return texts, calls, finish

    @staticmethod
    def map_finish_reason(finish, has_calls):
        if has_calls:
            return "tool_calls"
        if finish is None:
            return None
        return FINISH_REASON_MAPPING.get(finish, finish.lower())

    @staticmethod
    def get_completion_usage(usage_metadata):
        if usage_metadata is None:
            return None
        prompt = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total = getattr(usage_metadata, "total_token_count", None)
        cached = getattr(usage_metadata, "cached_content_token_count", 0) or 0
        return CompletionUsage(
            completion_tokens=completion,
            prompt_tokens=prompt,
            total_tokens=total if total is not None else prompt + completion,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
        )

    def convert_response(self, response):
        """Normalize a GenerateContentResponse to the OpenAI-shaped format."""
        texts, calls, finish = self.parse_candidate(response)
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=f"call_{i}",
                type="function",
                function=Function(
                    name=call["name"], arguments=json.dumps(call["args"])
                ),
            )
            for i, call in enumerate(calls)
        ]
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message = Message(
            content="".join(texts) or None,
            tool_calls=tool_calls or None,
            role="assistant",
            refusal=None,
        )
        normalized_response.choices[0].finish_reason = self.map_finish_reason(
            finish, bool(tool_calls)
        )
        normalized_response.usage = self.get_completion_usage(
            getattr(response, "usage_metadata", None)
        )
        return normalized_response


class GeminiProvider(Provider):
    def __init__(self, **config):
        """Initialize the Gemini provider. The API key comes from the config or the
        GEMINI_API_KEY / GOOGLE_API_KEY environment variables."""
        config.setdefault(
            "api_key", os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not config["api_key"]:
            raise ValueError(
                "Gemini API key is missing. Please provide it in the config or set "
                "the GEMINI_API_KEY environment variable."
            )
        # Lazy import so the SDK is only required when this provider is used.
        from google import genai

        self.client = genai.Client(**config)
        self.converter = GeminiMessageConverter()

    def _request_kwargs(self, model, messages, kwargs):
        system, contents = self.converter.convert_request(messages)
        kwargs = kwargs.copy()
        if "max_tokens" in kwargs and "max_output_tokens" not in kwargs:
            kwargs["max_output_tokens"] = kwargs["max_tokens"]
        if "stop" in kwargs and "stop_sequences" not in kwargs:
            stop = kwargs["stop"]
            kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        config = {key: value for key, value in kwargs.items() if key in CONFIG_KEYS}
        if system:
            config["system_instruction"] = system
        if "tools" in kwargs:
            converted = self.converter.convert_tool_spec(kwargs["tools"])
            if converted:
                config["tools"] = converted
        return {"model": model, "contents": contents, "config": config}

    def chat_completions_create(self, model, messages, **kwargs):
        """Create a chat completion using the Gemini API."""
        request = self._request_kwargs(model, messages, kwargs)
        response = self.client.models.generate_content(**request)
        return self.converter.convert_response(response)

    async def achat_completions_create(self, model, messages, **kwargs):
        """Async chat completion via the SDK's native async client."""
        request = self._request_kwargs(model, messages, kwargs)
        response = await self.client.aio.models.generate_content(**request)
        return self.converter.convert_response(response)

    def chat_completions_create_stream(self, model, messages, **kwargs):
        """Stream a chat completion as OpenAI-shaped chunks."""
        request = self._request_kwargs(model, messages, kwargs)
        state = _StreamState()
        yield state.role_chunk()
        for chunk in self.client.models.generate_content_stream(**request):
            for out in state.convert(chunk, self.converter):
                yield out
        yield state.final_chunk(self.converter)

    async def achat_completions_create_stream(self, model, messages, **kwargs):
        """Stream a chat completion natively async, as OpenAI-shaped chunks."""
        request = self._request_kwargs(model, messages, kwargs)
        state = _StreamState()
        yield state.role_chunk()
        stream = await self.client.aio.models.generate_content_stream(**request)
        async for chunk in stream:
            for out in state.convert(chunk, self.converter):
                yield out
        yield state.final_chunk(self.converter)


class _StreamState:
    """Fold Gemini stream chunks into OpenAI-shaped chunks.

    Unlike Anthropic, function_call parts arrive whole (args are a complete dict
    per part), so each becomes one complete tool-call fragment — no accumulation."""

    def __init__(self):
        self.call_index = 0
        self.finish = None
        self.usage_metadata = None

    @staticmethod
    def role_chunk():
        return ChatCompletionChunk(
            choices=[StreamChoice(delta=ChoiceDelta(role="assistant"))]
        )

    def convert(self, chunk, converter):
        texts, calls, finish = converter.parse_candidate(chunk)
        for text in texts:
            yield ChatCompletionChunk(
                choices=[StreamChoice(delta=ChoiceDelta(content=text))]
            )
        for call in calls:
            yield ChatCompletionChunk(
                choices=[
                    StreamChoice(
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=self.call_index,
                                    id=f"call_{self.call_index}",
                                    type="function",
                                    function=ChoiceDeltaFunction(
                                        name=call["name"],
                                        arguments=json.dumps(call["args"]),
                                    ),
                                )
                            ]
                        )
                    )
                ]
            )
            self.call_index += 1
        if finish:
            self.finish = finish
        metadata = getattr(chunk, "usage_metadata", None)
        if metadata is not None:
            self.usage_metadata = metadata

    def final_chunk(self, converter):
        return ChatCompletionChunk(
            choices=[
                StreamChoice(
                    delta=ChoiceDelta(),
                    finish_reason=converter.map_finish_reason(
                        self.finish, self.call_index > 0
                    ),
                )
            ],
            usage=converter.get_completion_usage(self.usage_metadata),
        )
