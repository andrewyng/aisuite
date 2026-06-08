from .provider import ProviderFactory
import os
from .utils.tools import Tools
from typing import Union, BinaryIO, Optional, Any, Literal
from contextlib import ExitStack
from .framework.message import (
    TranscriptionResponse,
)
from .framework.asr_params import ParamValidator
from .tracing.normalize import normalize_model_input, normalize_model_response
from .tracing.sinks import TraceEvent, emit_event

# Import MCP utilities for config dict support
try:
    from .mcp.config import is_mcp_config
    from .mcp.client import MCPClient

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class Client:
    def __init__(
        self,
        provider_configs: dict = {},
        extra_param_mode: Literal["strict", "warn", "permissive"] = "warn",
    ):
        """
        Initialize the client with provider configurations.
        Use the ProviderFactory to create provider instances.

        Args:
            provider_configs (dict): A dictionary containing provider configurations.
                Each key should be a provider string (e.g., "google" or "aws-bedrock"),
                and the value should be a dictionary of configuration options for that provider.
                For example:
                {
                    "openai": {"api_key": "your_openai_api_key"},
                    "aws-bedrock": {
                        "aws_access_key": "your_aws_access_key",
                        "aws_secret_key": "your_aws_secret_key",
                        "aws_region": "us-west-2"
                    }
                }
            extra_param_mode (str): How to handle unknown ASR parameters.
                - "strict": Raise ValueError on unknown params (production)
                - "warn": Log warning on unknown params (default, development)
                - "permissive": Allow all params without validation (testing)
        """
        self.providers = {}
        self.provider_configs = provider_configs
        self.extra_param_mode = extra_param_mode
        self.param_validator = ParamValidator(extra_param_mode)
        self._chat = None
        self._audio = None

    def _initialize_providers(self):
        """Helper method to initialize or update providers."""
        for provider_key, config in self.provider_configs.items():
            provider_key = self._validate_provider_key(provider_key)
            self.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )

    def _validate_provider_key(self, provider_key):
        """
        Validate if the provider key corresponds to a supported provider.
        """
        supported_providers = ProviderFactory.get_supported_providers()

        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Make sure the model string is formatted correctly as 'provider:model'."
            )

        return provider_key

    def configure(self, provider_configs: Optional[dict] = None):
        """
        Configure the client with provider configurations.
        """
        if provider_configs is None:
            return

        self.provider_configs.update(provider_configs)
        # Providers will be lazily initialized when needed

    @property
    def chat(self):
        """Return the chat API interface."""
        if not self._chat:
            self._chat = Chat(self)
        return self._chat

    @property
    def audio(self):
        """Return the audio API interface."""
        if not self._audio:
            self._audio = Audio(self)
        return self._audio


class Chat:
    def __init__(self, client: "Client"):
        self.client = client
        self._completions = Completions(self.client)

    @property
    def completions(self):
        """Return the completions interface."""
        return self._completions


class Completions:
    def __init__(self, client: "Client"):
        self.client = client

    def _active_trace_context(self):
        from .agents.context import get_active_run_context

        return get_active_run_context()

    def _emit_model_event(self, event_type, data):
        context = self._active_trace_context()
        if not context or not context.trace_sinks or not context.trace_id:
            return
        emit_event(
            context.trace_sinks,
            TraceEvent(
                event_type=event_type,
                trace_id=context.trace_id,
                agent_name=context.agent_name,
                run_name=context.run_name,
                parent_run_id=context.parent_run_id,
                group_id=context.group_id,
                tags=list(context.tags),
                metadata=dict(context.metadata),
                data=data,
            ),
        )

    def _process_mcp_configs(self, tools: list) -> tuple[list, list]:
        """
        Process tools list and convert MCP config dicts to callable tools.

        This method:
        1. Detects MCP config dicts ({"type": "mcp", ...})
        2. Creates MCPClient instances from configs
        3. Extracts callable tools with filtering and prefixing
        4. Mixes MCP tools with regular callable tools
        5. Returns both processed tools and MCP clients for cleanup

        Args:
            tools: List of tools (mix of callables and MCP configs)

        Returns:
            Tuple of (processed_tools, mcp_clients):
                - processed_tools: List of callable tools only
                - mcp_clients: List of MCPClient instances to be cleaned up

        Example:
            >>> tools = [
            ...     my_function,
            ...     {"type": "mcp", "name": "fs", "command": "npx", "args": [...]},
            ...     another_function
            ... ]
            >>> callable_tools, mcp_clients = self._process_mcp_configs(tools)
            >>> # Returns: ([my_function, fs_tool1, fs_tool2, ..., another_function], [mcp_client])
        """
        if not MCP_AVAILABLE:
            # If MCP not installed, check if user is trying to use it
            if any(is_mcp_config(tool) for tool in tools if isinstance(tool, dict)):
                raise ImportError(
                    "MCP tools require the 'mcp' package. "
                    "Install it with: pip install 'aisuite[mcp]' or pip install mcp"
                )
            return tools, []

        processed_tools = []
        mcp_clients = []

        for tool in tools:
            if isinstance(tool, dict) and is_mcp_config(tool):
                # It's an MCP config dict - convert to callable tools
                try:
                    mcp_client = MCPClient.from_config(tool)
                    mcp_clients.append(mcp_client)

                    # Get tools with config settings
                    mcp_tools = mcp_client.get_callable_tools(
                        allowed_tools=tool.get("allowed_tools"),
                        use_tool_prefix=tool.get("use_tool_prefix", False),
                    )

                    processed_tools.extend(mcp_tools)
                except Exception as e:
                    raise ValueError(
                        f"Failed to create MCP client from config: {e}\n"
                        f"Config: {tool}"
                    )
            else:
                # Regular callable tool - pass through
                processed_tools.append(tool)

        return processed_tools, mcp_clients

    def _extract_thinking_content(self, response):
        """
        Extract content between <think> tags if present and store it in reasoning_content.

        Args:
            response: The response object from the provider

        Returns:
            Modified response object
        """
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "content") and message.content:
                content = message.content.strip()
                if content.startswith("<think>") and "</think>" in content:
                    # Extract content between think tags
                    start_idx = len("<think>")
                    end_idx = content.find("</think>")
                    thinking_content = content[start_idx:end_idx].strip()

                    # Store the thinking content
                    message.reasoning_content = thinking_content

                    # Remove the think tags from the original content
                    message.content = content[end_idx + len("</think>") :].strip()

        return response

    def _init_tool_runner(self, tools, kwargs):
        """Validate/convert tools and set OpenAI-format specs on kwargs."""
        if isinstance(tools, Tools):
            tools_instance = tools
        else:
            if not all(callable(tool) for tool in tools):
                raise ValueError("One or more tools is not callable")
            tools_instance = Tools(tools)
        kwargs["tools"] = tools_instance.tools()
        return tools_instance

    def _emit_model_send(self, messages, model_identifier):
        self._emit_model_event(
            "model.send",
            normalize_model_input(messages, model=model_identifier),
        )

    def _emit_model_error(self, exc, model_identifier):
        self._emit_model_event(
            "model.error",
            {
                "model": model_identifier,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )

    def _handle_model_response(self, response, model_identifier):
        response = self._extract_thinking_content(response)
        self._emit_model_event(
            "model.response",
            normalize_model_response(response, model=model_identifier),
        )
        return response

    @staticmethod
    def _response_tool_calls(response):
        return (
            getattr(response.choices[0].message, "tool_calls", None)
            if hasattr(response, "choices")
            else None
        )

    def _finalize_runner_response(
        self,
        response,
        intermediate_responses,
        intermediate_messages,
        tool_policy_events,
        tool_events,
    ):
        # Exclude the final response from the intermediate list.
        response.intermediate_responses = intermediate_responses[:-1]
        response.choices[0].intermediate_messages = intermediate_messages
        response.tool_policy_events = tool_policy_events
        response.tool_events = tool_events
        response.tool_events_emitted = self._active_trace_context() is not None
        return response

    def _tool_runner(
        self,
        provider,
        model_name: str,
        model_identifier: str,
        messages: list,
        tools: Any,
        max_turns: int,
        tool_policy=None,
        tool_policy_context=None,
        **kwargs,
    ):
        """
        Handle tool execution loop for max_turns iterations.

        Args:
            provider: The provider instance to use for completions
            model_name: Name of the model to use
            messages: List of conversation messages
            tools: Tools instance or list of callable tools
            max_turns: Maximum number of tool execution turns
            **kwargs: Additional arguments to pass to the provider

        Returns:
            The final response from the model with intermediate responses and messages
        """
        tools_instance = self._init_tool_runner(tools, kwargs)

        turns = 0
        intermediate_responses = []  # Store intermediate responses
        intermediate_messages = []  # Store all messages including tool interactions
        tool_policy_events = []
        tool_events = []

        while turns < max_turns:
            self._emit_model_send(messages, model_identifier)
            try:
                response = provider.chat_completions_create(
                    model_name, messages, **kwargs
                )
            except Exception as exc:
                self._emit_model_error(exc, model_identifier)
                raise
            response = self._handle_model_response(response, model_identifier)

            intermediate_responses.append(response)
            tool_calls = self._response_tool_calls(response)
            intermediate_messages.append(response.choices[0].message)

            if not tool_calls:
                return self._finalize_runner_response(
                    response,
                    intermediate_responses,
                    intermediate_messages,
                    tool_policy_events,
                    tool_events,
                )

            results, tool_messages = tools_instance.execute_tool(
                tool_calls,
                tool_policy=tool_policy,
                tool_policy_context=tool_policy_context,
            )
            tool_policy_events.extend(getattr(tools_instance, "last_policy_events", []))
            tool_events.extend(getattr(tools_instance, "last_tool_events", []))
            intermediate_messages.extend(tool_messages)
            messages.extend([response.choices[0].message, *tool_messages])
            turns += 1

        return self._finalize_runner_response(
            response,
            intermediate_responses,
            intermediate_messages,
            tool_policy_events,
            tool_events,
        )

    async def _atool_runner(
        self,
        provider,
        model_name: str,
        model_identifier: str,
        messages: list,
        tools: Any,
        max_turns: int,
        tool_policy=None,
        tool_policy_context=None,
        **kwargs,
    ):
        """Async variant of ``_tool_runner``.

        Awaits the provider's async completion and the async tool execution.
        Event emission, response handling, and bookkeeping are shared with the
        sync path via helper methods.
        """
        tools_instance = self._init_tool_runner(tools, kwargs)

        turns = 0
        intermediate_responses = []
        intermediate_messages = []
        tool_policy_events = []
        tool_events = []

        while turns < max_turns:
            self._emit_model_send(messages, model_identifier)
            try:
                response = await provider.achat_completions_create(
                    model_name, messages, **kwargs
                )
            except Exception as exc:
                self._emit_model_error(exc, model_identifier)
                raise
            response = self._handle_model_response(response, model_identifier)

            intermediate_responses.append(response)
            tool_calls = self._response_tool_calls(response)
            intermediate_messages.append(response.choices[0].message)

            if not tool_calls:
                return self._finalize_runner_response(
                    response,
                    intermediate_responses,
                    intermediate_messages,
                    tool_policy_events,
                    tool_events,
                )

            results, tool_messages = await tools_instance.aexecute_tool(
                tool_calls,
                tool_policy=tool_policy,
                tool_policy_context=tool_policy_context,
            )
            tool_policy_events.extend(getattr(tools_instance, "last_policy_events", []))
            tool_events.extend(getattr(tools_instance, "last_tool_events", []))
            intermediate_messages.extend(tool_messages)
            messages.extend([response.choices[0].message, *tool_messages])
            turns += 1

        return self._finalize_runner_response(
            response,
            intermediate_responses,
            intermediate_messages,
            tool_policy_events,
            tool_events,
        )

    def _resolve_provider(self, model: str):
        """Validate the model string and return ``(provider, model_name)``."""
        # Check that correct format is used
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'provider:model', got '{model}'"
            )

        # Extract the provider key from the model identifier, e.g., "google:gemini-xx"
        provider_key, model_name = model.split(":", 1)

        # Validate if the provider is supported
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. Supported providers: {supported_providers}. "
                "Make sure the model string is formatted correctly as 'provider:model'."
            )

        # Initialize provider if not already initialized
        # TODO: Add thread-safe provider initialization with lock to prevent race conditions
        # when multiple threads try to initialize the same provider simultaneously.
        if provider_key not in self.client.providers:
            config = self.client.provider_configs.get(provider_key, {})
            self.client.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")
        return provider, model_name

    def create(self, model: str, messages: list, **kwargs):
        """
        Create chat completion based on the model, messages, and any extra arguments.
        Supports automatic tool execution when max_turns is specified.
        """
        provider, model_name = self._resolve_provider(model)

        # Extract tool-related parameters
        max_turns = kwargs.pop("max_turns", None)
        tools = kwargs.pop("tools", None)
        tool_policy = kwargs.pop("tool_policy", None)
        tool_policy_context = kwargs.pop("tool_policy_context", None)

        # Use ExitStack to manage MCP client cleanup automatically
        with ExitStack() as stack:
            # Convert MCP config dicts to callable tools and get MCP clients
            mcp_clients = []
            if tools is not None:
                tools, mcp_clients = self._process_mcp_configs(tools)
                # Register all MCP clients for automatic cleanup
                for mcp_client in mcp_clients:
                    stack.enter_context(mcp_client)

            # Check environment variable before allowing multi-turn tool execution
            if max_turns is not None and tools is not None:
                return self._tool_runner(
                    provider,
                    model_name,
                    model,
                    messages.copy(),
                    tools,
                    max_turns,
                    tool_policy=tool_policy,
                    tool_policy_context=tool_policy_context,
                    **kwargs,
                )

            # Manual tool calling (no max_turns): the provider must still see the tool
            # schemas, so re-add the processed tools to kwargs — schema dicts pass
            # through as-is, callables (including MCP-derived ones) become OpenAI-format
            # specs. Regression guard: a plain `kwargs.pop("tools")` here used to drop
            # them entirely (#266).
            if tools is not None:
                converted = []
                for tool in tools:
                    if callable(tool):
                        converted.extend(Tools(tools=[tool]).tools())
                    else:
                        converted.append(tool)
                kwargs["tools"] = converted

            # Delegate the chat completion to the correct provider's implementation
            response = provider.chat_completions_create(model_name, messages, **kwargs)
            return self._extract_thinking_content(response)

    async def acreate(self, model: str, messages: list, **kwargs):
        """Async variant of ``create``.

        Awaits the provider's async completion and, when ``max_turns`` and
        ``tools`` are supplied, the async tool-execution loop. MCP client
        cleanup remains synchronous and is handled by the ExitStack.
        """
        provider, model_name = self._resolve_provider(model)

        max_turns = kwargs.pop("max_turns", None)
        tools = kwargs.pop("tools", None)
        tool_policy = kwargs.pop("tool_policy", None)
        tool_policy_context = kwargs.pop("tool_policy_context", None)

        with ExitStack() as stack:
            mcp_clients = []
            if tools is not None:
                tools, mcp_clients = self._process_mcp_configs(tools)
                for mcp_client in mcp_clients:
                    stack.enter_context(mcp_client)

            if max_turns is not None and tools is not None:
                return await self._atool_runner(
                    provider,
                    model_name,
                    model,
                    messages.copy(),
                    tools,
                    max_turns,
                    tool_policy=tool_policy,
                    tool_policy_context=tool_policy_context,
                    **kwargs,
                )

            response = await provider.achat_completions_create(
                model_name, messages, **kwargs
            )
            return self._extract_thinking_content(response)


class Audio:
    """Audio API interface."""

    def __init__(self, client: "Client"):
        self.client = client
        self._transcriptions = Transcriptions(self.client)

    @property
    def transcriptions(self):
        """Return the transcriptions interface."""
        return self._transcriptions


class Transcriptions:
    """Transcriptions API interface."""

    def __init__(self, client: "Client"):
        self.client = client

    def create(
        self,
        *,
        model: str,
        file: Union[str, BinaryIO],
        **kwargs,
    ) -> TranscriptionResponse:
        """
        Create audio transcription with parameter validation.

        This method uses a pass-through approach with validation:
        - Common parameters (OpenAI-style) are auto-mapped to provider equivalents
        - Provider-specific parameters are passed through directly
        - Unknown parameters are handled based on extra_param_mode

        Args:
            model: Provider and model in format 'provider:model' (e.g., 'openai:whisper-1')
            file: Audio file to transcribe (file path or file-like object)
            **kwargs: Transcription parameters (provider-specific or common)
                Common parameters (portable across providers):
                    - language: Language code (e.g., "en")
                    - prompt: Context for the transcription
                    - temperature: Sampling temperature (0-1, OpenAI only)
                Provider-specific parameters are passed through directly.
                See provider documentation for valid parameters.

        Returns:
            TranscriptionResponse: Unified response (batch or streaming)

        Raises:
            ValueError: If model format invalid, provider not supported,
                       or unknown params in strict mode

        Examples:
            # Portable code (OpenAI-style params)
            >>> result = client.audio.transcriptions.create(
            ...     model="openai:whisper-1",
            ...     file="audio.mp3",
            ...     language="en"
            ... )

            # Provider-specific features
            >>> result = client.audio.transcriptions.create(
            ...     model="deepgram:nova-2",
            ...     file="audio.mp3",
            ...     language="en",  # Common param
            ...     punctuate=True,  # Deepgram-specific
            ...     diarize=True     # Deepgram-specific
            ... )
        """
        # Validate model format
        if ":" not in model:
            raise ValueError(
                f"Invalid model format. Expected 'provider:model', got '{model}'"
            )

        # Extract provider and model name
        provider_key, model_name = model.split(":", 1)

        # Validate provider is supported
        supported_providers = ProviderFactory.get_supported_providers()
        if provider_key not in supported_providers:
            raise ValueError(
                f"Invalid provider key '{provider_key}'. "
                f"Supported providers: {supported_providers}"
            )

        # Validate and map parameters
        validated_params = self.client.param_validator.validate_and_map(
            provider_key, kwargs
        )

        # Initialize provider if not already initialized
        if provider_key not in self.client.providers:
            config = self.client.provider_configs.get(provider_key, {})
            try:
                self.client.providers[provider_key] = ProviderFactory.create_provider(
                    provider_key, config
                )
            except ImportError as e:
                raise ValueError(f"Provider '{provider_key}' is not available: {e}")

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")

        # Check if provider supports audio transcription
        if not hasattr(provider, "audio") or provider.audio is None:
            raise ValueError(
                f"Provider '{provider_key}' does not support audio transcription."
            )

        # Determine if streaming is requested
        should_stream = validated_params.get("stream", False)

        # Delegate to provider implementation
        try:
            if should_stream:
                # Check if provider supports output streaming
                if hasattr(provider.audio, "transcriptions") and hasattr(
                    provider.audio.transcriptions, "create_stream_output"
                ):
                    return provider.audio.transcriptions.create_stream_output(
                        model_name, file, **validated_params
                    )
                else:
                    raise ValueError(
                        f"Provider '{provider_key}' does not support streaming transcription."
                    )
            else:
                # Non-streaming (batch) transcription
                if hasattr(provider.audio, "transcriptions") and hasattr(
                    provider.audio.transcriptions, "create"
                ):
                    return provider.audio.transcriptions.create(
                        model_name, file, **validated_params
                    )
                else:
                    raise ValueError(
                        f"Provider '{provider_key}' does not support audio transcription."
                    )
        except NotImplementedError:
            raise ValueError(
                f"Provider '{provider_key}' does not support audio transcription."
            )
