from .provider import ProviderFactory
import os
from .utils.tools import Tools


class Client:
    def __init__(self, provider_configs: dict = {}):
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
        """
        self.providers = {}
        self.provider_configs = provider_configs
        self._chat = None
        self._initialize_providers()

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

    def configure(self, provider_configs: dict = None):
        """
        Configure the client with provider configurations.
        """
        if provider_configs is None:
            return

        self.provider_configs.update(provider_configs)
        self._initialize_providers()  # NOTE: This will override existing provider instances.

    @property
    def chat(self):
        """Return the chat API interface."""
        if not self._chat:
            self._chat = Chat(self)
        return self._chat


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

    def _tool_runner(
        self,
        provider,
        model_name: str,
        messages: list,
        tools: any,
        max_turns: int,
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
        # Handle tools validation and conversion
        if isinstance(tools, Tools):
            tools_instance = tools
            kwargs["tools"] = tools_instance.tools()
        else:
            # Check if passed tools are callable
            if not all(callable(tool) for tool in tools):
                raise ValueError("One or more tools is not callable")
            tools_instance = Tools(tools)
            kwargs["tools"] = tools_instance.tools()

        turns = 0
        intermediate_responses = []  # Store intermediate responses
        intermediate_messages = []  # Store all messages including tool interactions

        while turns < max_turns:
            # Make the API call
            response = provider.chat_completions_create(model_name, messages, **kwargs)
            response = self._extract_thinking_content(response)

            # Store intermediate response
            intermediate_responses.append(response)

            # Check if there are tool calls in the response
            tool_calls = (
                getattr(response.choices[0].message, "tool_calls", None)
                if hasattr(response, "choices")
                else None
            )

            # Store the model's message
            intermediate_messages.append(response.choices[0].message)

            if not tool_calls:
                # Set the intermediate data in the final response
                response.intermediate_responses = intermediate_responses[
                    :-1
                ]  # Exclude final response
                response.choices[0].intermediate_messages = intermediate_messages
                return response

            # Execute tools and get results
            results, tool_messages = tools_instance.execute_tool(tool_calls)

            # Add tool messages to intermediate messages
            intermediate_messages.extend(tool_messages)

            # Add the assistant's response and tool results to messages
            messages.extend([response.choices[0].message, *tool_messages])

            turns += 1

        # Set the intermediate data in the final response
        response.intermediate_responses = intermediate_responses[
            :-1
        ]  # Exclude final response
        response.choices[0].intermediate_messages = intermediate_messages
        return response

    def create(self, model: str, messages: list, **kwargs):
        """
        Create chat completion based on the model, messages, and any extra arguments.
        Supports automatic tool execution when max_turns is specified.
        """
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
        if provider_key not in self.client.providers:
            config = self.client.provider_configs.get(provider_key, {})
            self.client.providers[provider_key] = ProviderFactory.create_provider(
                provider_key, config
            )

        provider = self.client.providers.get(provider_key)
        if not provider:
            raise ValueError(f"Could not load provider for '{provider_key}'.")

        # Extract tool-related parameters
        max_turns = kwargs.pop("max_turns", None)
        tools = kwargs.get("tools", None)

        # Check environment variable before allowing multi-turn tool execution
        if max_turns is not None and tools is not None:
            return self._tool_runner(
                provider,
                model_name,
                messages.copy(),
                tools,
                max_turns,
            )

        # Default behavior without tool execution
        # Delegate the chat completion to the correct provider's implementation
        response = provider.chat_completions_create(model_name, messages, **kwargs)
        return self._extract_thinking_content(response)
