from .provider import ProviderFactory
from .utils.tools import Tools
import asyncio

class ToolRunner:
    def __init__(self, provider, model_name, messages, tools, max_turns):
        self.provider = provider
        self.model_name = model_name
        self.messages = messages
        self.tools = tools
        self.max_turns = max_turns

    async def run_async(
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
            if provider is AsyncProvider:
                response = await provider.chat_completions_create_async(model_name, messages, **kwargs)
            else:
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

    def run(
        self,
        provider,
        model_name: str,
        messages: list,
        tools: any,
        max_turns: int,
        **kwargs,
    ):
        return asyncio.run(self.run_async(provider, model_name, messages, tools, max_turns, **kwargs))
