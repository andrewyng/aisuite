import openai
import os
from aisuite.provider import Provider, LLMError
from aisuite.framework.function_call import generate_function_calling_schema_for_openai
from loguru import logger


class OpenaiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )

        # NOTE: We could choose to remove above lines for api_key since OpenAI will automatically
        # infer certain values from the environment variables.
        # Eg: OPENAI_API_KEY, OPENAI_ORG_ID, OPENAI_PROJECT_ID, OPENAI_BASE_URL, etc.

        # Pass the entire config to the OpenAI client constructor
        self.client = openai.OpenAI(**config)

    def chat_completions_create(
        self, model, messages, tools=None, tool_choice=None, **kwargs
    ):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        if tools:
            logger.info("Generating function calling schema for tools")
            tools_with_schema = [
                generate_function_calling_schema_for_openai(tool) for tool in tools
            ]
            return self._process_tool_calls(
                model, messages, tools_with_schema, tool_choice, tools, **kwargs
            )
        logger.info("No tools provided, calling OpenAI API directly")
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=None,
            **kwargs,  # Pass any additional arguments to the OpenAI API
        )

    def _process_tool_calls(
        self, model, messages, tools_with_schema, tool_choice, tools, **kwargs
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_with_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
        logger.info("OpenAI API response received. Processing tool calls.")
        messages.append(response.choices[0].message)
        if response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            for tool_call in tool_calls:
                logger.info(f"Processing tool call: {tool_call}")
                tool_id = tool_call.id
                tool_name = tool_call.function.name
                arguments = tool_call.function.arguments

                tool_response = self.execute_tool(tool_name, arguments, tools)
                logger.info(f"Tool {tool_name} executed with response: {tool_response}")
                messages.append(
                    {
                        "role": "tool",
                        "content": str(tool_response),
                        "tool_call_id": tool_id,
                        "name": tool_name,
                    }
                )
        logger.info("All tool calls processed. Calling OpenAI API again.")
        final_response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=None,
            **kwargs,
        )
        return final_response

    def execute_tool(self, tool_name, arguments, tools):
        """Execute a tool call"""
        matching_tool = None
        for tool in tools:
            if tool.__name__ == tool_name:
                matching_tool = tool
                break

        if matching_tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in provided tools")

        try:
            import json

            print(json.loads(arguments))
            return matching_tool(**json.loads(arguments))
        except Exception as e:
            raise Exception(f"Error executing tool: {str(e)}")
