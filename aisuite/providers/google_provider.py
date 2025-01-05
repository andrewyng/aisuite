"""The interface to Google's Vertex AI."""

import os
import json
from typing import List, Dict, Any, Optional

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part,
    Tool,
    FunctionDeclaration,
)

from aisuite.framework import ProviderInterface, ChatCompletionResponse


DEFAULT_TEMPERATURE = 0.7


class GoogleMessageConverter:
    @staticmethod
    def convert_user_role_message(message: Dict[str, Any]) -> Content:
        """Convert user or system messages to Google Vertex AI format."""
        parts = [Part.from_text(message["content"])]
        return Content(role="user", parts=parts)

    @staticmethod
    def convert_assistant_role_message(message: Dict[str, Any]) -> Content:
        """Convert assistant messages to Google Vertex AI format."""
        parts = [Part.from_text(message["content"])]
        return Content(role="model", parts=parts)

    @staticmethod
    def convert_tool_role_message(message: Dict[str, Any]) -> Optional[Content]:
        """Convert tool messages to Google Vertex AI format."""
        if "content" not in message:
            return None

        try:
            content_json = json.loads(message["content"])
            parts = [Part.from_function_response(content_json)]
        except json.JSONDecodeError:
            parts = [Part.from_text(message["content"])]

        return Content(role="user", parts=parts)

    @staticmethod
    def convert_request(messages: List[Dict[str, Any]]) -> List[Content]:
        """Convert messages to Google Vertex AI format."""
        # Convert all messages to dicts if they're Message objects
        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message
            for message in messages
        ]

        formatted_messages = []
        for message in messages:
            if message["role"] == "tool":
                vertex_message = GoogleMessageConverter.convert_tool_role_message(
                    message
                )
                if vertex_message:
                    formatted_messages.append(vertex_message)
            elif message["role"] == "assistant":
                formatted_messages.append(
                    GoogleMessageConverter.convert_assistant_role_message(message)
                )
            else:  # user or system role
                formatted_messages.append(
                    GoogleMessageConverter.convert_user_role_message(message)
                )

        return formatted_messages

    @staticmethod
    def convert_response(response) -> ChatCompletionResponse:
        """Normalize the response from Vertex AI to match OpenAI's response format."""
        openai_response = ChatCompletionResponse()

        # Check if the response contains function calls
        if hasattr(response.candidates[0].content, "function_call"):
            function_call = response.candidates[0].content.function_call
            openai_response.choices[0].message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"call_{hash(function_call.name)}",  # Generate a unique ID
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(function_call.args),
                        },
                    }
                ],
            }
            openai_response.choices[0].finish_reason = "tool_calls"
        else:
            # Handle regular text response
            openai_response.choices[0].message.content = (
                response.candidates[0].content.parts[0].text
            )
            openai_response.choices[0].finish_reason = "stop"

        return openai_response


class GoogleProvider(ProviderInterface):
    """Implements the ProviderInterface for interacting with Google's Vertex AI."""

    def __init__(self, **config):
        """Set up the Google AI client with a project ID."""
        self.project_id = config.get("project_id") or os.getenv("GOOGLE_PROJECT_ID")
        self.location = config.get("region") or os.getenv("GOOGLE_REGION")
        self.app_creds_path = config.get("application_credentials") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )

        if not self.project_id or not self.location or not self.app_creds_path:
            raise EnvironmentError(
                "Missing one or more required Google environment variables: "
                "GOOGLE_PROJECT_ID, GOOGLE_REGION, GOOGLE_APPLICATION_CREDENTIALS. "
                "Please refer to the setup guide: /guides/google.md."
            )

        vertexai.init(project=self.project_id, location=self.location)

        self.transformer = GoogleMessageConverter()

    def chat_completions_create(self, model, messages, **kwargs):
        """Request chat completions from the Google AI API.

        Args:
        ----
            model (str): Identifies the specific provider/model to use.
            messages (list of dict): A list of message objects in chat history.
            kwargs (dict): Optional arguments for the Google AI API.

        Returns:
        -------
            The ChatCompletionResponse with the completion result.

        """

        # Set the temperature if provided, otherwise use the default
        temperature = kwargs.get("temperature", DEFAULT_TEMPERATURE)

        # Convert messages to Vertex AI format
        message_history = self.transformer.convert_request(messages)

        # Handle tools if provided
        tools = None
        if "tools" in kwargs:
            tools = [
                Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name=tool["function"]["name"],
                            description=tool["function"].get("description", ""),
                            parameters=tool["function"]["parameters"],
                        )
                    ]
                )
                for tool in kwargs["tools"]
            ]

        print(tools)
        # Create the GenerativeModel
        model = GenerativeModel(
            model,
            generation_config=GenerationConfig(temperature=temperature),
            # tools=tools
        )

        # Start chat and get response
        chat = model.start_chat(history=message_history[:-1])
        response = chat.send_message(
            message_history[-1].parts[0].text,
        )

        # Convert and return the response
        return self.transformer.convert_response(response)
