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
import pprint

from aisuite.framework import ProviderInterface, ChatCompletionResponse, Message


DEFAULT_TEMPERATURE = 0.7
ENABLE_DEBUG_MESSAGES = False

# Links.
# https://codelabs.developers.google.com/codelabs/gemini-function-calling#6
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#chat-samples


class GoogleMessageConverter:
    @staticmethod
    def convert_user_role_message(message: Dict[str, Any]) -> Content:
        """Convert user or system messages to Google Vertex AI format."""
        parts = [Part.from_text(message["content"])]
        return Content(role="user", parts=parts)

    @staticmethod
    def convert_assistant_role_message(message: Dict[str, Any]) -> Content:
        """Convert assistant messages to Google Vertex AI format."""
        if "tool_calls" in message and message["tool_calls"]:
            # Handle function calls
            tool_call = message["tool_calls"][
                0
            ]  # Assuming single function call for now
            function_call = tool_call["function"]

            # Create a Part from the function call
            parts = [
                Part.from_dict(
                    {
                        "function_call": {
                            "name": function_call["name"],
                            # "arguments": json.loads(function_call["arguments"])
                        }
                    }
                )
            ]
            # return Content(role="function", parts=parts)
        else:
            # Handle regular text messages
            parts = [Part.from_text(message["content"])]
            # return Content(role="model", parts=parts)

        return Content(role="model", parts=parts)

    @staticmethod
    def convert_tool_role_message(message: Dict[str, Any]) -> Part:
        """Convert tool messages to Google Vertex AI format."""
        if "content" not in message:
            raise ValueError("Tool result message must have a content field")

        try:
            content_json = json.loads(message["content"])
            part = Part.from_function_response(
                name=message["name"], response=content_json
            )
            # TODO: Return Content instead of Part. But returning Content is not working.
            return part
        except json.JSONDecodeError:
            raise ValueError("Tool result message must be valid JSON")

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

        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the response")
            pprint.pprint(response)

        # TODO: We need to go through each part, because function call may not be the first part.
        #       Currently, we are only handling the first part, but this is not enough.
        #
        # This is a valid response:
        # candidates {
        #   content {
        #     role: "model"
        #     parts {
        #       text: "The current temperature in San Francisco is 72 degrees Celsius. \n\n"
        #     }
        #     parts {
        #       function_call {
        #         name: "is_it_raining"
        #         args {
        #           fields {
        #             key: "location"
        #             value {
        #               string_value: "San Francisco"
        #             }
        #           }
        #         }
        #       }
        #     }
        #   }
        #   finish_reason: STOP

        # Check if the response contains function calls
        # Note: Just checking if the function_call attribute exists is not enough,
        #       it is important to check if the function_call is not None.
        if (
            hasattr(response.candidates[0].content.parts[0], "function_call")
            and response.candidates[0].content.parts[0].function_call
        ):
            function_call = response.candidates[0].content.parts[0].function_call

            # args is a MapComposite.
            # Convert the MapComposite to a dictionary
            args_dict = {}
            # Another way to try is: args_dict = dict(function_call.args)
            for key, value in function_call.args.items():
                args_dict[key] = value
            if ENABLE_DEBUG_MESSAGES:
                print("Dumping the args_dict")
                pprint.pprint(args_dict)

            openai_response.choices[0].message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"call_{hash(function_call.name)}",  # Generate a unique ID
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(args_dict),
                        },
                    }
                ],
                "refusal": None,
            }
            openai_response.choices[0].message = Message(
                **openai_response.choices[0].message
            )
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
                            parameters={
                                "type": "object",
                                "properties": {
                                    param_name: {
                                        "type": param_info.get("type", "string"),
                                        "description": param_info.get(
                                            "description", ""
                                        ),
                                        **(
                                            {"enum": param_info["enum"]}
                                            if "enum" in param_info
                                            else {}
                                        ),
                                    }
                                    for param_name, param_info in tool["function"][
                                        "parameters"
                                    ]["properties"].items()
                                },
                                "required": tool["function"]["parameters"].get(
                                    "required", []
                                ),
                            },
                        )
                        for tool in kwargs["tools"]
                    ]
                )
            ]

        # Create the GenerativeModel
        model = GenerativeModel(
            model,
            generation_config=GenerationConfig(temperature=temperature),
            tools=tools,
        )

        if ENABLE_DEBUG_MESSAGES:
            print("Dumping the message_history")
            pprint.pprint(message_history)

        # Start chat and get response
        chat = model.start_chat(history=message_history[:-1])
        last_message = message_history[-1]

        # If the last message is a function response, send the Part object directly
        # Otherwise, send just the text content
        message_to_send = (
            Content(role="function", parts=[last_message])
            if isinstance(last_message, Part)
            else last_message.parts[0].text
        )
        # response = chat.send_message(message_to_send)
        response = chat.send_message(message_to_send)

        # Convert and return the response
        return self.transformer.convert_response(response)
