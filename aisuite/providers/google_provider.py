"""The interface to Google's Vertex AI."""

import os

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from aisuite.framework import ProviderInterface, ChatCompletionResponse


DEFAULT_TEMPERATURE = 0.7


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

        # Transform the roles in the messages
        transformed_messages = self.transform_roles(messages)

        # Convert the messages to the format expected Google
        final_message_history = self.convert_openai_to_vertex_ai(
            transformed_messages[:-1]
        )

        # Get the last message from the transformed messages
        last_message = transformed_messages[-1]["content"]

        # Create the GenerativeModel with the specified model and generation configuration
        model = GenerativeModel(
            model, generation_config=GenerationConfig(temperature=temperature)
        )

        # Start a chat with the GenerativeModel and send the last message
        chat = model.start_chat(history=final_message_history)
        response = chat.send_message(last_message)

        # Convert the response to the format expected by the OpenAI API
        return self.normalize_response(response)

    def convert_openai_to_vertex_ai(self, messages):
        """Convert OpenAI messages to Google AI messages."""
        from vertexai.generative_models import Content, Part

        history = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            parts = [Part.from_text(content)]
            history.append(Content(role=role, parts=parts))
        return history

    def transform_roles(self, messages):
        """Transform the roles in the messages based on the provided transformations."""
        openai_roles_to_google_roles = {
            "system": "user",
            "assistant": "model",
        }

        for message in messages:
            if role := openai_roles_to_google_roles.get(message["role"], None):
                message["role"] = role
        return messages

    def normalize_response(self, response):
        """Normalize the response from Google AI to match OpenAI's response format."""
        openai_response = ChatCompletionResponse()
        openai_response.choices[0].message.content = (
            response.candidates[0].content.parts[0].text
        )
        return openai_response
