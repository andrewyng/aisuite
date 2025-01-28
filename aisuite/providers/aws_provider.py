import os
import json
from typing import List, Dict, Any, Tuple, Optional

import boto3
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message
import botocore


class BedrockConfig:
    INFERENCE_PARAMETERS = ["maxTokens", "temperature", "topP", "stopSequences"]

    def __init__(self, **config):
        self.region_name = config.get(
            "region_name", os.getenv("AWS_REGION", "us-west-2")
        )

    def create_client(self):
        return boto3.client("bedrock-runtime", region_name=self.region_name)


# AWS Bedrock API Example -
# https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-inference-call.html
# https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use-examples.html
class BedrockMessageConverter:
    @staticmethod
    def convert_request(
        messages: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Convert messages to AWS Bedrock format."""
        # Convert all messages to dicts if they're Message objects
        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message
            for message in messages
        ]

        # Handle system message
        system_message = []
        if messages and messages[0]["role"] == "system":
            system_message = [{"text": messages[0]["content"]}]
            messages = messages[1:]

        formatted_messages = []
        for message in messages:
            # Skip any additional system messages
            if message["role"] == "system":
                continue

            if message["role"] == "tool":
                bedrock_message = BedrockMessageConverter.convert_tool_result(message)
                if bedrock_message:
                    formatted_messages.append(bedrock_message)
            elif message["role"] == "assistant":
                bedrock_message = BedrockMessageConverter.convert_assistant(message)
                if bedrock_message:
                    formatted_messages.append(bedrock_message)
            else:  # user messages
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": [{"text": message["content"]}],
                    }
                )

        return system_message, formatted_messages

    @staticmethod
    def convert_response_tool_call(
        response: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Convert AWS Bedrock tool call response to OpenAI format."""
        if response.get("stopReason") != "tool_use":
            return None

        tool_calls = []
        for content in response["output"]["message"]["content"]:
            if "toolUse" in content:
                tool = content["toolUse"]
                tool_calls.append(
                    {
                        "type": "function",
                        "id": tool["toolUseId"],
                        "function": {
                            "name": tool["name"],
                            "arguments": json.dumps(tool["input"]),
                        },
                    }
                )

        if not tool_calls:
            return None

        return {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "refusal": None,
        }

    @staticmethod
    def convert_tool_result(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert OpenAI tool result format to AWS Bedrock format."""
        if message["role"] != "tool" or "content" not in message:
            return None

        tool_call_id = message.get("tool_call_id")
        if not tool_call_id:
            raise LLMError("Tool result message must include tool_call_id")

        try:
            content_json = json.loads(message["content"])
            content = [{"json": content_json}]
        except json.JSONDecodeError:
            content = [{"text": message["content"]}]

        return {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": tool_call_id, "content": content}}
            ],
        }

    @staticmethod
    def convert_assistant(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert OpenAI assistant format to AWS Bedrock format."""
        if message["role"] != "assistant":
            return None

        content = []

        if message.get("content"):
            content.append({"text": message["content"]})

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                if tool_call["type"] == "function":
                    try:
                        input_json = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        input_json = tool_call["function"]["arguments"]

                    content.append(
                        {
                            "toolUse": {
                                "toolUseId": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": input_json,
                            }
                        }
                    )

        return {"role": "assistant", "content": content} if content else None

    @staticmethod
    def convert_response(response: Dict[str, Any]) -> ChatCompletionResponse:
        """Normalize the response from the Bedrock API to match OpenAI's response format."""
        norm_response = ChatCompletionResponse()

        # Check if the model is requesting tool use
        if response.get("stopReason") == "tool_use":
            tool_message = BedrockMessageConverter.convert_response_tool_call(response)
            if tool_message:
                norm_response.choices[0].message = Message(**tool_message)
                norm_response.choices[0].finish_reason = "tool_calls"
                return norm_response

        # Handle regular text response
        norm_response.choices[0].message.content = response["output"]["message"][
            "content"
        ][0]["text"]

        # Map Bedrock stopReason to OpenAI finish_reason
        stop_reason = response.get("stopReason")
        if stop_reason == "complete":
            norm_response.choices[0].finish_reason = "stop"
        elif stop_reason == "max_tokens":
            norm_response.choices[0].finish_reason = "length"
        else:
            norm_response.choices[0].finish_reason = stop_reason

        return norm_response


class AwsProvider(Provider):
    def __init__(self, **config):
        """Initialize the AWS Bedrock provider with the given configuration."""
        self.config = BedrockConfig(**config)
        self.client = self.config.create_client()
        self.transformer = BedrockMessageConverter()

    def convert_response(self, response: Dict[str, Any]) -> ChatCompletionResponse:
        """Normalize the response from the Bedrock API to match OpenAI's response format."""
        return self.transformer.convert_response(response)

    def _convert_tool_spec(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert tool specifications to Bedrock format."""
        if "tools" not in kwargs:
            return None

        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", " "),
                        "inputSchema": {"json": tool["function"]["parameters"]},
                    }
                }
                for tool in kwargs["tools"]
            ]
        }
        return tool_config

    def _prepare_request_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the configuration for the Bedrock API request."""
        # Convert tools and remove from kwargs
        tool_config = self._convert_tool_spec(kwargs)
        kwargs.pop("tools", None)  # Remove tools from kwargs if present

        inference_config = {
            key: kwargs[key]
            for key in BedrockConfig.INFERENCE_PARAMETERS
            if key in kwargs
        }

        additional_fields = {
            key: value
            for key, value in kwargs.items()
            if key not in BedrockConfig.INFERENCE_PARAMETERS
        }

        request_config = {
            "inferenceConfig": inference_config,
            "additionalModelRequestFields": additional_fields,
        }

        if tool_config is not None:
            request_config["toolConfig"] = tool_config

        return request_config

    def chat_completions_create(
        self, model: str, messages: List[Dict[str, Any]], **kwargs
    ) -> ChatCompletionResponse:
        """Create a chat completion request to AWS Bedrock."""
        system_message, formatted_messages = self.transformer.convert_request(messages)
        request_config = self._prepare_request_config(kwargs)

        try:
            response = self.client.converse(
                modelId=model,
                messages=formatted_messages,
                system=system_message,
                **request_config
            )
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                error_message = e.response["Error"]["Message"]
                raise LLMError(error_message)
            else:
                raise

        return self.convert_response(response)
