# Hugging Face AI

To use Hugging Face with the `aisuite` library, you'll need to set up a Hugging Face account, obtain the necessary API credentials, and configure your environment for Hugging Face's API.

## Create a Hugging Face Account and Deploy a Model

1. Visit [Hugging Face](https://huggingface.co/) and sign up for an account if you don't already have one.
2. Explore conversational models on the [Hugging Face Model Hub](https://huggingface.co/models?inference=warm&other=conversational&sort=trending) and select a model you want to use. Popular models include conversational AI models like `gpt2`, `gpt3`, and `mistral`.
3. Deploy or host your chosen model if needed; Hugging Face provides various hosting options, including free, individual, and organizational hosting. Using Serverless Inference API is a fast way to get started.
5. Once the model is deployed (or if using a public model directly), note the model's unique identifier (e.g., `mistralai/Mistral-7B-Instruct-v0.3`), which you'll use for making requests.

## Obtain Necessary Details & Set Environment Variables

After setting up your model, you'll need to gather the following information:

- **API Token**: You can generate an API token in your [Hugging Face account settings](https://huggingface.co/settings/tokens).

Set the following environment variables to make authentication and requests easy:

```shell
export HF_TOKEN="your-api-token"
```

## Create a Chat Completion

With your account set up and environment variables configured, you can send a chat completion request as follows:

```python
import os
import aisuite as ai

# Either set the environment variables or define the parameters below.
# Setting the parameters in ai.Client() will override the environment variable values.
client = ai.Client()

model = "huggingface:your-model-name"  # Replace with your model's identifier.

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like today?"},
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
)

print(response.choices[0].message.content)
```

### Notes

- Ensure that the `model` variable matches the identifier of your model as seen in the Hugging Face Model Hub.
- If you encounter any rate limits or API access restrictions, you may have to upgrade your Hugging Face plan to enable higher usage limits.
"""

Happy coding! If you would like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).