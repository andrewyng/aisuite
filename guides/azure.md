# Azure AI

To use Azure AI with the `aisuite` library, you'll need to set up an Azure account and configure your environment for Azure AI services.

## Create an Azure Account and deploy a model from AI Studio

1. Visit [Azure Portal](https://portal.azure.com/) and sign up for an account if you don't have one.
2. Create a project and resource group.
3. Choose a model from https://ai.azure.com/explore/models and deploy it. You can choose serverless deployment option.
4. Give a deployment name. Lets say you choose to deploy Mistral-large-2407. You could leave the deployment names as "mistral-large-2407" or give a custom name.
5. You can see the deployment from project/deployment option. Note the Target URI from the Endpoint panel. It should look something like this - "https://aisuite-Mistral-large-2407.westus3.models.ai.azure.com".
6. Also note, that is provides a Chat completion URL. It should look like this - https://aisuite-Mistral-large-2407.westus3.models.ai.azure.com/v1/chat/completions


## Obtain Necessary Details & set environment variables.

After creating your deployment, you'll need to gather the following information:

1. API Key: Found in the "Keys and Endpoint" section of your Azure OpenAI resource.
2. Base URL: This can be obtained from your deployment details. It will look something like this - `https://aisuite-Mistral-large-2407.westus3.models.ai.azure.com/v1/`
3. API Version: Optional configuration and mainly introduced for Azure OpenAI services. Once specified, the `api-version` query parameters will be added in the end of the API request.


Set the following environment variables:

```shell
export AZURE_API_KEY="your-api-key"
export AZURE_BASE_URL="https://deployment-name.region-name.models.ai.azure.com/v1"
export AZURE_API_VERSION="=2024-08-01-preview"
```

## Create a Chat Completion

With your account set up and environment configured, you can send a chat completion request:

```python
import os
import aisuite as ai

# Either set the environment variables or pass the values in the provider config.
# Values passed in the config override the environment variables.
client = ai.Client({
    "azure": {
        "base_url": os.environ["AZURE_BASE_URL"],
        "api_key": os.environ["AZURE_API_KEY"],
        "api_version": os.environ["AZURE_API_VERSION"],
    }
})

model = "azure:aisuite-Mistral-large-2407"  # Replace with your deployment name.
# The model name must match the deployment name in the base-url.

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

## Keyless authentication (Microsoft Entra ID)

Instead of an API key you can authenticate with a Microsoft Entra ID token. Pass
`azure_ad_token_provider`, a callable that returns a token; it is called on every
request, so the token is refreshed when it expires. A static `azure_ad_token` is
also accepted. Either one takes precedence over `api_key` and is sent as
`Authorization: Bearer <token>`.

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import aisuite as ai

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

client = ai.Client({
    "azure": {
        "base_url": "https://<resource>.openai.azure.com/openai/deployments/<deployment>",
        "api_version": "2024-10-21",
        "azure_ad_token_provider": token_provider,
    }
})
```

Use the token scope your endpoint expects; the one above is for Azure OpenAI and
Azure AI Foundry resources.

## Custom headers

Headers set in `default_headers` are sent with every request, and `extra_headers`
adds (or overrides) headers for a single call:

```python
client = ai.Client({
    "azure": {
        "base_url": os.environ["AZURE_BASE_URL"],
        "api_key": os.environ["AZURE_API_KEY"],
        "default_headers": {"x-ms-useragent": "my-app/1.0"},
    }
})

response = client.chat.completions.create(
    model="azure:aisuite-Mistral-large-2407",
    messages=messages,
    extra_headers={"x-user-id": "user-1234"},
)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).