# Watsonx AI

To use Watsonx AI with `aisuite`, you will need to create an IBM Cloud account and navigate to [Watsonx AI](https://cloud.ibm.com/catalog/services/watsonx). Sign up to IBM Watsonx if you haven’t done so already.

## Create an IBM IAM Access Key

Watsonx API uses IBM Cloud Identity and Access Management (IAM) to authenticate requests. Follow the steps below to create an IAM Access Key:

1. Create a project in Watsonx and note the `project_id`. You will need this later. Do not create an access token at this stage.
2. Go to IAM and create a Service ID.
3. In the API keys section of the Service ID, create an API key and save it.
4. Return to the project in Watsonx and add the Service ID in the project's access controls section as a collaborator.
5. Generate an access token using the Service ID’s API key with IAM’s endpoint.
6. Use the access token and `project_id` in the API call to invoke Watsonx’s REST API.

For more details, checkout this [medium blogpost](https://medium.com/@harangpeter/setting-up-ibm-watsonx-ai-for-api-based-text-inference-435ef6d1a6a3) written by Péter Harang.

Once you have obtained the access token, set your Access Token, Project ID, Cluster URL in the env variables

```shell
export WATSONX_PROJECT_ID="your-project-id"
export IBM_IAM_ACCESS_TOKEN="your-access-token"
export WATSONX_CLUSTER_URL="https://your-cluster-url.com"
export WATSONX_API_VERSION="2023-10-25" # Optional
```

*Note: The `WATSONX_CLUSTER_URL` is the base URL for the Watsonx AI service cluster you are using. It is essential to use the correct cluster URL for the region where your project is hosted to ensure that your API requests are directed to the appropriate service endpoint. You can find the list of endpoints [here](https://cloud.ibm.com/apidocs/watsonx-ai#endpoint-url).*

## Supported Foundation models

IBM Watson supports a wide range of models. For a list of models supported by Watsonx, visit the [documentation](https://eu-de.dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx&audience=wdp).

## Introduction

Watsonx AI Provider offers powerful AI capabilities to enhance your applications with natural language processing, machine learning, and more.

## Create a Chat Completion

```python
import aisuite as ai
client = ai.Client()

provider="watsonx"
model_id = "meta-llama/llama-3-8b-instruct"

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)
print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
