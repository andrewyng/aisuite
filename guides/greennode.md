# GreenNode AI

To use GreenNode with `aisuite`, you'll need a [GreenNode AI Platform](https://aiplatform.console.greennode.ai) account, obtain the necessary API credentials, and configure your environment for GreenNode Inference's API.

## Create GreenNode AI Platform Account and Deploy a Model

1. Visit [GreenNode AI Platform](https://aiplatform.console.greennode.ai) and sign up for an account if you don't already have one.
2. After logging in, go to the [Inference](https://aiplatform.console.greennode.ai/inference), select and deploy any models you want to use. Popular models include conversational AI models like `deepseek`, `llama3`, and `qwen2.5`.
3. Deploy or host your chosen model if needed; GreenNode provides various hosting options, including **secured inference endpoint** and **public inference endpoint**
4. To use the **secured inference endpoint**, users will use the **client ID** and the **client secret keys** to be authorized via the authorization server (https://iam.api.greennode.ai/accounts/v2/auth/token) using the OAuth2 method.
5. To get the keys, go to [GreenNode AI Cloud IAM](https://iam.console.greennode.ai/service-accounts), **create a service account** to use the **secured inference endpoint** of the product.

6. Set the following environment variables to make authentication and requests easy:
```shell
export GREENNODE_API_KEY="your-iam-access-token"
export GREENNODE_BASE_URL="https://prediction-aiplatform-sea-central-1.api.greennode.ai/v1/model-inference-id/predict/v1"
```
7. If you use the **public inference endpoint** set:
```shell
export GREENNODE_API_KEY="EMPTY"
export GREENNODE_BASE_URL="https://prediction-aiplatform-sea-central-1.api.greennode.ai/v1/model-inference-id/predict/v1"
```

## Create a Chat Completion
With your account set up and environment variables configured, you can send a chat completion request as follows:

```python
import aisuite as ai

provider = "greennode"
model_id = "your-served-model-name"


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How many times has Jurgen Klopp won the Champions League?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
    stream=True
)

print(response.choices[0].message.content)
```

### Notes

- Ensure that the `model` variable matches the identifier of your served model name as configured in the GreenNode Model Registry.

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).
