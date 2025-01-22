# Google Gemini API

To use the Google Gemini API with `aisuite`, follow these steps:

## Prerequisites

1. **Google Cloud Account**: Ensure you have a Google Cloud account. If not, create one at [Google Cloud](https://cloud.google.com/).
2. **API Key**: Obtain an API key for the Google Gemini API. You can generate an API key from the [Google Cloud Console](https://console.cloud.google.com/).

## Installation

Install the `google-genai` Python client:

Example with pip:
```shell
pip install google-genai
```

Example with poetry:
```shell
poetry add google-genai
```

## Configuration

Set the `GEMINI_API_KEY` environment variable with your API key:

```shell
export GEMINI_API_KEY="your-gemini-api-key"
```

## Create a Chat Completion

In your code:
```python
import aisuite as ai
client = ai.Client()

provider = "google_genai"
model_id = "gemini-2.0-flash-exp"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
