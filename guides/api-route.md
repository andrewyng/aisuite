# API Route

To use [API Route](https://api-route.com) with `aisuite`, create an account and generate an API key from the dashboard. Then set:

```shell
export API_ROUTE_API_KEY="your-api-route-key"
```

API Route exposes an OpenAI-compatible API at `https://global.api-route.com/v1`, so the provider reuses the OpenAI Python client.

## Create a Chat Completion

Install the `openai` Python client:

```shell
pip install openai
```

Then use the `apiroute` provider name:

```python
import aisuite as ai

client = ai.Client()

response = client.chat.completions.create(
    model="apiroute:claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

You can override the endpoint with `API_ROUTE_BASE_URL` or by passing `base_url` in the provider config.
