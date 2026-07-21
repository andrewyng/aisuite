# Google (Gemini & Vertex AI)

To use Google models with the `aisuite` library, you have two authentication options:
1. **Google AI Studio (Recommended for most users):** A fast, developer-friendly way to authenticate using a standard `GOOGLE_API_KEY`.
2. **Google Vertex AI:** An enterprise-grade backend via Google Cloud requiring a `GOOGLE_PROJECT_ID` and GCP Service Account credentials.

---

## Option 1: Authenticate via Google AI Studio (Recommended)

This is the easiest way to get started. 

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click "Create API key" to generate your key.
4. Export the key to your environment:

```shell
export GOOGLE_API_KEY="your-api-key"
```

---

## Option 2: Authenticate via Google Cloud (Vertex AI)

If you are operating inside an enterprise GCP environment, `aisuite` seamlessly falls back to Vertex AI authentication.

1. Follow the [Vertex AI setup documentation](https://cloud.google.com/vertex-ai/docs/start/cloud-environment) to create a Google Cloud Project with Billing enabled.
2. Visit the [Service Accounts page](https://console.cloud.google.com/iam-admin/serviceaccounts) to create a service account and download its JSON key.
3. Check your `GOOGLE_REGION` at the bottom of the Vertex AI Dashboard.
4. Export your environment variables:

```shell
export GOOGLE_PROJECT_ID="your-project-id"
export GOOGLE_REGION="your-region"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
```

*(Note: The `GoogleProvider` will dynamically route through Vertex AI if a `GOOGLE_API_KEY` is not found but these GCP variables are defined).*

---

## Create a Chat Completion

With your credentials exported (either via Option 1 or Option 2), you are ready to send a chat completion request.

First, install the Google GenAI SDK extra:

```shell
pip install "aisuite[google]"
```

In your code:

```python
import aisuite as ai
client = ai.Client()

model="google:gemini-1.5-pro-latest"

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).
