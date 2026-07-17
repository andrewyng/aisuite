# Gemini (Gemini Developer API)

This provider talks to Google's [Gemini Developer API](https://ai.google.dev/) using a plain API key — no Google Cloud project, service account, or GCS bucket required. (For Vertex AI with GCP project auth, use the [Google](google.md) provider instead.)

## Get an API key

Visit [Google AI Studio](https://aistudio.google.com/apikey), sign in with a Google account, and click "Create API key."

## Set the API key in an environment variable

```shell
export GEMINI_API_KEY="your-gemini-api-key"
```

(`GOOGLE_API_KEY`, the SDK's own convention, works as a fallback.)

## Install the SDK and use the provider

```shell
pip install 'aisuite[gemini]'
```

```python
import aisuite as ai

client = ai.Client()

response = client.chat.completions.create(
    model="gemini:gemini-2.5-flash",
    messages=[{"role": "user", "content": "Tell me a joke."}],
)
print(response.choices[0].message.content)
```

Tool calling, streaming (`stream=True`), and image inputs are supported. Images must be base64 data URLs (`data:image/png;base64,...`) in OpenAI-style `image_url` content parts — the Gemini API does not fetch plain http(s) URLs.
