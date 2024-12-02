# Watsonx with `aisuite`

A a step-by-step guide to set up Watsonx with the `aisuite` library, enabling you to use IBM Watsonx's powerful AI models for various tasks.

## Setup Instructions

### Step 1: Create a Watsonx Account

1. Visit [IBM Watsonx](https://www.ibm.com/watsonx).
2. Sign up for a new account or log in with your existing IBM credentials.
3. Once logged in, navigate to the **Watsonx Dashboard**.

---

### Step 2: Obtain API Credentials

1. **Generate an API Key**:
   - Go to the **API Keys** section in your Watsonx account settings.
   - Click on **Create API Key**.
   - Provide a name for your API key (e.g., `MyWatsonxKey`).
   - Click **Generate**, then download or copy the API key. **Keep this key secure!**

2. **Locate the Service URL**:
   - Go to the **Endpoints** section in the Watsonx dashboard.
   - Find the URL corresponding to your service and note it. This is your `WATSONX_SERVICE_URL`.

3. **Get the Project ID**:
   - Navigate to the **Projects** tab in the dashboard.
   - Select the project you want to use.
   - Copy the **Project ID**. This will serve as your `WATSONX_PROJECT_ID`.

---

### Step 3: Set Environment Variables

To simplify authentication, set the following environment variables:

Run the following commands in your terminal:

```bash
export WATSONX_API_KEY="your-watsonx-api-key"
export WATSONX_SERVICE_URL="your-watsonx-service-url"
export WATSONX_PROJECT_ID="your-watsonx-project-id"
```


## Create a Chat Completion

Install the `ibm-watsonx-ai` Python client:

Example with pip:

```shell
pip install ibm-watsonx-ai
```

Example with poetry:

```shell
poetry add ibm-watsonx-ai
```

In your code:

```python
import aisuite as ai
client = ai.Client()

provider = "watsonx"
model_id = "meta-llama/llama-3-70b-instruct"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```