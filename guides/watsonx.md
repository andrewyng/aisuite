# Watsonx with `aisuite`

A a step-by-step guide to set up Watsonx with the `aisuite` library, enabling you to use IBM Watsonx's powerful AI models for various tasks.

## Setup Instructions

### Step 1: Create a Watsonx Account

1. Visit [IBM Watsonx](https://www.ibm.com/watsonx).
2. Sign up for a new account or log in with your existing IBM credentials.
3. Once logged in, navigate to the **Watsonx Dashboard** (<https://dataplatform.cloud.ibm.com>)

---

### Step 2: Obtain API Credentials

1. **Generate an API Key**:
   - Go to IAM > API keys and create a new API key (<https://cloud.ibm.com/iam/overview>)
   - Copy the API key. This is your `WATSONX_API_KEY`.

2. **Locate the Service URL**:
   - Your service URL is based on the region where your service is hosted.
   - Pick one from the list here <https://cloud.ibm.com/apidocs/watsonx-ai#endpoint-url>
   - Copy the service URL. This is your `WATSONX_SERVICE_URL`.

3. **Get the Project ID**:
   - Go to the **Watsonx Dashboard** (<https://dataplatform.cloud.ibm.com>)
   - Under the **Projects** section, If you don't have a sandbox project, create a new project.
   - Navigate to the **Manage** tab and find the **Project ID**.
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