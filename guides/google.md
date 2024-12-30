# Google (Vertex) AI

To use Google (Vertex) AI with the `aisuite` library, you'll first need to create a Google Cloud account and set up your environment to work with Google Cloud.

## Create a Google Cloud Account and Project

Google Cloud provides in-depth [documentation](https://cloud.google.com/vertex-ai/docs/start/cloud-environment) on getting started with their platform, but here are the basic steps:

### Create your account.

Visit [Google Cloud](https://cloud.google.com/free) and follow the instructions for registering a new account. If you already have an account with Google Cloud, sign in and skip to the next step.

### Create a new project and enable billing.

Once you have an account, you can create a new project. Visit the [project selector page](https://console.cloud.google.com/projectselector2/home/dashboard) and click the "New Project" button. Give your project a name and click "Create Project." Your project will be created and you will be redirected to the project dashboard.

Now that you have a project, you'll need to enable billing. Visit the [how-to page](https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#confirm_billing_is_enabled_on_a_project) for billing enablement instructions.

### Set your project ID in an environment variable.

Set the `GOOGLE_PROJECT_ID` environment variable to the ID of your project. You can find the Project ID by visiting the project dashboard in the "Project Info" section toward the top of the page.

### Set your preferred region in an environment variable.

Set the `GOOGLE_REGION` environment variable. You can find the region by going to Project Dashboard under VertexAI side navigation menu, and then scrolling to the bottom of the page.

## Create a Service Account For API Access

Because `aisuite` needs to authenticate with Google Cloud to access the Vertex AI API, you'll need to create a service account and set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of a JSON file containing the service account's credentials, which you can download from the Google Cloud Console.

This is documented [here](https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to), and the basic steps are as follows:

1. Visit the [service accounts page](https://console.cloud.google.com/iam-admin/serviceaccounts) in the Google Cloud Console.
2. Click the "+ Create Service Account" button toward the top of the page.
3. Follow the steps for naming your service account and granting access to the project.
4. Click "Done" to create the service account.
5. Now, click the "Keys" tab towards the top of the page.
6. Click the "Add Key" menu, then select "Create New Key."
6. Choose "JSON" as the key type, and click "Create."
7. Move this file to a location on your file system like your home directory.
8. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the JSON file.

## Double check your environment is configured correctly.

At this point, you should have three environment variables set to ensure your environment is set up correctly:

- `GOOGLE_PROJECT_ID`
- `GOOGLE_REGION`
- `GOOGLE_APPLICATION_CREDENTIALS`

Once these are set, you are ready to write some code and send a chat completion request.

## Create a chat completion.

With your account and service account set up, you can send a chat completion request.

Export the environment variables:

```shell
export GOOGLE_PROJECT_ID="your-project-id"
export GOOGLE_REGION="your-region"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"
```

Install the Vertex AI SDK:

```shell
pip install vertexai
```

In your code:

```python
import aisuite as ai
client = ai.Client()

model="google:gemini-1.5-pro-001"

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
