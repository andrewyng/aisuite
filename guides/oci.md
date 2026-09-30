# OCI Generative AI

OCI Generative AI exposes an OpenAI-compatible Chat Completions endpoint.
`aisuite` uses that compatibility layer, so OCI models are called with the same
`client.chat.completions.create(...)` interface used by the other providers.

## Quick start

Install the optional OCI dependencies:

```shell
pip install 'aisuite[oci]'
```

For local development, create a `.env` file in your project. Use an OCI
**Generative AI API key**, not an OCI IAM API signing key:

```dotenv
OCI_GENAI_REGION=eu-frankfurt-1
OCI_GENAI_API_KEY=your-oci-generative-ai-api-key
# Optional when required by the OCI endpoint:
# OCI_GENAI_PROJECT=ocid1.generativeaiproject.oc1...
```

Keep `.env` out of source control. `aisuite` reads process environment
variables and does not load `.env` automatically. Before running an application
from a local shell, load the file:

```shell
set -a; source .env; set +a
```

```python
import aisuite as ai

client = ai.Client()
response = client.chat.completions.create(
    model="oci:openai.gpt-5.5",
    messages=[{"role": "user", "content": "Explain OCI in one sentence."}],
)
print(response.choices[0].message.content)
```

API-key authentication is suitable for local development and testing. Keep the
secret out of source control, give it least-privilege IAM permissions, and rotate
it regularly.

## Resource Principal

For OCI Functions, OKE, Container Instances, and other OCI-managed production
workloads, use Resource Principal authentication. OCI injects and rotates the
credentials; do not set `OCI_GENAI_API_KEY`.

```python
import aisuite as ai

client = ai.Client(provider_configs={
    "oci": {
        "region": "eu-frankfurt-1",
        "auth_type": "resource_principal",
        # Optional when required by your OCI endpoint:
        # "project": "ocid1.generativeaiproject.oc1...",
    }
})

response = client.chat.completions.create(
    model="oci:openai.gpt-5.5",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Ensure the Resource Principal has an IAM policy permitting it to use OCI
Generative AI in the selected compartment.

## Configuration reference

| Setting | Required | Description |
| --- | --- | --- |
| `OCI_GENAI_REGION` or `region` | Yes, unless `base_url` is set | OCI region hosting the model. |
| `OCI_GENAI_API_KEY` or `api_key` | API-key mode only | OCI Generative AI service API key. |
| `auth_type` | No | `api_key` (default) or `resource_principal`. |
| `project` / `OCI_GENAI_PROJECT` | No | OCI Generative AI project OCID, when required by the endpoint. |
| `base_url` | No | Endpoint override; defaults to OCI's `/openai/v1` endpoint for the region. |

## Local integration tests

The OCI integration tests make real API calls and can incur costs. They are
skipped by default, even when a `.env` file is present. To run them locally with
the API key in `.env`, explicitly opt in:

```shell
AISUITE_RUN_OCI_INTEGRATION=1 conda run -n aisuite pytest \
  tests/providers/test_oci_integration.py -m "integration and llm" -v
```

The tests use `openai.gpt-5.5` by default. Set `OCI_GENAI_TEST_MODEL` to test a
different OCI OpenAI-compatible model.
