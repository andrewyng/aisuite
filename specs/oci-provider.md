# OCI Generative AI provider

## Goal

Add an `oci` provider that lets aisuite call OCI Generative AI's OpenAI-compatible
Chat Completions endpoint while retaining the existing OpenAI-shaped API.

## Scope

The provider supports synchronous and asynchronous chat completions and streaming.
It supports the following authentication modes:

* `api_key` (default): OCI Generative AI API key supplied via configuration or
  `OCI_GENAI_API_KEY`.
* `resource_principal`: OCI Resource Principal through `oci-genai-auth`.

The provider does not implement OCI's native inference API, Responses API,
embeddings, or audio endpoints.

## Configuration

`region` is required unless `base_url` is supplied. `project` is passed to the
OpenAI SDK when supplied. `base_url` allows private or future OCI endpoints to be
configured without changing code.

## Acceptance criteria

1. `oci:<model>` is automatically discovered by `ProviderFactory`.
2. API-key mode uses OCI's OpenAI-compatible endpoint and accepts config or env credentials.
3. Resource-principal mode creates correctly typed synchronous and asynchronous
   authenticated HTTP clients.
4. Invalid authentication mode or missing required configuration fails with a clear error.
5. Unit tests mock all external clients and make no live OCI requests.
6. Opt-in integration tests use locally supplied OCI API-key credentials to
   verify synchronous, asynchronous, and streaming chat completions.
