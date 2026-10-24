# AWS

To use AWS Bedrock with `aisuite` you will need to create an AWS account and
navigate to https://console.aws.amazon.com/bedrock/home. This route
will be redirected to your default region. In this example the region has been set to
`us-west-2`. Anywhere the region is specified can be replaced with your desired region.

Navigate to the [overview](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/overview) page
directly or by clicking on the `Get started` link.

## Foundation Model Access

You will first need to give your AWS account access to the foundation models by
visiting the [modelaccess](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)
page to enable the models you would like to use. 

After enabling the foundation models, navigate to [providers page](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers) 
and select the provider of the model you would like to use. From this page select the specific model you would like to use and 
make note of the `Model ID` (currently located near the bottom) this will be used when using the chat completion example below.

Once that has been enabled set your Access Key and Secret in the env variables:

```shell
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="region-name"
```

Credentials are resolved by boto3, so anything in its [default credential chain](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
also works (`~/.aws/credentials`, instance roles, etc.).

### Region

The region is picked in this order:

1. `region_name` in the provider config
2. `AWS_REGION`
3. `AWS_DEFAULT_REGION`, or the `region` of the active profile in `~/.aws/config`
4. `us-west-2`

### Named profiles and AWS SSO

To use a named profile from `~/.aws/config` (for example one configured with
`aws configure sso`), either export `AWS_PROFILE` or pass `profile_name`:

```shell
aws sso login --profile my-sso-profile
```

```python
import aisuite as ai

client = ai.Client({"aws": {"profile_name": "my-sso-profile"}})
```

## Create a Chat Completion

Install the boto3 client using your package installer

Example with pip
```shell
pip install boto3
```

Example with poetry
```shell
poetry add boto3
```

In your code:
```python
import aisuite as ai
client = ai.Client()


provider = "aws"
model_id = "meta.llama3-1-405b-instruct-v1:0" # Model ID from above

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.chat.completions.create(
    model=f"{provider}:{model_id}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](../CONTRIBUTING.md).





