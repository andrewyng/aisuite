"""Tests for AWS Bedrock provider configuration (profiles and region resolution)."""

import pytest

from aisuite.providers.aws_provider import AwsProvider, BedrockConfig

AWS_CONFIG = """\
[default]
region = eu-central-1

[profile sso-dev]
region = ap-southeast-2
aws_access_key_id = dev-access-key
aws_secret_access_key = dev-secret-key

[profile no-region]
aws_access_key_id = other-access-key
aws_secret_access_key = other-secret-key
"""


@pytest.fixture(autouse=True)
def isolated_aws_env(monkeypatch, tmp_path):
    """Point boto3 at a throwaway config so the developer's ~/.aws is never read."""
    config_file = tmp_path / "config"
    config_file.write_text(AWS_CONFIG)
    credentials_file = tmp_path / "credentials"
    credentials_file.write_text("")

    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    for var in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    return config_file


def test_region_from_config_takes_precedence(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")

    config = BedrockConfig(region_name="ca-central-1")

    assert config.region_name == "ca-central-1"


def test_aws_region_env_beats_default_region(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")

    assert BedrockConfig().region_name == "us-east-1"


def test_aws_default_region_env_is_honored(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")

    assert BedrockConfig().region_name == "us-east-2"


def test_region_from_default_profile_is_honored():
    assert BedrockConfig().region_name == "eu-central-1"


def test_falls_back_to_us_west_2_when_nothing_is_configured(isolated_aws_env):
    isolated_aws_env.write_text("")

    assert BedrockConfig().region_name == "us-west-2"


def test_profile_name_from_config_selects_profile():
    config = BedrockConfig(profile_name="sso-dev")

    assert config.session.profile_name == "sso-dev"
    assert config.region_name == "ap-southeast-2"
    credentials = config.session.get_credentials()
    assert credentials.access_key == "dev-access-key"
    assert credentials.secret_key == "dev-secret-key"


def test_aws_profile_env_selects_profile(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "sso-dev")

    config = BedrockConfig()

    assert config.session.profile_name == "sso-dev"
    assert config.region_name == "ap-southeast-2"


def test_profile_without_region_falls_back_to_default_region():
    assert BedrockConfig(profile_name="no-region").region_name == "us-west-2"


def test_explicit_region_overrides_profile_region():
    config = BedrockConfig(profile_name="sso-dev", region_name="us-east-1")

    assert config.region_name == "us-east-1"


def test_provider_client_uses_profile_session():
    provider = AwsProvider(profile_name="sso-dev")

    assert provider.client.meta.region_name == "ap-southeast-2"
    credentials = provider.client._request_signer._credentials
    assert credentials.access_key == "dev-access-key"


def test_unknown_profile_raises():
    from botocore.exceptions import ProfileNotFound

    with pytest.raises(ProfileNotFound):
        BedrockConfig(profile_name="does-not-exist")
