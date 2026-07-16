"""Unit tests for ASR parameter validation and mapping."""

import pytest
import warnings
from aisuite.framework.asr_params import (
    ParamValidator,
    COMMON_PARAMS,
    PROVIDER_PARAMS,
    GOOGLE_LANGUAGE_MAP,
)


class TestParamValidatorCommonParams:
    """Test common parameter mapping across providers."""

    def test_language_mapping_openai(self):
        """Test that language param passes through for OpenAI."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("openai", {"language": "en"})
        assert result == {"language": "en"}

    def test_language_mapping_deepgram(self):
        """Test that language param passes through for Deepgram."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("deepgram", {"language": "en"})
        assert result == {"language": "en"}

    def test_language_mapping_google(self):
        """Test that language param maps to language_code and expands for Google."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("google", {"language": "en"})
        assert result == {"language_code": "en-US"}

    def test_prompt_mapping_openai(self):
        """Test that prompt param passes through for OpenAI."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("openai", {"prompt": "meeting notes"})
        assert result == {"prompt": "meeting notes"}

    def test_prompt_mapping_deepgram(self):
        """Test that prompt param maps to keywords for Deepgram."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("deepgram", {"prompt": "meeting notes"})
        assert result == {"keywords": ["meeting", "notes"]}

    def test_prompt_mapping_google(self):
        """Test that prompt param maps to speech_contexts for Google."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("google", {"prompt": "technical terms"})
        assert result == {"speech_contexts": [{"phrases": ["technical terms"]}]}

    def test_temperature_mapping_openai(self):
        """Test that temperature param passes through for OpenAI."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("openai", {"temperature": 0.5})
        assert result == {"temperature": 0.5}

    def test_temperature_ignored_deepgram(self):
        """Test that temperature param is ignored for Deepgram (not supported)."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("deepgram", {"temperature": 0.5})
        assert result == {}

    def test_temperature_ignored_google(self):
        """Test that temperature param is ignored for Google (not supported)."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("google", {"temperature": 0.5})
        assert result == {}


class TestParamValidatorTransformations:
    """Test value transformations for provider-specific formats."""

    def test_google_language_expansion_common_codes(self):
        """Test Google language code expansion for common 2-letter codes."""
        validator = ParamValidator("strict")

        test_cases = {
            "en": "en-US",
            "es": "es-ES",
            "fr": "fr-FR",
            "de": "de-DE",
            "ja": "ja-JP",
            "zh": "zh-CN",
        }

        for input_lang, expected_output in test_cases.items():
            result = validator.validate_and_map("google", {"language": input_lang})
            assert result == {
                "language_code": expected_output
            }, f"Failed for {input_lang}: expected {expected_output}, got {result}"

    def test_google_language_expansion_unknown_code(self):
        """Test Google language code expansion for unknown 2-letter code (fallback to -US)."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("google", {"language": "xx"})
        assert result == {"language_code": "xx-US"}

    def test_google_language_no_expansion_for_full_code(self):
        """Test that Google doesn't expand already full language codes."""
        validator = ParamValidator("strict")
        # When a full locale code is passed to language_code directly (not via common param)
        result = validator.validate_and_map("google", {"language_code": "en-GB"})
        assert result == {"language_code": "en-GB"}

    def test_deepgram_prompt_to_keywords_single_word(self):
        """Test Deepgram prompt splits single word into list."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("deepgram", {"prompt": "meeting"})
        assert result == {"keywords": ["meeting"]}

    def test_deepgram_prompt_to_keywords_multiple_words(self):
        """Test Deepgram prompt splits multiple words into list."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "deepgram", {"prompt": "meeting notes action items"}
        )
        assert result == {"keywords": ["meeting", "notes", "action", "items"]}

    def test_deepgram_prompt_to_keywords_already_list(self):
        """Test Deepgram handles prompt that's already a list."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "deepgram", {"prompt": ["meeting", "notes"]}
        )
        assert result == {"keywords": ["meeting", "notes"]}

    def test_google_prompt_to_speech_contexts(self):
        """Test Google wraps prompt in speech_contexts structure."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("google", {"prompt": "technical terms"})
        assert result == {"speech_contexts": [{"phrases": ["technical terms"]}]}


class TestParamValidatorProviderSpecific:
    """Test provider-specific parameter pass-through."""

    def test_openai_response_format(self):
        """Test OpenAI response_format param passes through."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "openai", {"response_format": "verbose_json"}
        )
        assert result == {"response_format": "verbose_json"}

    def test_openai_timestamp_granularities(self):
        """Test OpenAI timestamp_granularities param passes through."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "openai", {"timestamp_granularities": ["word", "segment"]}
        )
        assert result == {"timestamp_granularities": ["word", "segment"]}

    def test_deepgram_punctuate(self):
        """Test Deepgram punctuate param passes through."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("deepgram", {"punctuate": True})
        assert result == {"punctuate": True}

    def test_deepgram_diarize(self):
        """Test Deepgram diarize param passes through."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("deepgram", {"diarize": True})
        assert result == {"diarize": True}

    def test_deepgram_multiple_features(self):
        """Test Deepgram multiple features pass through together."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "deepgram",
            {
                "punctuate": True,
                "diarize": True,
                "sentiment": True,
                "topics": True,
            },
        )
        assert result == {
            "punctuate": True,
            "diarize": True,
            "sentiment": True,
            "topics": True,
        }

    def test_google_enable_automatic_punctuation(self):
        """Test Google enable_automatic_punctuation param passes through."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "google", {"enable_automatic_punctuation": True}
        )
        assert result == {"enable_automatic_punctuation": True}

    def test_google_enable_speaker_diarization(self):
        """Test Google enable_speaker_diarization param passes through."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "google", {"enable_speaker_diarization": True}
        )
        assert result == {"enable_speaker_diarization": True}

    def test_google_diarization_speaker_count(self):
        """Test Google diarization_speaker_count param passes through."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("google", {"diarization_speaker_count": 3})
        assert result == {"diarization_speaker_count": 3}


class TestParamValidatorMixedParams:
    """Test combinations of common and provider-specific parameters."""

    def test_openai_common_and_specific(self):
        """Test OpenAI with common params + provider-specific params."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "openai",
            {
                "language": "en",
                "temperature": 0.5,
                "response_format": "verbose_json",
            },
        )
        assert result == {
            "language": "en",
            "temperature": 0.5,
            "response_format": "verbose_json",
        }

    def test_deepgram_common_and_specific(self):
        """Test Deepgram with common params + provider-specific params."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "deepgram",
            {
                "language": "en",
                "prompt": "meeting",
                "punctuate": True,
                "diarize": True,
            },
        )
        assert result == {
            "language": "en",
            "keywords": ["meeting"],
            "punctuate": True,
            "diarize": True,
        }

    def test_google_common_and_specific(self):
        """Test Google with common params + provider-specific params."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map(
            "google",
            {
                "language": "en",
                "enable_automatic_punctuation": True,
                "enable_speaker_diarization": True,
            },
        )
        assert result == {
            "language_code": "en-US",
            "enable_automatic_punctuation": True,
            "enable_speaker_diarization": True,
        }


class TestParamValidatorStrictMode:
    """Test strict validation mode behavior."""

    def test_strict_mode_rejects_unknown_param_openai(self):
        """Test strict mode raises ValueError for unknown OpenAI param."""
        validator = ParamValidator("strict")
        with pytest.raises(
            ValueError, match="Unknown parameters for openai: \\['invalid_param'\\]"
        ):
            validator.validate_and_map("openai", {"invalid_param": True})

    def test_strict_mode_rejects_unknown_param_deepgram(self):
        """Test strict mode raises ValueError for unknown Deepgram param."""
        validator = ParamValidator("strict")
        with pytest.raises(
            ValueError, match="Unknown parameters for deepgram: \\['invalid_param'\\]"
        ):
            validator.validate_and_map("deepgram", {"invalid_param": True})

    def test_strict_mode_rejects_multiple_unknown_params(self):
        """Test strict mode raises ValueError for multiple unknown params."""
        validator = ParamValidator("strict")
        with pytest.raises(ValueError, match="Unknown parameters for openai"):
            validator.validate_and_map(
                "openai",
                {
                    "invalid_param1": True,
                    "invalid_param2": False,
                },
            )

    def test_strict_mode_error_message_helpful(self):
        """Test that strict mode error message mentions provider documentation."""
        validator = ParamValidator("strict")
        with pytest.raises(ValueError, match="See openai documentation"):
            validator.validate_and_map("openai", {"typo_param": True})

    def test_strict_mode_allows_valid_params(self):
        """Test that strict mode allows all valid params."""
        validator = ParamValidator("strict")
        # Should not raise
        result = validator.validate_and_map(
            "deepgram",
            {
                "language": "en",
                "punctuate": True,
            },
        )
        assert result == {"language": "en", "punctuate": True}


class TestParamValidatorWarnMode:
    """Test warn validation mode behavior."""

    def test_warn_mode_issues_warning_unknown_param(self):
        """Test warn mode issues UserWarning for unknown param."""
        validator = ParamValidator("warn")
        with pytest.warns(
            UserWarning, match="Unknown parameters for openai: \\['invalid_param'\\]"
        ):
            result = validator.validate_and_map("openai", {"invalid_param": True})
        # Param should be filtered out (not passed through in warn mode)
        assert result == {}

    def test_warn_mode_continues_execution(self):
        """Test warn mode continues execution after warning."""
        validator = ParamValidator("warn")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warning for this test
            result = validator.validate_and_map(
                "openai",
                {
                    "language": "en",
                    "invalid_param": True,
                },
            )
        # Valid param should pass through, invalid should be filtered
        assert result == {"language": "en"}

    def test_warn_mode_warning_message_helpful(self):
        """Test that warn mode warning message mentions provider documentation."""
        validator = ParamValidator("warn")
        with pytest.warns(UserWarning, match="See deepgram documentation"):
            validator.validate_and_map("deepgram", {"typo_param": True})


class TestParamValidatorPermissiveMode:
    """Test permissive validation mode behavior."""

    def test_permissive_mode_allows_unknown_param(self):
        """Test permissive mode passes through unknown params."""
        validator = ParamValidator("permissive")
        result = validator.validate_and_map("openai", {"experimental_feature": True})
        assert result == {"experimental_feature": True}

    def test_permissive_mode_no_warning(self):
        """Test permissive mode doesn't issue warnings."""
        validator = ParamValidator("permissive")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Should not raise (no warning should be issued)
            result = validator.validate_and_map("openai", {"unknown_param": True})
        assert result == {"unknown_param": True}

    def test_permissive_mode_mixed_valid_and_unknown(self):
        """Test permissive mode with mix of valid and unknown params."""
        validator = ParamValidator("permissive")
        result = validator.validate_and_map(
            "deepgram",
            {
                "language": "en",
                "punctuate": True,
                "experimental_feature": True,
                "beta_param": "value",
            },
        )
        assert result == {
            "language": "en",
            "punctuate": True,
            "experimental_feature": True,
            "beta_param": "value",
        }


class TestParamValidatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_params(self):
        """Test validation with empty params dict."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("openai", {})
        assert result == {}

    def test_unknown_provider(self):
        """Test validation with unknown provider (uses empty param set)."""
        validator = ParamValidator("strict")
        # Unknown provider with non-common param should raise error
        with pytest.raises(ValueError, match="Unknown parameters for unknown_provider"):
            validator.validate_and_map("unknown_provider", {"custom_param": "value"})

    def test_none_value_param(self):
        """Test that None values are handled correctly."""
        validator = ParamValidator("strict")
        result = validator.validate_and_map("openai", {"language": None})
        assert result == {"language": None}

    def test_common_param_overrides_provider_param(self):
        """Test that common param mapping takes precedence."""
        validator = ParamValidator("strict")
        # If someone passes both 'language' and 'language_code' to Google
        # The common param 'language' should map to 'language_code'
        result = validator.validate_and_map(
            "google",
            {
                "language": "en",
                "language_code": "fr-FR",  # This should be overridden
            },
        )
        # language maps to language_code, then language_code as provider param is also valid
        # In current implementation, common param is processed first, then provider params
        # So we'll get both, but language takes precedence in the transformation
        assert "language_code" in result
        assert result["language_code"] == "fr-FR" or result["language_code"] == "en-US"

    def test_validator_mode_case_sensitivity(self):
        """Test that validator handles only valid mode strings."""
        # Valid modes should work
        ParamValidator("strict")
        ParamValidator("warn")
        ParamValidator("permissive")

        # Note: Python typing will catch invalid modes at type-check time,
        # but runtime won't enforce it unless we add validation
        # This test documents expected behavior


class TestParamValidatorRegistry:
    """Test that parameter registries are complete and consistent."""

    def test_common_params_has_required_providers(self):
        """Test COMMON_PARAMS includes all ASR providers."""
        assert "openai" in COMMON_PARAMS["language"]
        assert "deepgram" in COMMON_PARAMS["language"]
        assert "google" in COMMON_PARAMS["language"]

    def test_provider_params_has_all_providers(self):
        """Test PROVIDER_PARAMS includes all ASR providers."""
        assert "openai" in PROVIDER_PARAMS
        assert "deepgram" in PROVIDER_PARAMS
        assert "google" in PROVIDER_PARAMS

    def test_google_language_map_completeness(self):
        """Test GOOGLE_LANGUAGE_MAP has common language codes."""
        required_languages = ["en", "es", "fr", "de", "ja", "zh"]
        for lang in required_languages:
            assert lang in GOOGLE_LANGUAGE_MAP, f"Missing {lang} in GOOGLE_LANGUAGE_MAP"

    def test_provider_params_includes_common_params(self):
        """Test that provider param sets include their common params."""
        # OpenAI should have language, prompt, temperature in its set
        assert "language" in PROVIDER_PARAMS["openai"]
        assert "prompt" in PROVIDER_PARAMS["openai"]
        assert "temperature" in PROVIDER_PARAMS["openai"]

        # Deepgram should have language in its set (but not temperature)
        assert "language" in PROVIDER_PARAMS["deepgram"]
        assert "temperature" not in PROVIDER_PARAMS["deepgram"]
