"""Tests for OpenAI-style image content parts in the Anthropic converter."""

import pytest

from aisuite.providers.anthropic_provider import AnthropicMessageConverter

RED_PIXEL_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
    "nGP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
DATA_URL = f"data:image/png;base64,{RED_PIXEL_B64}"


@pytest.fixture
def converter():
    return AnthropicMessageConverter()


def _convert(converter, content):
    _, messages = converter.convert_request([{"role": "user", "content": content}])
    return messages[0]


def test_string_content_passes_through_unchanged(converter):
    message = _convert(converter, "Hello")
    assert message == {"role": "user", "content": "Hello"}


def test_data_url_image_becomes_base64_source_block(converter):
    message = _convert(
        converter, [{"type": "image_url", "image_url": {"url": DATA_URL}}]
    )
    assert message["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": RED_PIXEL_B64,
            },
        }
    ]


def test_http_url_image_becomes_url_source_block(converter):
    url = "https://example.com/cat.jpg"
    message = _convert(converter, [{"type": "image_url", "image_url": {"url": url}}])
    assert message["content"] == [
        {"type": "image", "source": {"type": "url", "url": url}}
    ]


def test_mixed_text_and_image_parts_preserve_order(converter):
    message = _convert(
        converter,
        [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": DATA_URL}},
            {"type": "text", "text": "Answer briefly."},
        ],
    )
    kinds = [block["type"] for block in message["content"]]
    assert kinds == ["text", "image", "text"]
    assert message["content"][0]["text"] == "What is in this image?"


def test_media_type_is_lowercased_from_data_url(converter):
    message = _convert(
        converter,
        [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/JPEG;base64,{RED_PIXEL_B64}"},
            }
        ],
    )
    assert message["content"][0]["source"]["media_type"] == "image/jpeg"


def test_empty_text_parts_are_dropped(converter):
    message = _convert(
        converter,
        [{"type": "text", "text": ""}, {"type": "text", "text": "hi"}],
    )
    assert message["content"] == [{"type": "text", "text": "hi"}]


def test_unsupported_image_scheme_raises(converter):
    with pytest.raises(ValueError, match="Unsupported image_url"):
        _convert(
            converter,
            [{"type": "image_url", "image_url": {"url": "file:///tmp/x.png"}}],
        )


def test_unknown_part_type_raises(converter):
    with pytest.raises(ValueError, match="content part type"):
        _convert(converter, [{"type": "input_audio", "input_audio": {}}])


def test_system_and_tool_paths_are_unaffected(converter):
    """Regression: the parts conversion only touches plain user/assistant content."""
    system, messages = converter.convert_request(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "result"},
        ]
    )
    assert system == "Be brief."
    assert messages[0]["content"] == [{"type": "text", "text": "hi"}]
    assert messages[1]["content"][0]["type"] == "tool_result"
