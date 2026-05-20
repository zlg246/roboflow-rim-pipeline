"""
Tests for vision_verifier (OpenAI GPT-4o).
OpenAI API is fully mocked — no real API calls or costs.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
import vision_verifier

SAMPLE_PREDICTIONS = [
    {"x": 0.5, "y": 0.4, "width": 0.2, "height": 0.15,
     "class": "scratch", "confidence": 0.87}
]


def _mock_response(decision, confidence=0.9, corrections=None):
    """Build a mock OpenAI ChatCompletion response."""
    payload = {
        "decision":    decision,
        "confidence":  confidence,
        "reasoning":   "Mock reasoning for test.",
        "corrections": corrections or [],
    }
    # Mimic OpenAI response structure: response.choices[0].message.content
    message          = MagicMock()
    message.content  = json.dumps(payload)
    choice           = MagicMock()
    choice.message   = message
    response         = MagicMock()
    response.choices = [choice]
    return response


@pytest.mark.parametrize("decision", [
    "correct", "partial", "missed", "other", "null", "bad_quality"
])
def test_all_valid_decisions_parsed(decision):
    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=_mock_response(decision)):
        result = vision_verifier.verify_image(
            "b64data", "image/jpeg", SAMPLE_PREDICTIONS, "test.jpg"
        )
    assert result["decision"] == decision


def test_returns_all_required_fields():
    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=_mock_response("correct")):
        result = vision_verifier.verify_image(
            "b64data", "image/jpeg", SAMPLE_PREDICTIONS, "test.jpg"
        )
    assert "decision"    in result
    assert "confidence"  in result
    assert "reasoning"   in result
    assert "corrections" in result


def test_empty_predictions_still_works():
    """No YOLO predictions — should still return a valid decision (e.g. missed)."""
    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=_mock_response("missed")):
        result = vision_verifier.verify_image(
            "b64data", "image/jpeg", [], "test.jpg"
        )
    assert result["decision"] == "missed"


def test_strips_markdown_json_fence():
    """GPT-4o sometimes wraps JSON in ```json fences — must be stripped."""
    payload = json.dumps({
        "decision": "null", "confidence": 0.95,
        "reasoning": "No damage.", "corrections": []
    })
    message         = MagicMock()
    message.content = f"```json\n{payload}\n```"
    choice          = MagicMock()
    choice.message  = message
    response        = MagicMock()
    response.choices = [choice]

    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=response):
        result = vision_verifier.verify_image(
            "b64data", "image/jpeg", [], "test.jpg"
        )
    assert result["decision"] == "null"


def test_raises_on_invalid_json():
    message         = MagicMock()
    message.content = "this is not valid json"
    choice          = MagicMock()
    choice.message  = message
    response        = MagicMock()
    response.choices = [choice]

    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=response):
        with pytest.raises(json.JSONDecodeError):
            vision_verifier.verify_image("b64data", "image/jpeg", [], "test.jpg")


def test_null_decision_has_empty_corrections():
    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=_mock_response("null", corrections=[])):
        result = vision_verifier.verify_image(
            "b64data", "image/jpeg", [], "test.jpg"
        )
    assert result["corrections"] == []


def test_corrections_structure_for_scratch_decision():
    corrections = [{
        "class": "scratch",
        "x": 0.5, "y": 0.4, "width": 0.2, "height": 0.15, "note": ""
    }]
    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=_mock_response("scratch", corrections=corrections)):
        result = vision_verifier.verify_image(
            "b64data", "image/jpeg", SAMPLE_PREDICTIONS, "test.jpg"
        )
    assert len(result["corrections"]) == 1
    assert result["corrections"][0]["class"] == "scratch"


def test_api_called_with_high_detail_image():
    """Verify the image is sent with detail='high' for better damage detection."""
    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=_mock_response("null")) as mock_create:
        vision_verifier.verify_image("b64data", "image/jpeg", [], "test.jpg")

    call_args = mock_create.call_args
    messages  = call_args.kwargs["messages"]

    # Find the user message content
    user_msg = next(m for m in messages if m["role"] == "user")
    image_block = next(c for c in user_msg["content"] if c["type"] == "image_url")
    assert image_block["image_url"]["detail"] == "high"


def test_system_prompt_is_passed():
    """Verify the system prompt is included in the API call."""
    with patch.object(vision_verifier.client.chat.completions, "create",
                      return_value=_mock_response("null")) as mock_create:
        vision_verifier.verify_image("b64data", "image/jpeg", [], "test.jpg")

    messages = mock_create.call_args.kwargs["messages"]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "quality-control agent" in system_msg["content"]
