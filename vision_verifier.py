"""
vision_verifier.py — Vision quality-control layer.
Supports OpenAI (GPT-4o) and Anthropic (Claude) backends.
Set VISION_PROVIDER="openai" or "claude" in .env to choose.
"""

import json
import logging
from typing import TYPE_CHECKING, Any

import config
from pipeline_types import VisionResult

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI

logger = logging.getLogger(__name__)

_MD_FENCE = "```"
_JSON_TAG  = "json"

SYSTEM_PROMPT = """
You are a quality-control agent for a vehicle panel damage annotation pipeline.
Frames come from a fixed camera recording vehicles passing through a scanner.
Uploaded images go to Roboflow where human annotators make the final labelling call.

CORE BIAS: always upload in-panel detections. A false positive costs one human review.
A missed scratch is lost forever. When uncertain, upload — never null.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLO MODEL CONTEXT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOLO predicts a single class — "scratch" — whenever it detects surface-level damage.
You receive its bounding boxes as normalised (centre-x, centre-y, width, height) coords.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SURFACE DAMAGE — class "scratch"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Scratches, scuffs, paint transfer from contact
  ✅ Paint chips, flaking, exposed primer or bare metal
  ✅ Deep scrapes, gouges, keymarks
  ✅ Any mark where the paint or clear coat is visibly broken

Detected but NOT genuine scratch damage → still upload as class "other":
  • Dents or creases with no paint break
  • Panel gaps, misalignment
  • Shadows, reflections, dirt, water marks that YOLO flagged

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For EACH YOLO prediction, reason explicitly before deciding:
  1. Does this bounding box fall on a vehicle panel surface?
       YES or UNCERTAIN → include in corrections:
           · looks like surface damage    → class "scratch"
           · does not look like surface damage → class "other"
       CLEARLY NO (box is on road surface, open air, background structure,
                   or a person — with no vehicle panel visible at all) → discard

Then scan the full image for any surface damage YOLO missed; add each as class "scratch".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  scratch — corrections contains at least one "scratch" box
  other   — corrections contains at least one box and none are "scratch"
  null    — corrections is empty because every prediction was CLEARLY off the vehicle
            body entirely, OR the image is completely unassessable (pitch-black /
            extreme blur / pure-white overexposure).

  ⛔ null is NOT permitted when:
     • Any YOLO prediction overlaps any vehicle panel surface, even partially
     • You are merely uncertain whether the detection is genuine scratch damage
     • The detection looks like a dent, shadow, or dirt  (→ use class "other")
     • The YOLO confidence is low  (→ still upload as "other" for human review)

Decision priority: scratch > other > null

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — valid JSON only, no markdown fences, no extra text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "decision":    "<scratch|other|null>",
  "confidence":  <float 0.0-1.0>,
  "reasoning":   "<for each YOLO prediction state: on-panel yes/no and kept/discarded with reason>",
  "corrections": [
    {
      "class":  "<scratch|other>",
      "x":      <normalised centre-x 0.0-1.0>,
      "y":      <normalised centre-y 0.0-1.0>,
      "width":  <normalised width 0.0-1.0>,
      "height": <normalised height 0.0-1.0>,
      "note":   "<optional short reason>"
    }
  ]
}
For decision "null", corrections MUST be [].
"""


# ── OpenAI backend ────────────────────────────────────────────────────

_openai_client: "OpenAI | None" = None


def _get_openai_client() -> "OpenAI":
    """Lazily initialise and cache the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _call_openai(b64_image: str, media_type: str,
                 pred_text: str, filename: str) -> str:
    """Send image + predictions to GPT-4o Vision and return raw response text."""
    client   = _get_openai_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        max_tokens=config.OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url":    f"data:{media_type};base64,{b64_image}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": f"Filename: {filename}\n\nYOLO predictions:\n{pred_text}",
                },
            ]},
        ],
    )
    return response.choices[0].message.content.strip()


# ── Anthropic / Claude backend ────────────────────────────────────────

_anthropic_client: "Anthropic | None" = None


def _get_anthropic_client() -> "Anthropic":
    """Lazily initialise and cache the Anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _anthropic_client


def _call_claude(b64_image: str, media_type: str,
                 pred_text: str, filename: str) -> str:
    """Send image + predictions to Claude Vision and return raw response text."""
    client   = _get_anthropic_client()
    response = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.ANTHROPIC_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       b64_image,
                    },
                },
                {
                    "type": "text",
                    "text": f"Filename: {filename}\n\nYOLO predictions:\n{pred_text}",
                },
            ]},
        ],
    )
    return response.content[0].text.strip()


# ── Public interface ──────────────────────────────────────────────────

def verify_image(
    b64_image:   str,
    media_type:  str,
    predictions: list[dict[str, Any]],
    filename:    str,
) -> VisionResult:
    """
    Send image + YOLO predictions to the configured vision model for verification.

    Provider is selected by config.VISION_PROVIDER ("openai" or "claude").

    Args:
        b64_image:   Base64-encoded image string.
        media_type:  MIME type, e.g. "image/jpeg".
        predictions: List of YOLO prediction dicts (may be empty).
        filename:    Original S3 key or filename (included in the prompt for context).

    Returns:
        Parsed VisionResult with keys: decision, confidence, reasoning, corrections.
    """
    pred_text = (
        json.dumps(predictions, indent=2)
        if predictions
        else "No predictions produced by the YOLO model."
    )

    if config.VISION_PROVIDER == "claude":
        raw = _call_claude(b64_image, media_type, pred_text, filename)
    else:
        raw = _call_openai(b64_image, media_type, pred_text, filename)

    # Strip accidental markdown fences some models add despite instructions.
    if raw.startswith(_MD_FENCE):
        parts = raw.split(_MD_FENCE)
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith(_JSON_TAG):
            raw = raw[len(_JSON_TAG):]

    return json.loads(raw.strip())
