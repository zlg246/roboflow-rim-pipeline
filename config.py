"""Project-wide configuration loaded from environment variables / .env."""

import os
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

# ── S3 (production buckets — READ ONLY) ─────────────────────────────
# S3_BUCKET is the DEFAULT bucket used when --bucket is not passed on CLI.
# Override per-run with: python pipeline.py --bucket prod-chatswood-toyota
S3_BUCKET            = os.environ.get("S3_BUCKET", "prod-castle-hill-toyota")
S3_ROOT_PREFIX       = os.environ.get("S3_ROOT_PREFIX", "prior_condition/")

# Subfolders representing camera angles to include (3=left, 7=right)
_raw_subfolders      = os.environ.get("S3_TARGET_SUBFOLDERS", "3,7")
S3_TARGET_SUBFOLDERS = set(s.strip() for s in _raw_subfolders.split(",") if s.strip())

VALID_EXTENSIONS     = {".jpg", ".jpeg", ".png", ".webp"}

# ── Known buckets (used for tab-completion hint in --help) ────────────
# Add all your dealership buckets here for reference — not enforced in code.
KNOWN_BUCKETS = [
    "prod-castle-hill-toyota",
    # "prod-chatswood-toyota",
    # add more as needed
]

# ── Log destination (SEPARATE from production bucket) ────────────────
# Set LOG_BUCKET to a different bucket, or leave empty to log locally only.
# The production bucket is NEVER written to.
LOG_BUCKET           = os.environ.get("LOG_BUCKET", "")          # empty = local only
LOG_S3_KEY           = os.environ.get("LOG_S3_KEY", "pipeline_logs/latest.jsonl")

# ── Date range defaults (Monday–Friday of current week) ──────────────

def _current_week_monday() -> date:
    """Return Monday of the current ISO week."""
    today = date.today()
    return today - timedelta(days=today.weekday())


def default_date_from() -> date:
    """Return the default start date: Monday of the current week."""
    return _current_week_monday()


def default_date_to() -> date:
    """Return the default end date: Friday of the current week."""
    return _current_week_monday() + timedelta(days=4)


# ── YOLO ─────────────────────────────────────────────────────────────
YOLO_MODEL_PATH      = os.environ.get("YOLO_MODEL_PATH", "best.pt")
YOLO_CONF_THRESHOLD  = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))

# ── Vision provider: "openai" or "claude" ────────────────────────────
VISION_PROVIDER      = os.environ.get("VISION_PROVIDER", "claude")

_VALID_VISION_PROVIDERS = {"openai", "claude"}
if VISION_PROVIDER not in _VALID_VISION_PROVIDERS:
    raise ValueError(
        f"VISION_PROVIDER must be one of {_VALID_VISION_PROVIDERS}, "
        f"got {VISION_PROVIDER!r}"
    )

# ── OpenAI ───────────────────────────────────────────────────────────
OPENAI_API_KEY          = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL            = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_MAX_TOKENS       = 1024
OPENAI_RATE_LIMIT_SLEEP = 0.5

# ── Anthropic / Claude ───────────────────────────────────────────────
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL      = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_MAX_TOKENS = 1024

# ── Roboflow ─────────────────────────────────────────────────────────
ROBOFLOW_API_KEY     = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE   = os.environ.get("ROBOFLOW_WORKSPACE", "your-workspace")
ROBOFLOW_PROJECT     = os.environ.get("ROBOFLOW_PROJECT", "panel_damage_bb")

# ── Pipeline ─────────────────────────────────────────────────────────
UPLOAD_DECISIONS     = {"scratch", "other"}
SKIP_DECISIONS       = {"null"}

SAVE_NULL_REVIEW     = os.environ.get("SAVE_NULL_REVIEW", "true").lower() == "true"

# JPEG compression quality used when saving images (0–95).
UPLOAD_JPEG_QUALITY      = int(os.environ.get("UPLOAD_JPEG_QUALITY", "100"))
NULL_REVIEW_JPEG_QUALITY = int(os.environ.get("NULL_REVIEW_JPEG_QUALITY", "90"))

# ── EC2 (disabled when running locally) ──────────────────────────────
EC2_SELF_STOP        = os.environ.get("EC2_SELF_STOP", "false").lower() == "true"
EC2_INSTANCE_ID      = os.environ.get("EC2_INSTANCE_ID", "")
AWS_REGION           = os.environ.get("AWS_REGION", "ap-southeast-2")
SNS_TOPIC_ARN        = os.environ.get("SNS_TOPIC_ARN", "")
