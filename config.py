"""Project-wide configuration loaded from environment variables / .env."""

import os
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

# ── S3 (production bucket — READ ONLY) ──────────────────────────────
# S3_BUCKET is the default when --bucket is not passed on the CLI.
# Override per-run with: python rim_pipeline.py --bucket prod-chatswood-toyota
S3_BUCKET = os.environ.get("S3_BUCKET", "prod-castle-hill-toyota")

# ── Rim / Wheel Scanner Pipeline ─────────────────────────────────────
# S3 prefix where wheel scanner images live.
RIM_S3_ROOT_PREFIX = os.environ.get("RIM_S3_ROOT_PREFIX", "wheel_scanner/")

# ── Known buckets (shown in --help) ──────────────────────────────────
KNOWN_BUCKETS = [
    "prod-castle-hill-toyota",
    # add more dealership buckets here as needed
]

# ── Log destination (SEPARATE from production bucket) ────────────────
# Set LOG_BUCKET to a different bucket, or leave empty to log locally only.
LOG_BUCKET = os.environ.get("LOG_BUCKET", "")
LOG_S3_KEY = os.environ.get("LOG_S3_KEY", "pipeline_logs/latest.jsonl")

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
# Rim scratch segmentation model used by rim_pipeline.py.
RIM_YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "rim_segmentation_model.pt")
YOLO_CONF_THRESHOLD = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_IOU_THRESHOLD  = float(os.environ.get("YOLO_IOU_THRESHOLD", "0.70"))

# Douglas-Peucker epsilon (pixels) applied to the raw YOLO segmentation contour.
# Larger → fewer, smoother polygon points.  Smaller → tighter but noisier.
# 2 px means no simplified point is more than 2 px from the original contour.
POLYGON_SIMPLIFY_EPSILON = float(os.environ.get("POLYGON_SIMPLIFY_EPSILON", "2.0"))

# ── Roboflow ─────────────────────────────────────────────────────────
ROBOFLOW_API_KEY     = os.environ.get("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE   = os.environ.get("ROBOFLOW_WORKSPACE", "your-workspace")
RIM_ROBOFLOW_PROJECT = os.environ.get("RIM_ROBOFLOW_PROJECT", "test_project-e4bw5")

# ── Upload ───────────────────────────────────────────────────────────
# JPEG compression quality used when saving images for upload (0–95).
UPLOAD_JPEG_QUALITY = int(os.environ.get("UPLOAD_JPEG_QUALITY", "100"))

# Brief sleep between uploads to avoid hammering the Roboflow API.
UPLOAD_SLEEP = float(os.environ.get("UPLOAD_SLEEP", "0.5"))

# ── EC2 (disabled when running locally) ──────────────────────────────
EC2_SELF_STOP   = os.environ.get("EC2_SELF_STOP", "false").lower() == "true"
EC2_INSTANCE_ID = os.environ.get("EC2_INSTANCE_ID", "")
AWS_REGION      = os.environ.get("AWS_REGION", "ap-southeast-2")
SNS_TOPIC_ARN   = os.environ.get("SNS_TOPIC_ARN", "")
