"""
s3_loader.py — Read-only access to the production S3 bucket.

IMPORTANT: This module NEVER writes to the production bucket.
  - list_rim_image_keys() → s3:ListBucket  (read)
  - load_image()          → s3:GetObject   (read)
  - upload_log_to_s3()    → s3:PutObject   to LOG_BUCKET only (separate bucket)
"""

import base64
import logging
import re
from datetime import date
from io import BytesIO
from pathlib import Path

import boto3
from PIL import Image
from PIL.Image import Image as PilImage

import config
from credentials import get_session

logger = logging.getLogger(__name__)


def _get_client(bucket: str):
    """Return an S3 client using the appropriate credentials for the given bucket."""
    return get_session(bucket).client("s3", region_name=config.AWS_REGION)


# ── Session folder parsing ────────────────────────────────────────────
# Matches session folder names like: SCANNER_A_0009bf_2026-03-16_14-10-49
# Captures the date segment (group 1).
_SESSION_DATE_RE = re.compile(
    r"SCANNER_[^/]+_(\d{4}-\d{2}-\d{2})_[^/]+/"
)


# ── Rim wheel scanner listing ─────────────────────────────────────────

def list_rim_image_keys(
    date_from:   date | None = None,
    date_to:     date | None = None,
    bucket:      str         = config.S3_BUCKET,
    root_prefix: str         = config.RIM_S3_ROOT_PREFIX,
) -> list[str]:
    """
    List wheel scanner org_image keys from the given date range.

    S3 key structure::

        <bucket>/wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/left_rear_wheel_org_image.jpg

    Only files ending with ``_org_image.jpg`` are returned; overlay images
    (``_overlay.jpg``, etc.) are silently skipped.

    Args:
        date_from:   Inclusive start date (default: Monday this week).
        date_to:     Inclusive end date (default: Friday this week).
        bucket:      S3 bucket name.
        root_prefix: Top-level folder prefix (default: ``wheel_scanner/``).

    Returns:
        Sorted list of matching S3 keys.
    """
    date_from = date_from or config.default_date_from()
    date_to   = date_to   or config.default_date_to()

    if date_from > date_to:
        raise ValueError(f"date_from ({date_from}) must be <= date_to ({date_to})")

    logger.info("[s3-rim] Bucket       : %s  (READ ONLY)", bucket)
    logger.info("[s3-rim] Root prefix  : %s", root_prefix)
    logger.info("[s3-rim] Date range   : %s → %s", date_from, date_to)

    s3        = _get_client(bucket)
    paginator = s3.get_paginator("list_objects_v2")

    # Pass 1: find session folders within the date range.
    # CommonPrefixes look like: wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/
    logger.info("[s3-rim] Pass 1: finding sessions in date range...")
    matching_sessions: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            session_prefix = cp["Prefix"]
            m = _SESSION_DATE_RE.search(session_prefix)
            if not m:
                continue
            try:
                session_date = date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if date_from <= session_date <= date_to:
                matching_sessions.append(session_prefix)

    logger.info("[s3-rim] Sessions in range: %d", len(matching_sessions))
    if not matching_sessions:
        logger.info("[s3-rim] ✅ Matched   : 0 images\n")
        return []

    # Pass 2: list org_image files directly inside each session folder.
    # There is no numeric subfolder for wheel scanner images.
    logger.info("[s3-rim] Pass 2: listing org images in matched sessions...")
    keys:    list[str] = []
    skipped: int       = 0
    for session_prefix in matching_sessions:
        for page in paginator.paginate(Bucket=bucket, Prefix=session_prefix):
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                filename = key.rsplit("/", 1)[-1]
                if filename.endswith("_org_image.jpg"):
                    keys.append(key)
                else:
                    skipped += 1

    keys.sort()
    logger.info("[s3-rim] Skipped (non-org): %d", skipped)
    logger.info("[s3-rim] ✅ Matched   : %d images\n", len(keys))
    return keys


# ── Image loading (read-only) ─────────────────────────────────────────

def load_image(key: str, bucket: str = config.S3_BUCKET) -> tuple[PilImage, str, str]:
    """
    Download one image from S3 (read-only GetObject).

    Args:
        key:    S3 object key.
        bucket: S3 bucket name.

    Returns:
        (pil_image, base64_str, media_type)
    """
    obj  = _get_client(bucket).get_object(Bucket=bucket, Key=key)
    raw  = obj["Body"].read()
    ext  = key.lower().rsplit(".", 1)[-1]
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    pil  = Image.open(BytesIO(raw)).convert("RGB")
    b64  = base64.standard_b64encode(raw).decode("utf-8")
    return pil, b64, mime


def load_image_from_local(local_path: str | Path) -> tuple[PilImage, str, str]:
    """
    Load an image from local disk.

    Used for testing without S3 access.

    Args:
        local_path: Path to the image file.

    Returns:
        (pil_image, base64_str, media_type)
    """
    path = Path(local_path)
    raw  = path.read_bytes()
    ext  = path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    pil  = Image.open(BytesIO(raw)).convert("RGB")
    b64  = base64.standard_b64encode(raw).decode("utf-8")
    return pil, b64, mime


# ── Log upload (writes to LOG_BUCKET only — never to production) ──────

def upload_log_to_s3(local_path: str) -> None:
    """
    Upload the pipeline log to LOG_BUCKET.

    Skips the upload and keeps the log local if LOG_BUCKET is empty.
    Refuses to write if LOG_BUCKET equals S3_BUCKET (production).
    NEVER writes to the production S3 bucket.

    Args:
        local_path: Path to the local JSONL log file.
    """
    if not config.LOG_BUCKET:
        logger.info("[s3] LOG_BUCKET not set — log kept local only: %s", local_path)
        return

    if config.LOG_BUCKET == config.S3_BUCKET:
        logger.warning(
            "[s3] ⚠️  LOG_BUCKET equals S3_BUCKET (production). "
            "Skipping log upload to protect production bucket. "
            "Set LOG_BUCKET to a different bucket in .env."
        )
        return

    try:
        _get_client(config.LOG_BUCKET).upload_file(
            local_path, config.LOG_BUCKET, config.LOG_S3_KEY,
        )
        logger.info(
            "[s3] Log uploaded → s3://%s/%s", config.LOG_BUCKET, config.LOG_S3_KEY,
        )
    except Exception as e:
        logger.error("[s3] Log upload failed (non-fatal): %s", e)
        logger.error("[s3] Log is still available locally at: %s", local_path)
