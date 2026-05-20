"""
s3_loader.py — Read-only access to the production S3 bucket.

IMPORTANT: This module NEVER writes to the production bucket.
  - list_image_keys()    → s3:ListBucket  (read)
  - load_image()         → s3:GetObject   (read)
  - upload_log_to_s3()   → s3:PutObject   to LOG_BUCKET only (separate bucket)
"""

import base64
import logging
import re
from collections import defaultdict
from datetime import date
from io import BytesIO
from pathlib import Path

import boto3
from PIL import Image
from PIL.Image import Image as PilImage

import config

logger = logging.getLogger(__name__)

s3 = boto3.client("s3", region_name=config.AWS_REGION)

# ── Session folder parsing ────────────────────────────────────────────
# Matches folder names like: SCANNER_A_0009bf_2026-03-16_14-10-49
# Captures the date segment (group 1)
_SESSION_DATE_RE = re.compile(
    r"SCANNER_[^/]+_(\d{4}-\d{2}-\d{2})_[^/]+/"
)


def _parse_session_info(key: str) -> tuple[str | None, str | None, date | None]:
    """
    Extract (session_folder, subfolder, session_date) from a full S3 key.

    Expected key structure:
      prior_condition/SCANNER_A_0009bf_2026-03-16_14-10-49/3/frame_001.jpg
      └─ root_prefix ─┘└──────────── session ────────────┘└┘└─ filename ─┘
                                                           subfolder

    Returns:
        (session_folder_name, subfolder_str, date_object),
        or (None, None, None) if the key does not match the expected structure.
    """
    parts = key.split("/")

    # Need at least: root_prefix / session / subfolder / filename
    if len(parts) < 4:
        return None, None, None

    session_folder = parts[-3]
    subfolder      = parts[-2]

    m = _SESSION_DATE_RE.search(key)
    if not m:
        return None, None, None

    try:
        session_date = date.fromisoformat(m.group(1))
    except ValueError:
        return None, None, None

    return session_folder, subfolder, session_date


# ── Main listing function ─────────────────────────────────────────────

def list_image_keys(
    date_from:         date | None        = None,
    date_to:           date | None        = None,
    bucket:            str                = config.S3_BUCKET,
    root_prefix:       str                = config.S3_ROOT_PREFIX,
    target_subfolders: set[str] | None    = None,
) -> list[str]:
    """
    List image keys from target camera subfolders within the date range.
    Performs only s3:ListBucket — no writes to the production bucket.

    S3 key structure:
      <bucket>/prior_condition/SCANNER_<device>_<YYYY-MM-DD>_<time>/<subfolder>/<image>

    Args:
        date_from:          Inclusive start date (default: Monday this week).
        date_to:            Inclusive end date (default: Friday this week).
        bucket:             S3 bucket name.
        root_prefix:        Top-level folder prefix.
        target_subfolders:  Set of subfolder names, e.g. {"3", "7"}.

    Returns:
        Sorted list of matching S3 keys.
    """
    date_from         = date_from         or config.default_date_from()
    date_to           = date_to           or config.default_date_to()
    target_subfolders = target_subfolders or config.S3_TARGET_SUBFOLDERS

    if date_from > date_to:
        raise ValueError(
            f"date_from ({date_from}) must be <= date_to ({date_to})"
        )

    logger.info("[s3] Bucket         : %s  (READ ONLY)", bucket)
    logger.info("[s3] Root prefix    : %s", root_prefix)
    logger.info("[s3] Date range     : %s → %s", date_from, date_to)
    logger.info("[s3] Camera angles  : subfolders %s", sorted(target_subfolders))

    paginator = s3.get_paginator("list_objects_v2")

    # Pass 1: list session folders only (Delimiter avoids descending into objects).
    # Each CommonPrefix looks like: prior_condition/SCANNER_A_0009bf_2026-03-16_14-10-49/
    # Filter by date before touching any actual image files.
    logger.info("[s3] Pass 1: finding sessions in date range...")
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

    logger.info("[s3] Sessions in range: %d", len(matching_sessions))
    if not matching_sessions:
        logger.info("[s3] ✅ Matched     : 0 images\n")
        return []

    # Pass 2: list objects only inside matching sessions × target subfolders.
    # This skips all data outside the date range entirely.
    logger.info("[s3] Pass 2: listing images in matched sessions...")
    keys:     list[str] = []
    skip_ext: int       = 0
    for session_prefix in matching_sessions:
        for subfolder in target_subfolders:
            subfolder_prefix = f"{session_prefix}{subfolder}/"
            for page in paginator.paginate(Bucket=bucket, Prefix=subfolder_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if any(key.lower().endswith(ext) for ext in config.VALID_EXTENSIONS):
                        keys.append(key)
                    else:
                        skip_ext += 1

    keys.sort()

    logger.info("[s3] Skipped (ext)  : %d", skip_ext)
    logger.info("[s3] ✅ Matched     : %d images\n", len(keys))

    _print_session_summary(keys)
    return keys


def _print_session_summary(keys: list[str]) -> None:
    """Log a grouped count of matched S3 keys by session folder and subfolder."""
    if not keys:
        return
    sessions: dict = defaultdict(lambda: defaultdict(int))
    for key in keys:
        session, subfolder, _ = _parse_session_info(key)
        if session:
            sessions[session][subfolder] += 1

    logger.info("[s3] Session summary:")
    for session in sorted(sessions):
        logger.info("  📁 %s", session)
        for sf in sorted(sessions[session]):
            logger.info("       subfolder %s: %d images", sf, sessions[session][sf])
    logger.info("")


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
    obj  = s3.get_object(Bucket=bucket, Key=key)
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
        s3.upload_file(local_path, config.LOG_BUCKET, config.LOG_S3_KEY)
        logger.info(
            "[s3] Log uploaded → s3://%s/%s", config.LOG_BUCKET, config.LOG_S3_KEY
        )
    except Exception as e:
        logger.error("[s3] Log upload failed (non-fatal): %s", e)
        logger.error("[s3] Log is still available locally at: %s", local_path)
