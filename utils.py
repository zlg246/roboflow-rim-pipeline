"""Shared utility functions for the rim pipeline."""

import argparse
import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import boto3

import config
from pipeline_types import PipelineRecord

logger = logging.getLogger(__name__)

# Module-level log path (monkeypatched by tests)
LOG_PATH = Path(f"logs/pipeline_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl")


# ── Bucket helpers ───────────────────────────────────────────────────

def _extract_dealership(bucket: str) -> str:
    """
    Extract a clean dealership identifier from a bucket name.

    Handles two naming conventions::

        prod-castle-hill-toyota                         → castle_hill_toyota
        cmt-prod-ap-southeast-2-melton-toyota           → melton_toyota
    """
    new_prefix = f"cmt-prod-{config.AWS_REGION}-"
    if bucket.startswith(new_prefix):
        return bucket.removeprefix(new_prefix).replace("-", "_")
    return bucket.removeprefix("prod-").replace("-", "_")


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_date_arg(value: str) -> date:
    """
    Accept flexible date inputs.

    Supported formats:
      2026-03-16     → explicit ISO date
      today          → date.today()
      yesterday      → today - 1 day
      monday         → this week's Monday
      friday         → this week's Friday
      last-monday    → most recent past Monday
      last-friday    → most recent past Friday
      (same pattern for all weekday names)
    """
    v     = value.strip().lower()
    today = date.today()

    if v == "today":
        return today
    if v == "yesterday":
        return today - timedelta(days=1)

    weekday_names = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }

    for name, wd in weekday_names.items():
        if v == name:
            delta = (today.weekday() - wd) % 7
            return today - timedelta(days=delta)
        if v == f"last-{name}":
            delta = (today.weekday() - wd) % 7
            delta = delta if delta > 0 else 7
            return today - timedelta(days=delta)

    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD or a keyword: "
            "today, yesterday, monday, friday, last-friday, etc."
        )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rim scratch annotation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--from", dest="date_from", type=_parse_date_arg, default=None,
        metavar="DATE",
        help="Start date inclusive. Default: Monday of current week.",
    )
    parser.add_argument(
        "--to", dest="date_to", type=_parse_date_arg, default=None,
        metavar="DATE",
        help="End date inclusive. Default: Friday of current week.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        metavar="N",
        help="Process at most N images. Useful for testing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run YOLO but skip all Roboflow uploads.",
    )
    parser.add_argument(
        "--skip-rim-scratch", action="store_true",
        help=(
            "Disable upload to the rim_scratch project "
            f"({config.RIM_SCRATCH_ROBOFLOW_PROJECT}). "
            "YOLO inference still runs."
        ),
    )
    parser.add_argument(
        "--skip-rim-seg", action="store_true",
        help=(
            "Disable upload to the rim_seg project "
            f"({config.RIM_SEG_ROBOFLOW_PROJECT})."
        ),
    )
    parser.add_argument(
        "--bucket", type=str, default=None,
        metavar="BUCKET",
        help=(
            "S3 bucket to process. Default: S3_BUCKET from .env. "
            f"Known buckets: {config.KNOWN_BUCKETS}"
        ),
    )
    return parser.parse_args()


# ── Logging ───────────────────────────────────────────────────────────

def _make_log_path() -> Path:
    """Ensure the logs/ directory exists and return the current run's log path."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    return LOG_PATH


def _load_processed(log_path: Path) -> set[str]:
    """
    Read already-processed S3 keys from an existing log file.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        Set of s3_key strings already present in the log.
    """
    done: set[str] = set()
    if not log_path.exists():
        return done
    try:
        lines = log_path.read_text().splitlines()
    except OSError as e:
        logger.error("[pipeline] Cannot read log file %s: %s", log_path, e)
        raise
    for lineno, line in enumerate(lines, 1):
        try:
            done.add(json.loads(line)["s3_key"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("[pipeline] Skipping malformed log line %d: %s", lineno, e)
    return done


def _log(log_path: Path, record: PipelineRecord) -> None:
    """Append one JSON record to the JSONL log file."""
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Notifications & infrastructure ────────────────────────────────────

def _notify(subject: str, message: str) -> None:
    """Publish a summary notification to SNS (no-op when SNS_TOPIC_ARN is unset)."""
    if not config.SNS_TOPIC_ARN:
        return
    try:
        boto3.client("sns", region_name=config.AWS_REGION).publish(
            TopicArn=config.SNS_TOPIC_ARN,
            Subject=subject,
            Message=message,
        )
        logger.info("[notify] Email sent: %s", subject)
    except Exception as e:
        logger.warning("[notify] SNS failed (non-fatal): %s", e)


def _stop_self() -> None:
    """Stop the current EC2 instance (no-op when EC2_SELF_STOP is false)."""
    if config.EC2_SELF_STOP and config.EC2_INSTANCE_ID:
        logger.info("[ec2] Stopping instance...")
        boto3.client("ec2", region_name=config.AWS_REGION).stop_instances(
            InstanceIds=[config.EC2_INSTANCE_ID]
        )
