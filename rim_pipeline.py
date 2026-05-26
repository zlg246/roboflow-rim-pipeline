#!/usr/bin/env python3
"""
Rim Scratch Annotation Pipeline
S3 (read-only) → YOLO rim scratch model → Roboflow

Reads wheel scanner images from S3 under the ``wheel_scanner/`` prefix,
runs a dedicated rim scratch segmentation model, and uploads every image
to the configured Roboflow project — with predicted annotations when
scratches are detected, or as a plain image when none are found.

S3 key structure::

    <bucket>/wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/left_rear_wheel_org_image.jpg

Only ``*_org_image.jpg`` files are processed; overlay images are skipped.

Roboflow image name format::

    {dealership}_{scan_folder}_{wheel_position}.jpg

Example::

    castle_hill_toyota_SCANNER_A_001608_2026-05-06_09-06-18_left_rear_wheel.jpg

Usage examples::

    # Current week Monday–Friday (default)
    python rim_pipeline.py

    # Specific date range
    python rim_pipeline.py --from 2026-05-06 --to 2026-05-06

    # Single day
    python rim_pipeline.py --from 2026-05-06 --to 2026-05-06

    # Dry run — run YOLO but skip Roboflow upload
    python rim_pipeline.py --from 2026-05-06 --to 2026-05-06 --dry-run

    # Limit images (useful for quick tests)
    python rim_pipeline.py --from 2026-05-06 --to 2026-05-06 --limit 5

    # Override bucket
    python rim_pipeline.py --bucket prod-chatswood-toyota
"""

import logging
import sys
import time
from datetime import date, datetime, timezone

from botocore.exceptions import ClientError

import config
from pipeline_types import PipelineRecord
from roboflow_uploader import upload_rim
from s3_loader import list_rim_image_keys, load_image, upload_log_to_s3
from utils import (
    LOG_PATH,
    _extract_dealership,
    _load_processed,
    _log,
    _make_log_path,
    _notify,
    _parse_args,
    _stop_self,
)
from yolo_runner import run_rim_inference

logger = logging.getLogger(__name__)


def run(
    date_from: date | None = None,
    date_to:   date | None = None,
    bucket:    str | None  = None,
    limit:     int | None  = None,
    dry_run:   bool        = False,
) -> None:
    """
    Run the rim scratch annotation pipeline for a given date range.

    Reads ``*_org_image.jpg`` files from the ``wheel_scanner/`` S3 prefix,
    runs YOLO rim scratch inference, then uploads every image to Roboflow
    (with annotations if predictions were found, without if not).

    No vision verification step — YOLO predictions are uploaded directly.

    Args:
        date_from: Inclusive start date (default: Monday this week).
        date_to:   Inclusive end date (default: Friday this week).
        bucket:    S3 bucket to read from (default from config).
        limit:     Maximum number of images to process (for testing).
        dry_run:   If True, skip Roboflow upload but run all other steps.
    """
    start     = datetime.now(timezone.utc)
    date_from = date_from or config.default_date_from()
    date_to   = date_to   or config.default_date_to()
    bucket    = bucket    or config.S3_BUCKET

    dealership    = _extract_dealership(bucket)
    date_from_str = date_from.strftime("%Y-%m-%d")
    date_to_str   = date_to.strftime("%Y-%m-%d")
    batch_name    = f"Agent_{dealership}_{date_from_str}_{date_to_str}"

    _make_log_path()
    log_path = LOG_PATH

    logger.info("\n%s", "=" * 62)
    logger.info("  Rim Pipeline start  : %sZ", start.isoformat())
    logger.info("  Production bucket   : %s  (READ ONLY)", bucket)
    logger.info("  Date range          : %s → %s", date_from, date_to)
    logger.info("  S3 prefix           : %s", config.RIM_S3_ROOT_PREFIX)
    logger.info("  YOLO model          : %s", config.RIM_YOLO_MODEL_PATH)
    logger.info("  Roboflow project    : %s", config.RIM_ROBOFLOW_PROJECT)
    logger.info("  Roboflow batch      : %s", batch_name)
    logger.info("  Dry run             : %s", dry_run)
    logger.info("  Log file            : %s", log_path)
    logger.info("%s\n", "=" * 62)

    # ── Fetch image list (read-only) ──────────────────────────────────
    try:
        keys = list_rim_image_keys(
            date_from=date_from,
            date_to=date_to,
            bucket=bucket,
        )
    except RuntimeError as e:
        logger.error("[rim_pipeline] ❌ Credential error for bucket '%s': %s", bucket, e)
        logger.error("[rim_pipeline]    Skipping this dealership.")
        return
    except ClientError as e:
        logger.error(
            "[rim_pipeline] ❌ S3 access error for bucket '%s': %s", bucket,
            e.response["Error"].get("Message", e),
        )
        logger.error("[rim_pipeline]    Skipping this dealership.")
        return

    processed = _load_processed(log_path)
    pending   = [k for k in keys if k not in processed]

    if limit:
        pending = pending[:limit]

    logger.info("[rim_pipeline] In range    : %d", len(keys))
    logger.info("[rim_pipeline] Already done: %d", len(processed))
    logger.info("[rim_pipeline] To process  : %d\n", len(pending))

    if not pending:
        logger.info("[rim_pipeline] Nothing new to process. Exiting.")
        return

    uploaded = 0
    errors   = 0

    for i, key in enumerate(pending, 1):
        logger.info("[%4d/%d]  %s", i, len(pending), key)
        record: PipelineRecord = {
            "s3_key":    key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch":     batch_name,
            "dry_run":   dry_run,
        }

        try:
            # 1 ── Load image from S3 (read-only GetObject)
            pil, _b64, _mime = load_image(key, bucket=bucket)

            # 2 ── Local YOLO rim scratch inference
            predictions  = run_rim_inference(pil)
            pred_summary = (
                "  ".join(f"{p['class']}@{p['confidence']:.2f}" for p in predictions)
                or "none"
            )
            logger.info(
                "         [yolo-rim] %d pred(s)  %s",
                len(predictions), pred_summary,
            )

            record["yolo_count"]       = len(predictions)
            record["yolo_predictions"] = predictions
            record["uploaded"]         = False

            # 3 ── Upload to Roboflow (always — with or without predictions)
            if dry_run:
                logger.info(
                    "         [upload] DRY RUN — would upload with %d annotation(s)",
                    len(predictions),
                )
                record["uploaded"] = "dry_run"
            else:
                ok = upload_rim(
                    pil_image   = pil,
                    s3_key      = key,
                    predictions = predictions,
                    img_w       = pil.width,
                    img_h       = pil.height,
                    batch_name  = batch_name,
                    dealership  = dealership,
                )
                record["uploaded"] = ok
                if ok:
                    uploaded += 1

        except Exception as e:
            logger.error("         [error]  %s", e)
            record["error"] = str(e)
            errors += 1

        _log(log_path, record)
        time.sleep(config.UPLOAD_SLEEP)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    summary = (
        f"Rim Pipeline complete\n"
        f"Production bucket  : {bucket}  (read-only, no writes)\n"
        f"Date range         : {date_from} → {date_to}\n"
        f"S3 prefix          : {config.RIM_S3_ROOT_PREFIX}\n"
        f"Roboflow project   : {config.RIM_ROBOFLOW_PROJECT}\n"
        f"Duration           : {elapsed:.0f}s\n"
        f"Processed          : {len(pending)} images\n"
        f"Uploaded           : {uploaded} to Roboflow batch '{batch_name}' "
        f"(dry_run={dry_run})\n"
        f"Errors             : {errors}\n"
        f"Local log          : {log_path}"
    )
    logger.info("\n%s\n%s\n%s\n", "=" * 62, summary, "=" * 62)

    upload_log_to_s3(str(log_path))

    _notify(
        subject=f"[Rim Pipeline] {batch_name} — {uploaded} uploaded",
        message=summary,
    )
    _stop_self()


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = _parse_args()
    try:
        run(
            date_from = args.date_from,
            date_to   = args.date_to,
            bucket    = args.bucket,
            limit     = args.limit,
            dry_run   = args.dry_run,
        )
    except Exception as e:
        logger.error("[rim_pipeline] ❌ Unexpected error: %s", e)
        sys.exit(1)
