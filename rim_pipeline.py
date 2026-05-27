#!/usr/bin/env python3
"""
Rim Scratch Annotation Pipeline
S3 (read-only) → YOLO rim scratch model → Roboflow (two projects)

Reads wheel scanner images from S3 under the ``wheel_scanner/`` prefix and
uploads every image to two parallel Roboflow projects:

  rim_scratch  — image + YOLO polygon annotations (config.RIM_SCRATCH_ROBOFLOW_PROJECT)
  rim_seg      — raw image, no annotations        (config.RIM_SEG_ROBOFLOW_PROJECT)

S3 key structure::

    <bucket>/wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/left_rear_wheel_org_image.jpg

Only ``*_org_image.jpg`` files are processed; overlay images are skipped.

Roboflow image name format::

    {dealership}_{scan_folder}_{wheel_position}.jpg

Example::

    castle_hill_toyota_SCANNER_A_001608_2026-05-06_09-06-18_left_rear_wheel.jpg

Usage examples::

    # Current week Monday–Friday, both projects (default)
    python rim_pipeline.py

    # Specific date range
    python rim_pipeline.py --from 2026-05-06 --to 2026-05-06

    # Single day
    python rim_pipeline.py --from 2026-05-06 --to 2026-05-06

    # Dry run — run YOLO but skip all Roboflow uploads
    python rim_pipeline.py --from 2026-05-06 --to 2026-05-06 --dry-run

    # Upload to rim_seg only (skip annotated scratch project)
    python rim_pipeline.py --skip-rim-scratch

    # Upload to rim_scratch only (skip raw image project)
    python rim_pipeline.py --skip-rim-seg

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
from roboflow_uploader import upload_rim_scratch, upload_rim_seg
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
    date_from:       date | None = None,
    date_to:         date | None = None,
    bucket:          str | None  = None,
    limit:           int | None  = None,
    dry_run:         bool        = False,
    run_rim_scratch: bool        = True,
    run_rim_seg:     bool        = True,
) -> None:
    """
    Run the rim scratch annotation pipeline for a given date range.

    Reads ``*_org_image.jpg`` files from the ``wheel_scanner/`` S3 prefix,
    runs YOLO rim scratch inference, then uploads every image to the enabled
    Roboflow projects:

    - **rim_scratch**: image + YOLO polygon annotations (or unannotated if no
      predictions). Disabled with ``--skip-rim-scratch``.
    - **rim_seg**: raw image, no annotations — all images regardless of
      detections. Disabled with ``--skip-rim-seg``.

    No vision verification step — YOLO predictions are uploaded directly.

    Args:
        date_from:       Inclusive start date (default: Monday this week).
        date_to:         Inclusive end date (default: Friday this week).
        bucket:          S3 bucket to read from (default from config).
        limit:           Maximum number of images to process (for testing).
        dry_run:         If True, skip all Roboflow uploads but run other steps.
        run_rim_scratch: If False, skip rim_scratch uploads entirely.
        run_rim_seg:     If False, skip rim_seg uploads entirely.
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
    logger.info(
        "  Rim-scratch project : %s  (%s)",
        config.RIM_SCRATCH_ROBOFLOW_PROJECT,
        "enabled" if run_rim_scratch else "SKIPPED",
    )
    logger.info(
        "  Rim-seg project     : %s  (%s)",
        config.RIM_SEG_ROBOFLOW_PROJECT,
        "enabled" if run_rim_seg else "SKIPPED",
    )
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

    rim_scratch_uploaded = 0
    rim_seg_uploaded     = 0
    errors               = 0

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
                "         [yolo-rim-scratch] %d pred(s)  %s",
                len(predictions), pred_summary,
            )

            record["yolo_count"]       = len(predictions)
            record["yolo_predictions"] = predictions
            record["rim_scratch_uploaded"] = False
            record["rim_seg_uploaded"]     = False

            # 3 ── Upload to rim_scratch project (image + YOLO annotations)
            if not run_rim_scratch:
                logger.info("         [rim-scratch] Skipped (--skip-rim-scratch)")
                record["rim_scratch_uploaded"] = "skipped"
            elif dry_run:
                logger.info(
                    "         [rim-scratch] DRY RUN — would upload with %d annotation(s)",
                    len(predictions),
                )
                record["rim_scratch_uploaded"] = "dry_run"
            else:
                ok = upload_rim_scratch(
                    pil_image   = pil,
                    s3_key      = key,
                    predictions = predictions,
                    img_w       = pil.width,
                    img_h       = pil.height,
                    batch_name  = batch_name,
                    dealership  = dealership,
                )
                record["rim_scratch_uploaded"] = ok
                if ok:
                    rim_scratch_uploaded += 1

            # 4 ── Upload to rim_seg project (raw image, no annotation)
            if not run_rim_seg:
                logger.info("         [rim-seg] Skipped (--skip-rim-seg)")
                record["rim_seg_uploaded"] = "skipped"
            elif dry_run:
                logger.info(
                    "         [rim-seg] DRY RUN — would upload to rim_seg project",
                )
                record["rim_seg_uploaded"] = "dry_run"
            else:
                seg_ok = upload_rim_seg(
                    pil_image  = pil,
                    s3_key     = key,
                    batch_name = batch_name,
                    dealership = dealership,
                )
                record["rim_seg_uploaded"] = seg_ok
                if seg_ok:
                    rim_seg_uploaded += 1

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
        f"Rim-scratch project: {config.RIM_SCRATCH_ROBOFLOW_PROJECT}\n"
        f"Rim-seg project    : {config.RIM_SEG_ROBOFLOW_PROJECT}\n"
        f"Duration           : {elapsed:.0f}s\n"
        f"Processed          : {len(pending)} images\n"
        f"Rim-scratch upload : {rim_scratch_uploaded} to batch '{batch_name}' "
        f"(dry_run={dry_run})\n"
        f"Rim-seg upload     : {rim_seg_uploaded} to batch '{batch_name}' "
        f"(dry_run={dry_run})\n"
        f"Errors             : {errors}\n"
        f"Local log          : {log_path}"
    )
    logger.info("\n%s\n%s\n%s\n", "=" * 62, summary, "=" * 62)

    upload_log_to_s3(str(log_path))

    _notify(
        subject=(
            f"[Rim Pipeline] {batch_name} — "
            f"{rim_scratch_uploaded} scratch / {rim_seg_uploaded} seg uploaded"
        ),
        message=summary,
    )
    _stop_self()


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = _parse_args()
    try:
        run(
            date_from       = args.date_from,
            date_to         = args.date_to,
            bucket          = args.bucket,
            limit           = args.limit,
            dry_run         = args.dry_run,
            run_rim_scratch = not args.skip_rim_scratch,
            run_rim_seg     = not args.skip_rim_seg,
        )
    except Exception as e:
        logger.error("[rim_pipeline] ❌ Unexpected error: %s", e)
        sys.exit(1)
