#!/usr/bin/env python3
"""
Vehicle Panel Damage Annotation Pipeline
S3 (read-only) → YOLO (local) → Claude/OpenAI Vision → Roboflow

Usage examples:
  # Current week Monday–Friday (default)
  python pipeline.py

  # Specific date range
  python pipeline.py --from 2026-03-16 --to 2026-03-20

  # Single day
  python pipeline.py --from 2026-03-16 --to 2026-03-16

  # Named shortcuts
  python pipeline.py --from last-friday --to last-friday
  python pipeline.py --from monday --to friday

  # Dry run — run YOLO + vision model but skip Roboflow upload
  python pipeline.py --from 2026-03-16 --to 2026-03-16 --dry-run

  # Limit images (useful for quick tests)
  python pipeline.py --from 2026-03-16 --to 2026-03-16 --limit 5

  # Override camera subfolders
  python pipeline.py --from 2026-03-16 --to 2026-03-16 --subfolders 3,7
"""

import json
import logging
import time
from datetime import date, datetime, timezone

import config
from pipeline_types import PipelineRecord
from roboflow_uploader import upload
from s3_loader import list_image_keys, load_image, upload_log_to_s3
from utils import (
    LOG_PATH,
    _load_processed,
    _log,
    _make_log_path,
    _notify,
    _parse_args,
    _save_null_review,
    _stop_self,
)
from vision_verifier import verify_image
from yolo_runner import run_inference

logger = logging.getLogger(__name__)


def run(
    date_from:         date | None     = None,
    date_to:           date | None     = None,
    target_subfolders: set[str] | None = None,
    bucket:            str | None      = None,
    limit:             int | None      = None,
    dry_run:           bool            = False,
) -> None:
    """
    Run the full annotation pipeline for a given date range.

    Reads images from S3 (read-only), runs YOLO inference locally,
    verifies detections with the configured vision model, and uploads
    images with confirmed damage to Roboflow.

    Args:
        date_from:          Inclusive start date (default: Monday this week).
        date_to:            Inclusive end date (default: Friday this week).
        target_subfolders:  Camera subfolder names to include (default from config).
        bucket:             S3 bucket to read from (default from config).
        limit:              Maximum number of images to process (for testing).
        dry_run:            If True, skip Roboflow upload but run all other steps.
    """
    start      = datetime.now(timezone.utc)
    date_from  = date_from  or config.default_date_from()
    date_to    = date_to    or config.default_date_to()
    subfolders = target_subfolders or config.S3_TARGET_SUBFOLDERS
    bucket     = bucket or config.S3_BUCKET

    # Batch name: <dealership>_<date_from>_<date_to>
    # e.g. castle_hill_toyota_2026-05-12_2026-05-16
    dealership    = bucket.removeprefix("prod-").replace("-", "_")
    date_from_str = date_from.strftime("%Y-%m-%d")
    date_to_str   = date_to.strftime("%Y-%m-%d")
    batch_name    = f"{dealership}_{date_from_str}_{date_to_str}"

    _make_log_path()
    log_path = LOG_PATH

    logger.info("\n%s", "=" * 62)
    logger.info("  Pipeline start   : %sZ", start.isoformat())
    logger.info("  Production bucket: %s  (READ ONLY)", bucket)
    logger.info("  Date range       : %s → %s", date_from, date_to)
    logger.info("  Camera subfolders: %s", sorted(subfolders))
    logger.info("  Roboflow batch   : %s", batch_name)
    logger.info("  Vision provider  : %s", config.VISION_PROVIDER)
    logger.info("  Dry run          : %s", dry_run)
    logger.info("  Log file         : %s", log_path)
    logger.info("%s\n", "=" * 62)

    # ── Fetch image list (read-only) ──────────────────────────────────
    keys      = list_image_keys(
        date_from=date_from,
        date_to=date_to,
        bucket=bucket,
        target_subfolders=subfolders,
    )
    processed = _load_processed(log_path)
    pending   = [k for k in keys if k not in processed]

    if limit:
        pending = pending[:limit]

    logger.info("[pipeline] In range   : %d", len(keys))
    logger.info("[pipeline] Already done: %d", len(processed))
    logger.info("[pipeline] To process : %d\n", len(pending))

    if not pending:
        logger.info("[pipeline] Nothing new to process. Exiting.")
        return

    stats: dict[str, int] = {
        d: 0 for d in ("correct", "partial", "missed", "other", "null", "bad_quality", "error")
    }
    uploaded:      int = 0
    null_reviewed: int = 0

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
            pil, b64, mime = load_image(key, bucket=bucket)

            # 2 ── Local YOLO inference
            predictions = run_inference(pil)
            pred_summary = ("  ".join(
                f"{p['class']}@{p['confidence']:.2f}" for p in predictions
            ) or "none")
            logger.info("         [yolo]   %d pred(s)  %s", len(predictions), pred_summary)

            # 3 ── Vision verification (only when YOLO found something)
            if not predictions:
                decision    = "null"
                corrections = []
                result = {
                    "decision":    decision,
                    "confidence":  1.0,
                    "reasoning":   "Skipped — no YOLO predictions.",
                    "corrections": [],
                }
                logger.info("         [%s] skipped — no YOLO predictions", config.VISION_PROVIDER)
            else:
                result      = verify_image(b64, mime, predictions, key)
                decision    = result["decision"]
                corrections = result.get("corrections", [])
                logger.info(
                    "         [%s] %-12s  conf=%.2f  %s",
                    config.VISION_PROVIDER, decision,
                    result["confidence"], result["reasoning"],
                )

            record["decision"]          = decision
            record["vision_provider"]   = config.VISION_PROVIDER
            record["vision_confidence"] = result["confidence"]
            record["reasoning"]         = result["reasoning"]
            record["yolo_count"]        = len(predictions)
            record["yolo_predictions"]  = predictions
            record["corrections"]       = corrections
            record["uploaded"]          = False

            stats[decision] = stats.get(decision, 0) + 1

            # 4 ── Upload to Roboflow (damage-related decisions only)
            if decision in config.UPLOAD_DECISIONS:
                if dry_run:
                    logger.info("         [upload] DRY RUN — would upload (%s)", decision)
                    record["uploaded"] = "dry_run"
                else:
                    ok = upload(
                        pil_image   = pil,
                        s3_key      = key,
                        corrections = corrections,
                        img_w       = pil.width,
                        img_h       = pil.height,
                        batch_name  = batch_name,
                        dealership  = dealership,
                    )
                    record["uploaded"] = ok
                    if ok:
                        uploaded += 1
            else:
                logger.info("         [upload] Skipped (%s)", decision)
                if config.SAVE_NULL_REVIEW and decision == "null" and predictions:
                    _save_null_review(bucket, pil, key, predictions)
                    null_reviewed += 1

        except json.JSONDecodeError as e:
            logger.error("         [error]  Bad JSON from vision model: %s", e)
            record["error"] = f"JSONDecodeError: {e}"
            stats["error"] += 1

        except Exception as e:
            logger.error("         [error]  %s", e)
            record["error"] = str(e)
            stats["error"] += 1

        _log(log_path, record)
        time.sleep(config.OPENAI_RATE_LIMIT_SLEEP)

    # ── Summary ───────────────────────────────────────────────────────
    elapsed  = (datetime.now(timezone.utc) - start).total_seconds()
    summary  = (
        f"Pipeline complete\n"
        f"Production bucket : {config.S3_BUCKET}  (read-only, no writes)\n"
        f"Date range        : {date_from} → {date_to}\n"
        f"Vision provider   : {config.VISION_PROVIDER}\n"
        f"Duration          : {elapsed:.0f}s\n"
        f"Processed         : {len(pending)} images\n"
        f"Uploaded          : {uploaded} to Roboflow batch '{batch_name}' "
        f"(dry_run={dry_run})\n"
        f"Null review saved : {null_reviewed} images → null_review/\n"
        f"Decisions         :\n"
        + "\n".join(f"  {k:<12}: {v}" for k, v in stats.items())
        + f"\nLocal log         : {log_path}"
    )
    logger.info("\n%s\n%s\n%s\n", "=" * 62, summary, "=" * 62)

    upload_log_to_s3(str(log_path))

    _notify(
        subject=f"[Panel Pipeline] {batch_name} — {uploaded} uploaded",
        message=summary,
    )
    _stop_self()


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = _parse_args()
    subfolders = (
        set(s.strip() for s in args.subfolders.split(",") if s.strip())
        if args.subfolders
        else None
    )
    run(
        date_from         = args.date_from,
        date_to           = args.date_to,
        target_subfolders = subfolders,
        bucket            = args.bucket,
        limit             = args.limit,
        dry_run           = args.dry_run,
    )
