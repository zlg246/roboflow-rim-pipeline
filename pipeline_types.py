"""Shared TypedDict definitions used across the pipeline."""

from typing import Any, TypedDict


class PipelineRecord(TypedDict, total=False):
    """One JSONL log entry written per processed image."""

    s3_key:            str
    timestamp:         str
    batch:             str
    dry_run:           bool
    yolo_count:        int
    yolo_predictions:  list[Any]   # list of {x, y, width, height, class, confidence}
    rim_scratch_uploaded: bool | str  # True | False | "dry_run"  (rim_scratch)
    rim_seg_uploaded:  bool | str  # True | False | "dry_run"  (rim_seg)
    error:             str
