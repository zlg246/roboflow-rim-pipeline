"""Shared TypedDict definitions used across the pipeline."""

from typing import Any, TypedDict


# Functional form required because "class" is a Python reserved keyword.
Prediction = TypedDict(
    "Prediction",
    {
        "x":          float,
        "y":          float,
        "width":      float,
        "height":     float,
        "class":      str,
        "confidence": float,
    },
)

Correction = TypedDict(
    "Correction",
    {
        "class":  str,
        "x":      float,
        "y":      float,
        "width":  float,
        "height": float,
        "note":   str,     # optional — omitted when not relevant
    },
    total=False,
)


class VisionResult(TypedDict):
    """Structured response returned by verify_image()."""

    decision:    str              # "scratch" | "other" | "null"
    confidence:  float            # 0.0–1.0
    reasoning:   str
    corrections: list[Any]        # list[Correction]


class PipelineRecord(TypedDict, total=False):
    """One JSONL log entry written per processed image."""

    s3_key:           str
    timestamp:        str
    batch:            str
    dry_run:          bool
    decision:         str
    vision_provider:  str
    vision_confidence: float
    reasoning:        str
    yolo_count:       int
    yolo_predictions: list[Any]   # list[Prediction]
    corrections:      list[Any]   # list[Correction]
    uploaded:         bool | str  # True | False | "dry_run"
    error:            str
