"""
yolo_runner.py — Local YOLO inference using the rim scratch segmentation model.
No network calls. Runs entirely on local GPU/CPU.
"""

import logging
from typing import Any

from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

_rim_model: YOLO | None = None


def _get_rim_model() -> YOLO:
    """Lazily load and cache the rim scratch YOLO model."""
    global _rim_model
    if _rim_model is None:
        logger.info("[yolo-rim] Loading model: %s", config.RIM_YOLO_MODEL_PATH)
        _rim_model = YOLO(config.RIM_YOLO_MODEL_PATH)
    return _rim_model


def run_rim_inference(pil_image) -> list[dict[str, Any]]:
    """
    Run YOLO rim scratch segmentation inference on a PIL image.

    Reads polygon masks from ``results.masks.xyn`` (normalised coordinates).
    Each returned prediction contains a ``polygon`` key — a list of ``[x, y]``
    pairs in the range ``[0, 1]``.

    Args:
        pil_image: A PIL Image object (RGB).

    Returns:
        List of prediction dicts::

            {
                "class":      str,          # e.g. "scratch"
                "confidence": float,        # 0.0–1.0
                "polygon":    list[list[float]],  # [[x, y], …] normalised
            }

        Returns [] when no detections are above the confidence threshold.
        If the loaded model is an object-detection model (no masks), a warning
        is logged and ``polygon`` will be an empty list for every detection.
    """
    model   = _get_rim_model()
    results = model(pil_image, conf=config.YOLO_CONF_THRESHOLD, iou=config.YOLO_IOU_THRESHOLD, verbose=False)[0]

    has_masks = results.masks is not None and len(results.masks) > 0

    if not has_masks and len(results.boxes) > 0:
        logger.warning(
            "[yolo-rim] %d detection(s) found but model produced no polygon masks. "
            "Ensure the loaded model is a segmentation model, not object detection.",
            len(results.boxes),
        )

    predictions: list[dict[str, Any]] = []
    for i, box in enumerate(results.boxes):
        # Normalised polygon points from the segmentation mask, shape (N_pts, 2).
        # Falls back to an empty list if masks are unavailable.
        polygon: list[list[float]] = (
            results.masks.xyn[i].tolist() if has_masks and i < len(results.masks) else []
        )

        predictions.append({
            "class":      results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "polygon":    polygon,
        })

    return predictions
