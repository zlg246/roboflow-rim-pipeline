"""
yolo_runner.py — Local YOLO inference using your trained model.
No network calls. Runs entirely on local GPU/CPU.
"""

import logging
from typing import Any

from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)

_model: YOLO | None = None


def _get_model() -> YOLO:
    """Lazily load and cache the YOLO model (avoids reloading on each call)."""
    global _model
    if _model is None:
        logger.info("[yolo] Loading model: %s", config.YOLO_MODEL_PATH)
        _model = YOLO(config.YOLO_MODEL_PATH)
    return _model


def run_inference(pil_image) -> list[dict[str, Any]]:
    """
    Run YOLO inference on a PIL image.

    Args:
        pil_image: A PIL Image object (RGB).

    Returns:
        List of prediction dicts with keys: x, y, width, height, class, confidence.
        All coordinates are normalised (0–1) in centre-based YOLO xywh format.
        Returns [] if no detections are above the confidence threshold.
    """
    model   = _get_model()
    results = model(pil_image, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]

    predictions: list[dict[str, Any]] = []
    for box in results.boxes:
        x, y, w, h = box.xywhn[0].tolist()
        predictions.append({
            "x":          round(x, 4),
            "y":          round(y, 4),
            "width":      round(w, 4),
            "height":     round(h, 4),
            "class":      results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
        })
    return predictions
