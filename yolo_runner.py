"""
yolo_runner.py — Local YOLO inference using the rim scratch segmentation model.
No network calls. Runs entirely on local GPU/CPU.
"""

import logging
from typing import Any

import cv2
import numpy as np
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


def _simplify_polygon(
    polygon_px: np.ndarray,
    img_w:      int,
    img_h:      int,
    epsilon:    float,
) -> list[list[float]]:
    """
    Apply Douglas-Peucker simplification to a pixel-space polygon contour and
    return normalised ``[x, y]`` pairs in the range ``[0, 1]``.

    Ultralytics extracts contours via ``cv2.findContours`` on a mask that was
    upsampled from the model's low-resolution output (e.g. 160×160 → full image
    size). This produces hundreds of densely-packed points that trace every
    pixel of the jagged upsampling artefacts. Douglas-Peucker (``approxPolyDP``)
    removes collinear and near-collinear points while guaranteeing that no
    retained point deviates more than ``epsilon`` pixels from the original
    contour, producing a smoother, tighter polygon.

    Args:
        polygon_px: Pixel-space contour array, shape ``(N, 2)``, float.
                    Typically ``results.masks.xy[i]``.
        img_w:      Original image width in pixels (used to normalise x).
        img_h:      Original image height in pixels (used to normalise y).
        epsilon:    Douglas-Peucker tolerance in pixels
                    (``config.POLYGON_SIMPLIFY_EPSILON``).

    Returns:
        List of ``[x_norm, y_norm]`` pairs, each rounded to 4 d.p.
        Returns an empty list if ``polygon_px`` is empty.
    """
    if len(polygon_px) == 0:
        return []

    if len(polygon_px) < 4:
        # Too few points to simplify meaningfully — just normalise as-is.
        return [
            [round(float(pt[0]) / img_w, 4), round(float(pt[1]) / img_h, 4)]
            for pt in polygon_px
        ]

    # approxPolyDP requires shape (N, 1, 2) with dtype int32.
    contour    = polygon_px.reshape(-1, 1, 2).astype(np.int32)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)  # shape (M, 1, 2)
    pts        = simplified.reshape(-1, 2).astype(float)

    return [
        [round(pt[0] / img_w, 4), round(pt[1] / img_h, 4)]
        for pt in pts
    ]


def run_rim_inference(pil_image) -> list[dict[str, Any]]:
    """
    Run YOLO rim scratch segmentation inference on a PIL image.

    Polygon extraction pipeline
    ---------------------------
    1. ``results.masks.xy[i]`` — pixel-space contour at the original image
       resolution (already upsampled by Ultralytics from the model's internal
       mask resolution).
    2. ``_simplify_polygon`` — Douglas-Peucker simplification via
       ``cv2.approxPolyDP`` removes upsampling artefacts and reduces point
       count while keeping the polygon within ``POLYGON_SIMPLIFY_EPSILON``
       pixels of the original contour.
    3. Normalised to ``[0, 1]`` for storage and COCO JSON output.

    Args:
        pil_image: A PIL Image object (RGB).

    Returns:
        List of prediction dicts::

            {
                "class":      str,                # e.g. "scratch"
                "confidence": float,              # 0.0–1.0
                "polygon":    list[list[float]],  # [[x, y], …] normalised 0-1
            }

        Returns ``[]`` when no detections are above the confidence threshold.
        Logs a warning and returns ``polygon: []`` per detection if the loaded
        model is an object-detection model (no segmentation masks).
    """
    model   = _get_rim_model()
    results = model(
        pil_image,
        conf=config.YOLO_CONF_THRESHOLD,
        iou=config.YOLO_IOU_THRESHOLD,
        verbose=False,
    )[0]

    has_masks = results.masks is not None and len(results.masks) > 0

    if not has_masks and len(results.boxes) > 0:
        logger.warning(
            "[yolo-rim] %d detection(s) found but model produced no polygon masks. "
            "Ensure the loaded model is a segmentation model, not object detection.",
            len(results.boxes),
        )

    img_w = pil_image.width
    img_h = pil_image.height

    predictions: list[dict[str, Any]] = []
    for i, box in enumerate(results.boxes):
        polygon: list[list[float]] = []

        if has_masks and i < len(results.masks):
            # masks.xy[i] → pixel-space contour, shape (N_pts, 2), dtype float.
            # Using xy (pixel coords) rather than xyn (normalised) so that the
            # Douglas-Peucker epsilon is expressed in interpretable pixel units.
            polygon = _simplify_polygon(
                results.masks.xy[i],
                img_w,
                img_h,
                config.POLYGON_SIMPLIFY_EPSILON,
            )
            logger.debug(
                "[yolo-rim]   mask %d: %d raw pts → %d after simplification",
                i,
                len(results.masks.xy[i]),
                len(polygon),
            )

        predictions.append({
            "class":      results.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "polygon":    polygon,
        })

    return predictions
