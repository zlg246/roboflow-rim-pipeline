"""
roboflow_uploader.py — Upload rim wheel images and polygon annotations to Roboflow.
"""

import json
import logging
import os
import tempfile
from io import BytesIO
from typing import Any

import config

logger = logging.getLogger(__name__)

_rim_rf_project = None


def _get_rim_project():
    """Lazily connect to the rim scratch Roboflow project and cache the handle."""
    global _rim_rf_project
    if _rim_rf_project is None:
        from roboflow import Roboflow
        rf              = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        _rim_rf_project = (
            rf.workspace(config.ROBOFLOW_WORKSPACE)
            .project(config.RIM_ROBOFLOW_PROJECT)
        )
        logger.info("[roboflow-rim] Connected: %s", _rim_rf_project.name)
    return _rim_rf_project


def _pil_to_bytes(pil_image) -> bytes:
    """Encode a PIL image to JPEG bytes at the configured upload quality."""
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=config.UPLOAD_JPEG_QUALITY)
    return buf.getvalue()


def _build_coco_json(
    filename:    str,
    img_w:       int,
    img_h:       int,
    predictions: list[dict[str, Any]],
) -> str:
    """
    Build a COCO JSON annotation string for polygon segmentation predictions.

    Each prediction must contain a ``polygon`` key — a list of ``[x_norm, y_norm]``
    pairs (normalised 0–1). The function denormalises them to pixel coordinates
    and computes the bounding box from the polygon extents.

    COCO segmentation format uses a flat list of pixel coordinates::

        "segmentation": [[x1, y1, x2, y2, x3, y3, ...]]

    Args:
        filename:    Image filename embedded in the JSON.
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.
        predictions: List of dicts with keys: class, confidence, polygon.

    Returns:
        COCO JSON string.
    """
    # Build category map: class name → integer ID (1-indexed)
    class_names = sorted(set(p["class"] for p in predictions))
    cat_id_map  = {name: i + 1 for i, name in enumerate(class_names)}

    categories = [
        {"id": cat_id, "name": name, "supercategory": ""}
        for name, cat_id in cat_id_map.items()
    ]

    annotations = []
    for ann_id, pred in enumerate(predictions, start=1):
        polygon_norm = pred["polygon"]  # [[x_norm, y_norm], ...]

        # Denormalise to pixel coordinates
        px_coords = [(pt[0] * img_w, pt[1] * img_h) for pt in polygon_norm]

        # COCO segmentation: flat list [x1, y1, x2, y2, ...]
        flat = [coord for pt in px_coords for coord in pt]

        # Bounding box derived from polygon extents
        xs     = [pt[0] for pt in px_coords]
        ys     = [pt[1] for pt in px_coords]
        x_min  = min(xs);  x_max = max(xs)
        y_min  = min(ys);  y_max = max(ys)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        area   = bbox_w * bbox_h

        annotations.append({
            "id":           ann_id,
            "image_id":     1,
            "category_id":  cat_id_map[pred["class"]],
            "segmentation": [flat],                                       # list of polygons
            "area":         round(area, 2),
            "bbox":         [round(x_min, 2), round(y_min, 2),
                             round(bbox_w, 2), round(bbox_h, 2)],         # [x, y, w, h]
            "iscrowd":      0,
        })

    coco = {
        "images":      [{"id": 1, "file_name": filename,
                          "width": img_w, "height": img_h}],
        "annotations": annotations,
        "categories":  categories,
    }
    return json.dumps(coco, indent=2)


def upload_rim(
    pil_image:   Any,
    s3_key:      str,
    predictions: list[dict[str, Any]],
    img_w:       int,
    img_h:       int,
    batch_name:  str,
    dealership:  str,
) -> bool:
    """
    Upload one rim wheel image + polygon predictions to Roboflow.

    Always uploads — with a COCO JSON annotation file when predictions are
    present, or as an unannotated image when there are none.

    Filename format::

        {dealership}_{scan_folder}_{wheel_position}.jpg

    Example::

        castle_hill_toyota_SCANNER_A_001608_2026-05-06_09-06-18_left_rear_wheel.jpg

    The ``wheel_position`` is derived from the S3 filename by stripping the
    ``_org_image.jpg`` suffix (e.g. ``left_rear_wheel_org_image.jpg`` →
    ``left_rear_wheel``).

    Args:
        pil_image:   PIL Image to upload.
        s3_key:      Original S3 key — used to derive the upload filename.
                     Expected structure:
                     ``wheel_scanner/<scan_folder>/<wheel_pos>_org_image.jpg``
        predictions: List of prediction dicts (class, confidence, polygon).
                     Pass an empty list when no scratches were detected.
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.
        batch_name:  Roboflow batch name.
        dealership:  Dealership identifier derived from the bucket name.

    Returns:
        True on success, False on failure.
    """
    # Derive upload filename from S3 key
    # e.g. wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/left_rear_wheel_org_image.jpg
    parts       = s3_key.split("/")
    scan_folder = parts[-2] if len(parts) >= 3 else "unknown"
    orig_name   = parts[-1]                                    # left_rear_wheel_org_image.jpg
    wheel_pos   = orig_name.replace("_org_image.jpg", "")     # left_rear_wheel
    image_name  = f"{dealership}_{scan_folder}_{wheel_pos}.jpg"

    tmp_dir = img_tmp = ann_tmp = None

    try:
        project = _get_rim_project()

        tmp_dir = tempfile.mkdtemp()
        img_tmp = os.path.join(tmp_dir, image_name)
        with open(img_tmp, "wb") as f:
            f.write(_pil_to_bytes(pil_image))

        if predictions:
            coco_str = _build_coco_json(image_name, img_w, img_h, predictions)
            ann_tmp  = os.path.join(tmp_dir, image_name.rsplit(".", 1)[0] + ".json")
            with open(ann_tmp, "w", encoding="utf-8") as f:
                f.write(coco_str)

        project.upload(
            image_path        = img_tmp,
            annotation_path   = ann_tmp,
            batch_name        = batch_name,
            tag_names         = [],
            is_prediction     = True,
            num_retry_uploads = 3,
        )

        if predictions:
            logger.info(
                "    [roboflow-rim] ✅ %d polygon(s) uploaded: %s",
                len(predictions), image_name,
            )
        else:
            logger.info(
                "    [roboflow-rim] 📭 No annotation uploaded: %s", image_name,
            )
        return True

    except Exception as e:
        logger.error("    [roboflow-rim] ❌ Upload failed: %s", e)
        return False

    finally:
        for path in (img_tmp, ann_tmp):
            if path and os.path.exists(path):
                os.unlink(path)
        if tmp_dir and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
