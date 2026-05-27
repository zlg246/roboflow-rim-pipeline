"""
roboflow_uploader.py — Upload rim wheel images to Roboflow.

Two parallel projects:
  rim_scratch  — images + YOLO polygon annotations  (config.RIM_SCRATCH_ROBOFLOW_PROJECT)
  rim_seg      — raw images, no annotations          (config.RIM_SEG_ROBOFLOW_PROJECT)
"""

import json
import logging
import os
import tempfile
from io import BytesIO
from typing import Any

import config

logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────

def _derive_image_name(s3_key: str, dealership: str) -> str:
    """Derive the Roboflow upload filename from an S3 key and dealership name.

    Input:  ``wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/left_rear_wheel_org_image.jpg``
    Output: ``castle_hill_toyota_SCANNER_A_001608_2026-05-06_09-06-18_left_rear_wheel.jpg``
    """
    parts       = s3_key.split("/")
    scan_folder = parts[-2] if len(parts) >= 3 else "unknown"
    wheel_pos   = parts[-1].replace("_org_image.jpg", "")
    return f"{dealership}_{scan_folder}_{wheel_pos}.jpg"


def _pil_to_bytes(pil_image) -> bytes:
    """Encode a PIL image to JPEG bytes at the configured upload quality."""
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=config.UPLOAD_JPEG_QUALITY)
    return buf.getvalue()


# ── rim_scratch project ───────────────────────────────────────────────

_rim_scratch_rf_project = None


def _get_rim_scratch_project():
    """Lazily connect to the rim_scratch Roboflow project and cache the handle."""
    global _rim_scratch_rf_project
    if _rim_scratch_rf_project is None:
        from roboflow import Roboflow
        rf                    = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        _rim_scratch_rf_project = (
            rf.workspace(config.ROBOFLOW_WORKSPACE)
            .project(config.RIM_SCRATCH_ROBOFLOW_PROJECT)
        )
        logger.info("[roboflow-rim-scratch] Connected: %s", _rim_scratch_rf_project.name)
    return _rim_scratch_rf_project


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


def upload_rim_scratch(
    pil_image:   Any,
    s3_key:      str,
    predictions: list[dict[str, Any]],
    img_w:       int,
    img_h:       int,
    batch_name:  str,
    dealership:  str,
) -> bool:
    """
    Upload one rim wheel image + polygon predictions to the rim_scratch Roboflow project.

    Always uploads — with a COCO JSON annotation file when predictions are
    present, or as an unannotated image when there are none.

    Filename format::

        {dealership}_{scan_folder}_{wheel_position}.jpg

    Example::

        castle_hill_toyota_SCANNER_A_001608_2026-05-06_09-06-18_left_rear_wheel.jpg

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
    image_name = _derive_image_name(s3_key, dealership)

    tmp_dir = img_tmp = ann_tmp = None

    try:
        project = _get_rim_scratch_project()

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
                "    [roboflow-rim-scratch] ✅ %d polygon(s) uploaded: %s",
                len(predictions), image_name,
            )
        else:
            logger.info(
                "    [roboflow-rim-scratch] 📭 No annotation uploaded: %s", image_name,
            )
        return True

    except Exception as e:
        logger.error("    [roboflow-rim-scratch] ❌ Upload failed: %s", e)
        return False

    finally:
        for path in (img_tmp, ann_tmp):
            if path and os.path.exists(path):
                os.unlink(path)
        if tmp_dir and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)


# ── rim_seg project ───────────────────────────────────────────────────

_rim_seg_rf_project = None


def _get_rim_seg_project():
    """Lazily connect to the rim_seg Roboflow project and cache the handle."""
    global _rim_seg_rf_project
    if _rim_seg_rf_project is None:
        from roboflow import Roboflow
        rf               = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        _rim_seg_rf_project = (
            rf.workspace(config.ROBOFLOW_WORKSPACE)
            .project(config.RIM_SEG_ROBOFLOW_PROJECT)
        )
        logger.info("[roboflow-rim-seg] Connected: %s", _rim_seg_rf_project.name)
    return _rim_seg_rf_project


def upload_rim_seg(
    pil_image:  Any,
    s3_key:     str,
    batch_name: str,
    dealership: str,
) -> bool:
    """
    Upload a raw image (no annotations) to the rim_seg Roboflow project.

    All S3 images are uploaded regardless of scratch detections, providing
    an unannotated image pool in ``config.RIM_SEG_ROBOFLOW_PROJECT``.

    Args:
        pil_image:  PIL Image to upload.
        s3_key:     Original S3 key — used to derive the upload filename.
        batch_name: Roboflow batch name (shared with the rim_scratch upload).
        dealership: Dealership identifier derived from the bucket name.

    Returns:
        True on success, False on failure.
    """
    image_name = _derive_image_name(s3_key, dealership)

    tmp_dir = img_tmp = None

    try:
        project = _get_rim_seg_project()

        tmp_dir = tempfile.mkdtemp()
        img_tmp = os.path.join(tmp_dir, image_name)
        with open(img_tmp, "wb") as f:
            f.write(_pil_to_bytes(pil_image))

        project.upload(
            image_path        = img_tmp,
            annotation_path   = None,
            batch_name        = batch_name,
            tag_names         = [],
            is_prediction     = False,
            num_retry_uploads = 3,
        )

        logger.info("    [roboflow-rim-seg] ✅ Uploaded: %s", image_name)
        return True

    except Exception as e:
        logger.error("    [roboflow-rim-seg] ❌ Upload failed: %s", e)
        return False

    finally:
        if img_tmp and os.path.exists(img_tmp):
            os.unlink(img_tmp)
        if tmp_dir and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
