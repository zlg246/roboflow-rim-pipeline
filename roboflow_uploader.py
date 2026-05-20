"""
roboflow_uploader.py — Upload images and annotations to Roboflow.
"""

import logging
import os
import tempfile
from io import BytesIO
from typing import Any

import config

logger = logging.getLogger(__name__)

_rf_project = None


def _get_project():
    """Lazily connect to the Roboflow project and cache the handle."""
    global _rf_project
    if _rf_project is None:
        from roboflow import Roboflow
        rf          = Roboflow(api_key=config.ROBOFLOW_API_KEY)
        _rf_project = rf.workspace(config.ROBOFLOW_WORKSPACE).project(config.ROBOFLOW_PROJECT)
        logger.info("[roboflow] Connected: %s", _rf_project.name)
    return _rf_project


def _pil_to_bytes(pil_image) -> bytes:
    """Encode a PIL image to JPEG bytes at the configured upload quality."""
    buf = BytesIO()
    pil_image.save(buf, format="JPEG", quality=config.UPLOAD_JPEG_QUALITY)
    return buf.getvalue()


def _build_voc_xml(filename: str, img_w: int, img_h: int,
                   annotations: list[dict[str, Any]]) -> str:
    """
    Build a Pascal VOC XML annotation string.

    Converts normalised (0–1) centre-x/y/w/h coordinates to pixel
    xmin/ymin/xmax/ymax required by the VOC format.

    Args:
        filename:    Image filename embedded in the XML.
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.
        annotations: List of correction dicts with keys: class, x, y, width, height.

    Returns:
        VOC XML string.
    """
    objects_xml = ""
    for ann in annotations:
        cx   = ann["x"]      * img_w
        cy   = ann["y"]      * img_h
        w    = ann["width"]  * img_w
        h    = ann["height"] * img_h
        xmin = max(0,     int(cx - w / 2))
        ymin = max(0,     int(cy - h / 2))
        xmax = min(img_w, int(cx + w / 2))
        ymax = min(img_h, int(cy + h / 2))
        objects_xml += f"""
  <object>
    <name>{ann['class']}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>"""

    return f"""<annotation>
  <filename>{filename}</filename>
  <size>
    <width>{img_w}</width>
    <height>{img_h}</height>
    <depth>3</depth>
  </size>{objects_xml}
</annotation>"""


def upload(
    pil_image:   Any,
    s3_key:      str,
    corrections: list[dict[str, Any]],
    img_w:       int,
    img_h:       int,
    batch_name:  str,
    dealership:  str,
) -> bool:
    """
    Upload one image + annotations to Roboflow.

    The image is saved to a temporary directory under the desired filename so that
    Roboflow names the image correctly (it uses the filename from image_path).

    Args:
        pil_image:   PIL Image to upload.
        s3_key:      Original S3 key — used to derive the upload filename.
        corrections: List of annotation dicts (class, x, y, width, height).
        img_w:       Image width in pixels.
        img_h:       Image height in pixels.
        batch_name:  Roboflow batch name.
        dealership:  Dealership identifier derived from the bucket name.

    Returns:
        True on success, False on failure.
    """
    # Build name: dealership_scanfolder_subfolder_imagefile
    # s3_key: prior_condition/SCANNER_A_.../3/frame_001.jpg
    parts       = s3_key.split("/")
    scan_folder = parts[-3] if len(parts) >= 4 else "unknown"
    subfolder   = parts[-2] if len(parts) >= 3 else "unknown"
    orig_name   = parts[-1]
    image_name  = f"{dealership}_{scan_folder}_{subfolder}_{orig_name}"

    to_upload = corrections
    tmp_dir = img_tmp = xml_tmp = None

    try:
        project = _get_project()

        # Use a named temp dir so we control the filename Roboflow sees.
        tmp_dir = tempfile.mkdtemp()
        img_tmp = os.path.join(tmp_dir, image_name)
        with open(img_tmp, "wb") as f:
            f.write(_pil_to_bytes(pil_image))

        if to_upload:
            xml     = _build_voc_xml(image_name, img_w, img_h, to_upload)
            xml_tmp = os.path.join(tmp_dir, image_name.rsplit(".", 1)[0] + ".xml")
            with open(xml_tmp, "w", encoding="utf-8") as f:
                f.write(xml)

        project.upload(
            image_path        = img_tmp,
            annotation_path   = xml_tmp,
            batch_name        = batch_name,
            tag_names         = [],
            is_prediction     = True,
            num_retry_uploads = 3,
        )

        if to_upload:
            logger.info("    [roboflow] ✅ %d box(es) uploaded: %s", len(to_upload), image_name)
        else:
            logger.info("    [roboflow] 📭 Null label uploaded: %s", image_name)
        return True

    except Exception as e:
        logger.error("    [roboflow] ❌ Upload failed: %s", e)
        return False

    finally:
        for path in (img_tmp, xml_tmp):
            if path and os.path.exists(path):
                os.unlink(path)
        if tmp_dir and os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)
