#!/usr/bin/env python3
"""
debug_roboflow.py — Upload image WITHOUT annotation, create job, then add annotation.
Run with: python debug_roboflow.py
"""

import json
import os
import tempfile
import time

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from io import BytesIO

load_dotenv()

import config
from roboflow import Roboflow

WORKSPACE = config.ROBOFLOW_WORKSPACE
PROJECT   = config.ROBOFLOW_PROJECT
API_KEY   = config.ROBOFLOW_API_KEY

RF_BASE = "https://api.roboflow.com"


def make_image() -> tuple[bytes, int, int]:
    """Create a simple grey test image with a red bounding box drawn on it."""
    img  = Image.new("RGB", (640, 480), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 300, 250], outline="red", width=4)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue(), 640, 480


def make_xml(filename: str, img_w: int, img_h: int) -> str:
    """Build a minimal Pascal VOC XML annotation with a single scratch box."""
    return f"""<annotation>
  <filename>{filename}</filename>
  <size><width>{img_w}</width><height>{img_h}</height><depth>3</depth></size>
  <object>
    <name>scratch</name>
    <pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>100</xmin><ymin>100</ymin><xmax>300</xmax><ymax>250</ymax></bndbox>
  </object>
</annotation>"""


def main() -> None:
    """Run the debug workflow: upload image → create job → add annotation → check status."""
    ts        = str(int(time.time()))
    job_name  = f"review_workflow_{ts}"
    rf        = Roboflow(api_key=API_KEY)
    project   = rf.workspace(WORKSPACE).project(PROJECT)

    image_bytes, img_w, img_h = make_image()
    image_name = f"debug_workflow_{ts}.jpg"

    print(f"\n{'='*60}")
    print(f"  Job name : {job_name}")
    print(f"  Image    : {image_name}")
    print(f"{'='*60}\n")

    # ── Step 1: upload image WITHOUT annotation ───────────────────────
    print("── Step 1: Upload image only (no annotation) ──")
    img_tmp  = None
    image_id = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            img_tmp = f.name
        result = project.single_upload(
            image_path        = img_tmp,
            annotation_path   = None,
            batch_name        = job_name,
            is_prediction     = False,
            num_retry_uploads = 1,
        )
        image_id = (result or {}).get("image", {}).get("id")
        print(f"   image_id={image_id}  result={result}")
    finally:
        if img_tmp and os.path.exists(img_tmp):
            os.unlink(img_tmp)

    time.sleep(1)

    # ── Step 2: create job BEFORE adding annotation ───────────────────
    print(f"\n── Step 2: Create job (image has no annotation yet) ──")
    jobs_resp = requests.get(
        f"{RF_BASE}/{WORKSPACE}/{PROJECT}/jobs",
        params={"api_key": API_KEY},
    )
    api_batch = next(
        (j.get("sourceBatch") for j in jobs_resp.json().get("jobs", [])
         if j.get("sourceBatch", "").endswith("/api")),
        None,
    )
    if not api_batch:
        print("   ❌ Could not retrieve /api batch ID — aborting.")
        return

    payload = {
        "name":       job_name,
        "batch":      api_batch,
        "num_images": 1,
    }
    resp = requests.post(
        f"{RF_BASE}/{WORKSPACE}/{PROJECT}/jobs",
        params={"api_key": API_KEY},
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    print(f"   HTTP={resp.status_code} → {resp.text[:200]}")
    job_id = resp.json().get("id") if resp.status_code == 200 else None

    # ── Step 3: add annotation as a prediction (suggestion) ──────────
    # is_prediction=True uploads as a model suggestion, not ground truth,
    # keeping the image in review for human verification.
    if image_id:
        print(f"\n── Step 3: Add annotation as prediction (suggestion) ──")
        xml     = make_xml(image_name, img_w, img_h)
        xml_tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
                f.write(xml)
                xml_tmp = f.name
            result2 = project.single_upload(
                image_path        = None,
                image_id          = image_id,
                annotation_path   = xml_tmp,
                is_prediction     = True,
                num_retry_uploads = 1,
            )
            print(f"   result: {result2}")
        except Exception as e:
            print(f"   Error: {e}")
        finally:
            if xml_tmp and os.path.exists(xml_tmp):
                os.unlink(xml_tmp)

    time.sleep(2)

    # ── Step 4: check job status ──────────────────────────────────────
    print(f"\n── Step 4: Check jobs ──")
    jobs_resp = requests.get(f"{RF_BASE}/{WORKSPACE}/{PROJECT}/jobs", params={"api_key": API_KEY})
    for j in jobs_resp.json().get("jobs", []):
        if j.get("name") == job_name or j.get("id") == job_id:
            print(f"   name={j.get('name')!r}")
            print(f"   status={j.get('status')!r}")
            print(f"   numImages={j.get('numImages')}")
            print(f"   unannotated={j.get('unannotated')}")
            print(f"   annotated={j.get('annotated')}")
            print(f"   sourceBatch={j.get('sourceBatch')!r}")
            print(f"   → Check Roboflow UI: is this job in Review column?")


if __name__ == "__main__":
    main()
