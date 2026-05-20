"""
Tests for roboflow_uploader.
All HTTP calls are mocked — no real Roboflow API calls.
"""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import roboflow_uploader


def _pil(w=640, h=480):
    return Image.new("RGB", (w, h), color=(100, 150, 200))


KEEP = {"class": "scratch", "x": 0.5, "y": 0.4, "width": 0.2, "height": 0.15, "note": ""}
ADD  = {"class": "scratch", "x": 0.3, "y": 0.3, "width": 0.1, "height": 0.1,  "note": "missed"}
ADJ  = {"class": "other",   "x": 0.6, "y": 0.5, "width": 0.15,"height": 0.1,  "note": "reclassified"}


def _rf_mock(image_ok=True, ann_ok=True):
    call_n = {"n": 0}
    def post(*args, **kwargs):
        call_n["n"] += 1
        r = MagicMock()
        if call_n["n"] == 1:
            r.json.return_value = {"success": image_ok, "id": "img_abc123"}
        else:
            r.json.return_value = {"success": ann_ok}
        return r
    return post


# ── Happy paths ───────────────────────────────────────────────────────
def test_upload_with_keep_annotation():
    with patch("roboflow_uploader.requests.post", side_effect=_rf_mock()):
        ok = roboflow_uploader.upload(_pil(), "prior_condition/SESSION/3/f001.jpg",
                                      [KEEP], 640, 480, "test_batch")
    assert ok is True


def test_upload_uses_multipart_not_base64():
    """Image must be sent as multipart/form-data files= not base64 in body."""
    post_mock = MagicMock(side_effect=_rf_mock())
    with patch("roboflow_uploader.requests.post", post_mock):
        roboflow_uploader.upload(_pil(), "SESSION/3/f001.jpg",
                                 [KEEP], 640, 480, "test_batch")
    first_call_kwargs = post_mock.call_args_list[0].kwargs
    assert "files" in first_call_kwargs, "Image upload must use multipart files=, not base64 body"
    assert "data" not in first_call_kwargs or not first_call_kwargs.get("data")


def test_upload_with_add_annotation():
    with patch("roboflow_uploader.requests.post", side_effect=_rf_mock()):
        ok = roboflow_uploader.upload(_pil(), "SESSION/3/f001.jpg",
                                      [ADD], 640, 480, "test_batch")
    assert ok is True


def test_upload_null_label_only_one_post_call():
    """Empty corrections → image uploaded, no annotation POST."""
    post_mock = MagicMock(return_value=MagicMock(
        json=MagicMock(return_value={"success": True, "id": "img_xyz"})
    ))
    with patch("roboflow_uploader.requests.post", post_mock):
        ok = roboflow_uploader.upload(_pil(), "SESSION/3/clean.jpg",
                                      [], 640, 480, "test_batch")
    assert ok is True
    assert post_mock.call_count == 1   # only image upload


def test_adjust_correction_uploaded():
    post_mock = MagicMock(side_effect=_rf_mock())
    with patch("roboflow_uploader.requests.post", post_mock):
        ok = roboflow_uploader.upload(_pil(), "SESSION/7/f001.jpg",
                                      [ADJ], 640, 480, "test_batch")
    assert ok is True
    assert post_mock.call_count == 2


# ── Failure paths ─────────────────────────────────────────────────────
def test_returns_false_on_image_upload_failure():
    post_mock = MagicMock(return_value=MagicMock(
        json=MagicMock(return_value={"success": False, "error": "invalid key"})
    ))
    with patch("roboflow_uploader.requests.post", post_mock):
        ok = roboflow_uploader.upload(_pil(), "SESSION/3/f001.jpg",
                                      [KEEP], 640, 480, "test_batch")
    assert ok is False


# ── VOC XML coordinate conversion ────────────────────────────────────
class TestVocXml:

    def test_coordinate_conversion(self):
        # KEEP: centre=(0.5,0.4) size=(0.2,0.15) on 640×480
        # cx=320 cy=192 w=128 h=72 → xmin=256 ymin=156 xmax=384 ymax=228
        xml = roboflow_uploader._build_voc_xml("test.jpg", 640, 480, [KEEP])
        assert "<xmin>256</xmin>" in xml
        assert "<ymin>156</ymin>" in xml
        assert "<xmax>384</xmax>" in xml
        assert "<ymax>228</ymax>" in xml

    def test_class_names_in_xml(self):
        xml = roboflow_uploader._build_voc_xml("test.jpg", 640, 480, [KEEP, ADJ])
        assert "<name>scratch</name>" in xml
        assert "<name>other</name>"   in xml

    def test_coordinates_clamped_to_image_bounds(self):
        edge = {"class": "scratch",
                "x": 0.99, "y": 0.99, "width": 0.5, "height": 0.5, "note": ""}
        xml = roboflow_uploader._build_voc_xml("test.jpg", 640, 480, [edge])
        assert "<xmax>640</xmax>" in xml
        assert "<ymax>480</ymax>" in xml

    def test_s3_key_slashes_replaced_in_image_name(self):
        image_name = "prior_condition__SESSION__3__f001.jpg"
        xml = roboflow_uploader._build_voc_xml(image_name, 640, 480, [KEEP])
        assert image_name in xml
