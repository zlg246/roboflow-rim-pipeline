"""
Tests for roboflow_uploader.
All Roboflow SDK calls are mocked — no real API calls.
"""
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import roboflow_uploader


def _pil(w=640, h=480):
    return Image.new("RGB", (w, h), color=(100, 150, 200))


# Sample rim wheel S3 key
_RIM_KEY = "wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/left_rear_wheel_org_image.jpg"

# Polygon predictions — normalised [x, y] pairs
PRED_SCRATCH = {
    "class": "scratch", "confidence": 0.87,
    "polygon": [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]],
}
PRED_OTHER = {
    "class": "other",   "confidence": 0.65,
    "polygon": [[0.2, 0.3], [0.4, 0.3], [0.4, 0.5], [0.2, 0.5]],
}


# ── upload_rim ────────────────────────────────────────────────────────

class TestUploadRim:

    def test_upload_with_predictions_succeeds(self):
        mock_project = MagicMock()
        with patch("roboflow_uploader._get_rim_project", return_value=mock_project):
            ok = roboflow_uploader.upload_rim(
                _pil(), _RIM_KEY, [PRED_SCRATCH], 640, 480,
                "test_batch", "castle_hill_toyota",
            )
        assert ok is True
        mock_project.upload.assert_called_once()

    def test_upload_without_predictions_succeeds(self):
        """Empty predictions → image uploaded without annotation."""
        mock_project = MagicMock()
        with patch("roboflow_uploader._get_rim_project", return_value=mock_project):
            ok = roboflow_uploader.upload_rim(
                _pil(), _RIM_KEY, [], 640, 480,
                "test_batch", "castle_hill_toyota",
            )
        assert ok is True
        mock_project.upload.assert_called_once()
        _, kwargs = mock_project.upload.call_args
        assert kwargs.get("annotation_path") is None

    def test_correct_image_name_format(self):
        """Filename: dealership_scanfolder_wheelposition.jpg"""
        mock_project = MagicMock()
        with patch("roboflow_uploader._get_rim_project", return_value=mock_project):
            roboflow_uploader.upload_rim(
                _pil(), _RIM_KEY, [], 640, 480,
                "test_batch", "castle_hill_toyota",
            )
        _, kwargs = mock_project.upload.call_args
        assert kwargs["image_path"].endswith(
            "castle_hill_toyota_SCANNER_A_001608_2026-05-06_09-06-18_left_rear_wheel.jpg"
        )

    def test_annotation_path_is_json_when_predictions_present(self):
        mock_project = MagicMock()
        with patch("roboflow_uploader._get_rim_project", return_value=mock_project):
            roboflow_uploader.upload_rim(
                _pil(), _RIM_KEY, [PRED_SCRATCH], 640, 480,
                "test_batch", "castle_hill_toyota",
            )
        _, kwargs = mock_project.upload.call_args
        ann_path = kwargs.get("annotation_path")
        assert ann_path is not None
        assert ann_path.endswith(".json"), f"Expected .json annotation, got: {ann_path}"

    def test_different_wheel_positions(self):
        for wheel in ("left_rear_wheel", "right_front_wheel", "right_rear_wheel"):
            key = f"wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/{wheel}_org_image.jpg"
            mock_project = MagicMock()
            with patch("roboflow_uploader._get_rim_project", return_value=mock_project):
                ok = roboflow_uploader.upload_rim(
                    _pil(), key, [], 640, 480, "batch", "castle_hill_toyota",
                )
            assert ok is True
            _, kwargs = mock_project.upload.call_args
            assert wheel in kwargs["image_path"]

    def test_returns_false_on_sdk_exception(self):
        mock_project = MagicMock()
        mock_project.upload.side_effect = RuntimeError("API error")
        with patch("roboflow_uploader._get_rim_project", return_value=mock_project):
            ok = roboflow_uploader.upload_rim(
                _pil(), _RIM_KEY, [PRED_SCRATCH], 640, 480,
                "test_batch", "castle_hill_toyota",
            )
        assert ok is False


# ── _build_coco_json ──────────────────────────────────────────────────

class TestCocoJson:

    def test_top_level_keys(self):
        result = json.loads(
            roboflow_uploader._build_coco_json("test.jpg", 640, 480, [PRED_SCRATCH])
        )
        assert "images"      in result
        assert "annotations" in result
        assert "categories"  in result

    def test_image_metadata(self):
        result = json.loads(
            roboflow_uploader._build_coco_json("wheel.jpg", 1920, 1080, [PRED_SCRATCH])
        )
        img = result["images"][0]
        assert img["file_name"] == "wheel.jpg"
        assert img["width"]     == 1920
        assert img["height"]    == 1080

    def test_polygon_denormalised_to_pixel_coords(self):
        # PRED_SCRATCH polygon: [[0.1,0.2],[0.5,0.2],[0.5,0.6],[0.1,0.6]] on 640×480
        # Expected pixel: [64,96, 320,96, 320,288, 64,288]
        result = json.loads(
            roboflow_uploader._build_coco_json("test.jpg", 640, 480, [PRED_SCRATCH])
        )
        seg = result["annotations"][0]["segmentation"][0]
        assert seg[0] == pytest.approx(0.1 * 640)   # x1 = 64
        assert seg[1] == pytest.approx(0.2 * 480)   # y1 = 96
        assert seg[2] == pytest.approx(0.5 * 640)   # x2 = 320
        assert seg[3] == pytest.approx(0.2 * 480)   # y2 = 96

    def test_bbox_derived_from_polygon_extents(self):
        # polygon x range: 0.1–0.5 → 64–320 (w=256); y range: 0.2–0.6 → 96–288 (h=192)
        result = json.loads(
            roboflow_uploader._build_coco_json("test.jpg", 640, 480, [PRED_SCRATCH])
        )
        bbox = result["annotations"][0]["bbox"]
        assert bbox[0] == pytest.approx(0.1 * 640)   # x_min = 64
        assert bbox[1] == pytest.approx(0.2 * 480)   # y_min = 96
        assert bbox[2] == pytest.approx(0.4 * 640)   # width = 256
        assert bbox[3] == pytest.approx(0.4 * 480)   # height = 192

    def test_class_names_in_categories(self):
        result = json.loads(
            roboflow_uploader._build_coco_json("test.jpg", 640, 480, [PRED_SCRATCH, PRED_OTHER])
        )
        cat_names = {c["name"] for c in result["categories"]}
        assert "scratch" in cat_names
        assert "other"   in cat_names

    def test_multiple_annotations_get_unique_ids(self):
        result = json.loads(
            roboflow_uploader._build_coco_json("test.jpg", 640, 480, [PRED_SCRATCH, PRED_OTHER])
        )
        ids = [a["id"] for a in result["annotations"]]
        assert len(ids) == len(set(ids)), "Annotation IDs must be unique"
        assert len(result["annotations"]) == 2

    def test_category_id_matches_annotation_category_id(self):
        result = json.loads(
            roboflow_uploader._build_coco_json("test.jpg", 640, 480, [PRED_SCRATCH])
        )
        cat_ids  = {c["id"] for c in result["categories"]}
        ann_cats = {a["category_id"] for a in result["annotations"]}
        assert ann_cats.issubset(cat_ids), "All annotation category_ids must exist in categories"

    def test_filename_embedded(self):
        filename = "castle_hill_toyota_SCANNER_A_001608_2026-05-06_09-06-18_left_rear_wheel.jpg"
        result   = json.loads(
            roboflow_uploader._build_coco_json(filename, 640, 480, [PRED_SCRATCH])
        )
        assert result["images"][0]["file_name"] == filename
