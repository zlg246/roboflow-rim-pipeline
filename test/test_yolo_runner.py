"""
Tests for yolo_runner.
YOLO model is fully mocked — no GPU or model file required.
"""
import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import yolo_runner


def _make_pil():
    return Image.new("RGB", (640, 480))


def _make_mock_results(boxes_data):
    """
    boxes_data: list of (x, y, w, h, cls_idx, conf) — all normalised 0-1
    """
    mock_results        = MagicMock()
    mock_results.names  = {0: "scratch", 1: "other"}
    mock_boxes          = []

    for x, y, w, h, cls_idx, conf in boxes_data:
        box          = MagicMock()
        box.xywhn    = [torch.tensor([x, y, w, h])]
        box.cls      = torch.tensor([cls_idx])
        box.conf     = torch.tensor([conf])
        mock_boxes.append(box)

    mock_results.boxes = mock_boxes
    return mock_results


# ── run_inference ─────────────────────────────────────────────────────
def test_returns_list_of_dicts():
    mock_model = MagicMock(return_value=[_make_mock_results([(0.5, 0.4, 0.2, 0.15, 0, 0.87)])])
    with patch("yolo_runner._get_model", return_value=mock_model):
        preds = yolo_runner.run_inference(_make_pil())
    assert len(preds) == 1
    p = preds[0]
    assert p["class"]      == "scratch"
    assert p["confidence"] == 0.87
    assert 0 <= p["x"] <= 1
    assert 0 <= p["y"] <= 1
    assert 0 <= p["width"] <= 1
    assert 0 <= p["height"] <= 1


def test_no_detections_returns_empty_list():
    mock_model = MagicMock(return_value=[_make_mock_results([])])
    with patch("yolo_runner._get_model", return_value=mock_model):
        preds = yolo_runner.run_inference(_make_pil())
    assert preds == []


def test_multiple_boxes_different_classes():
    boxes = [
        (0.3, 0.3, 0.1, 0.1, 0, 0.90),   # scratch
        (0.7, 0.6, 0.2, 0.2, 1, 0.65),   # other
    ]
    mock_model = MagicMock(return_value=[_make_mock_results(boxes)])
    with patch("yolo_runner._get_model", return_value=mock_model):
        preds = yolo_runner.run_inference(_make_pil())
    assert len(preds) == 2
    assert {p["class"] for p in preds} == {"scratch", "other"}


def test_coordinates_rounded_to_4dp():
    boxes = [(0.123456789, 0.987654321, 0.111111, 0.222222, 0, 0.9)]
    mock_model = MagicMock(return_value=[_make_mock_results(boxes)])
    with patch("yolo_runner._get_model", return_value=mock_model):
        preds = yolo_runner.run_inference(_make_pil())
    for key in ("x", "y", "width", "height"):
        val_str = str(preds[0][key])
        decimal_part = val_str.split(".")[-1].rstrip("0") if "." in val_str else ""
        assert len(decimal_part) <= 4


def test_model_loaded_once_then_cached():
    """_get_model() should only instantiate YOLO once across multiple calls."""
    mock_model = MagicMock(return_value=[_make_mock_results([])])

    with patch("yolo_runner._model", None), \
         patch("yolo_runner.YOLO", return_value=mock_model) as mock_yolo:
        yolo_runner._model = None
        yolo_runner._get_model()
        yolo_runner._get_model()
        assert mock_yolo.call_count == 1
