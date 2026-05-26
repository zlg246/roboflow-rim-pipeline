"""
Tests for yolo_runner.
YOLO model is fully mocked — no GPU or model file required.
"""
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import yolo_runner


def _make_pil():
    return Image.new("RGB", (640, 480))


def _make_mock_results(boxes_data, polygons=None):
    """
    Build a mock Ultralytics result for a segmentation model.

    Args:
        boxes_data: list of (x, y, w, h, cls_idx, conf) — normalised 0-1.
        polygons:   list of polygon arrays, one per detection.
                    Each polygon is a list of [x_norm, y_norm] pairs.
                    Pass None to simulate a model that produced no masks
                    (e.g. an object-detection model).
    """
    mock_results       = MagicMock()
    mock_results.names = {0: "scratch", 1: "other"}
    mock_boxes         = []

    for x, y, w, h, cls_idx, conf in boxes_data:
        box          = MagicMock()
        box.xywhn    = [torch.tensor([x, y, w, h])]
        box.cls      = torch.tensor([cls_idx])
        box.conf     = torch.tensor([conf])
        mock_boxes.append(box)

    mock_results.boxes = mock_boxes

    if polygons is not None:
        mock_masks      = MagicMock()
        mock_masks.xyn  = [np.array(p, dtype=float) for p in polygons]
        mock_results.masks = mock_masks
        # len(mock_results.masks) must return the correct count
        mock_masks.__len__ = lambda self: len(polygons)
    else:
        mock_results.masks = None

    return mock_results


# ── run_rim_inference — polygon extraction ────────────────────────────

def test_returns_list_of_dicts_with_polygon():
    poly = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]]
    mock_model = MagicMock(
        return_value=[_make_mock_results([(0.3, 0.4, 0.4, 0.4, 0, 0.87)], polygons=[poly])]
    )
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    assert len(preds) == 1
    p = preds[0]
    assert p["class"]      == "scratch"
    assert p["confidence"] == 0.87
    assert "polygon" in p
    assert len(p["polygon"]) == 4
    for pt in p["polygon"]:
        assert len(pt) == 2
        assert 0.0 <= pt[0] <= 1.0
        assert 0.0 <= pt[1] <= 1.0


def test_no_detections_returns_empty_list():
    mock_model = MagicMock(return_value=[_make_mock_results([], polygons=[])])
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())
    assert preds == []


def test_multiple_detections_multiple_polygons():
    polys = [
        [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],   # scratch
        [[0.5, 0.5], [0.8, 0.5], [0.8, 0.9], [0.5, 0.9]],   # other
    ]
    boxes = [
        (0.2, 0.2, 0.2, 0.2, 0, 0.90),
        (0.65, 0.7, 0.3, 0.4, 1, 0.65),
    ]
    mock_model = MagicMock(return_value=[_make_mock_results(boxes, polygons=polys)])
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    assert len(preds) == 2
    assert {p["class"] for p in preds} == {"scratch", "other"}
    assert all("polygon" in p for p in preds)
    assert all(len(p["polygon"]) == 4 for p in preds)


def test_polygon_coords_normalised():
    """All polygon coordinates must stay within [0, 1]."""
    poly = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    mock_model = MagicMock(
        return_value=[_make_mock_results([(0.5, 0.5, 1.0, 1.0, 0, 0.9)], polygons=[poly])]
    )
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    for pt in preds[0]["polygon"]:
        assert 0.0 <= pt[0] <= 1.0
        assert 0.0 <= pt[1] <= 1.0


def test_no_masks_logs_warning_and_returns_empty_polygon(caplog):
    """When masks are None (object-detection model), warn and return empty polygon."""
    import logging
    mock_model = MagicMock(
        return_value=[_make_mock_results([(0.5, 0.5, 0.2, 0.2, 0, 0.85)], polygons=None)]
    )
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        with caplog.at_level(logging.WARNING, logger="yolo_runner"):
            preds = yolo_runner.run_rim_inference(_make_pil())

    assert len(preds) == 1
    assert preds[0]["polygon"] == []
    assert "no polygon masks" in caplog.text.lower() or "segmentation" in caplog.text.lower()


def test_rim_model_loaded_once_then_cached():
    """_get_rim_model() should instantiate YOLO only once across multiple calls."""
    mock_model = MagicMock(return_value=[_make_mock_results([], polygons=[])])

    with patch("yolo_runner._rim_model", None), \
         patch("yolo_runner.YOLO", return_value=mock_model) as mock_yolo:
        yolo_runner._rim_model = None
        yolo_runner._get_rim_model()
        yolo_runner._get_rim_model()
        assert mock_yolo.call_count == 1
