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

# Image dimensions used throughout the tests
IMG_W, IMG_H = 640, 480


def _make_pil(w=IMG_W, h=IMG_H):
    return Image.new("RGB", (w, h))


def _norm_to_px(polygon_norm):
    """Convert normalised [[x,y],...] to pixel [[x,y],...] for the test image."""
    return [[pt[0] * IMG_W, pt[1] * IMG_H] for pt in polygon_norm]


def _make_mock_results(boxes_data, polygons_px=None):
    """
    Build a mock Ultralytics result for a segmentation model.

    Args:
        boxes_data:  list of (x, y, w, h, cls_idx, conf) in normalised coords.
        polygons_px: list of pixel-space polygon arrays [[x, y], ...], one per
                     detection.  ``None`` simulates an object-detection model
                     (no masks).  Pass ``[]`` for a seg model with no detections.
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

    if polygons_px is not None:
        mock_masks     = MagicMock()
        # masks.xy — pixel-space contour arrays (what the updated code reads)
        mock_masks.xy  = [np.array(p, dtype=float) for p in polygons_px]
        mock_masks.__len__ = lambda self: len(polygons_px)
        mock_results.masks = mock_masks
    else:
        mock_results.masks = None

    return mock_results


# ── run_rim_inference — polygon extraction ────────────────────────────

def test_returns_list_of_dicts_with_polygon():
    poly_norm = [[0.1, 0.2], [0.5, 0.2], [0.5, 0.6], [0.1, 0.6]]
    poly_px   = _norm_to_px(poly_norm)
    mock_model = MagicMock(
        return_value=[_make_mock_results([(0.3, 0.4, 0.4, 0.4, 0, 0.87)],
                                         polygons_px=[poly_px])]
    )
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    assert len(preds) == 1
    p = preds[0]
    assert p["class"]      == "scratch"
    assert p["confidence"] == 0.87
    assert "polygon" in p
    assert len(p["polygon"]) >= 3           # at least 3 points (triangle minimum)
    for pt in p["polygon"]:
        assert len(pt) == 2
        assert 0.0 <= pt[0] <= 1.0
        assert 0.0 <= pt[1] <= 1.0


def test_no_detections_returns_empty_list():
    mock_model = MagicMock(return_value=[_make_mock_results([], polygons_px=[])])
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())
    assert preds == []


def test_multiple_detections_multiple_polygons():
    poly1_norm = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
    poly2_norm = [[0.5, 0.5], [0.8, 0.5], [0.8, 0.9], [0.5, 0.9]]
    boxes = [(0.2, 0.2, 0.2, 0.2, 0, 0.90), (0.65, 0.7, 0.3, 0.4, 1, 0.65)]
    mock_model = MagicMock(return_value=[_make_mock_results(
        boxes,
        polygons_px=[_norm_to_px(poly1_norm), _norm_to_px(poly2_norm)],
    )])
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    assert len(preds) == 2
    assert {p["class"] for p in preds} == {"scratch", "other"}
    assert all("polygon" in p for p in preds)
    assert all(len(p["polygon"]) >= 3 for p in preds)


def test_polygon_coords_normalised():
    """All polygon coordinates must stay within [0, 1] after simplification."""
    # Axis-aligned rectangle occupying the full image
    poly_px = [[0.0, 0.0], [IMG_W, 0.0], [IMG_W, IMG_H], [0.0, IMG_H]]
    mock_model = MagicMock(return_value=[_make_mock_results(
        [(0.5, 0.5, 1.0, 1.0, 0, 0.9)], polygons_px=[poly_px]
    )])
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    for pt in preds[0]["polygon"]:
        assert 0.0 <= pt[0] <= 1.0
        assert 0.0 <= pt[1] <= 1.0


def test_simplification_reduces_point_count():
    """
    A densely-sampled collinear contour should be collapsed to far fewer points.

    We build a rectangle whose edges are sampled at 1-pixel intervals (~1200 pts
    total), then verify approxPolyDP (epsilon=2.0 px) collapses them to ~4 pts.
    """
    # Dense rectangle: sample each edge every pixel
    top    = [[x, 0]       for x in range(0, IMG_W + 1)]
    right  = [[IMG_W, y]   for y in range(1, IMG_H + 1)]
    bottom = [[x, IMG_H]   for x in range(IMG_W - 1, -1, -1)]
    left   = [[0, y]       for y in range(IMG_H - 1, 0, -1)]
    dense_poly_px = top + right + bottom + left   # ~1200 points

    mock_model = MagicMock(return_value=[_make_mock_results(
        [(0.5, 0.5, 1.0, 1.0, 0, 0.9)], polygons_px=[dense_poly_px]
    )])
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    n_out = len(preds[0]["polygon"])
    assert n_out < 20, (
        f"Expected far fewer than 1200 points after simplification, got {n_out}"
    )
    assert n_out >= 3, "Polygon must have at least 3 points"


def test_simplification_preserves_corners():
    """
    The four corners of a rectangle must survive simplification.
    We verify that each corner pixel appears (within rounding) in the output.
    """
    # Rectangle with slightly noisy intermediate points along each edge
    rng = np.random.default_rng(42)
    corners = [(0, 0), (IMG_W, 0), (IMG_W, IMG_H), (0, IMG_H)]
    noisy_poly = []
    for i in range(4):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % 4]
        # 20 intermediate points with small noise
        ts = np.linspace(0, 1, 22)[1:-1]
        for t in ts:
            noise = rng.uniform(-0.5, 0.5, 2)
            noisy_poly.append([x0 + t * (x1 - x0) + noise[0],
                                y0 + t * (y1 - y0) + noise[1]])
        noisy_poly.append([float(x1), float(y1)])

    mock_model = MagicMock(return_value=[_make_mock_results(
        [(0.5, 0.5, 1.0, 1.0, 0, 0.9)], polygons_px=[noisy_poly]
    )])
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        preds = yolo_runner.run_rim_inference(_make_pil())

    pts = preds[0]["polygon"]
    # All corners should map to (roughly) 0.0 or 1.0 after normalisation
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    tol = 0.01    # 1% of image dimension
    assert min(xs) < tol,        "Left edge corner missing"
    assert max(xs) > 1.0 - tol,  "Right edge corner missing"
    assert min(ys) < tol,        "Top edge corner missing"
    assert max(ys) > 1.0 - tol,  "Bottom edge corner missing"


def test_no_masks_logs_warning_and_returns_empty_polygon(caplog):
    """When masks are None (object-detection model), warn and return empty polygon."""
    import logging
    mock_model = MagicMock(
        return_value=[_make_mock_results([(0.5, 0.5, 0.2, 0.2, 0, 0.85)],
                                          polygons_px=None)]
    )
    with patch("yolo_runner._get_rim_model", return_value=mock_model):
        with caplog.at_level(logging.WARNING, logger="yolo_runner"):
            preds = yolo_runner.run_rim_inference(_make_pil())

    assert len(preds) == 1
    assert preds[0]["polygon"] == []
    assert "no polygon masks" in caplog.text.lower() or "segmentation" in caplog.text.lower()


def test_rim_model_loaded_once_then_cached():
    """_get_rim_model() should instantiate YOLO only once across multiple calls."""
    mock_model = MagicMock(return_value=[_make_mock_results([], polygons_px=[])])

    with patch("yolo_runner._rim_model", None), \
         patch("yolo_runner.YOLO", return_value=mock_model) as mock_yolo:
        yolo_runner._rim_model = None
        yolo_runner._get_rim_model()
        yolo_runner._get_rim_model()
        assert mock_yolo.call_count == 1


# ── _simplify_polygon unit tests ──────────────────────────────────────

class TestSimplifyPolygon:
    """Direct unit tests for the _simplify_polygon helper."""

    def test_empty_input_returns_empty(self):
        result = yolo_runner._simplify_polygon(np.array([]), IMG_W, IMG_H, 2.0)
        assert result == []

    def test_fewer_than_4_points_normalised_as_is(self):
        pts = np.array([[64.0, 96.0], [320.0, 96.0], [192.0, 288.0]])
        result = yolo_runner._simplify_polygon(pts, IMG_W, IMG_H, 2.0)
        assert len(result) == 3
        assert result[0][0] == pytest.approx(64.0 / IMG_W, abs=1e-3)
        assert result[0][1] == pytest.approx(96.0 / IMG_H, abs=1e-3)

    def test_output_coordinates_within_unit_range(self):
        pts = np.array([[0.0, 0.0], [IMG_W, 0.0], [IMG_W, IMG_H], [0.0, IMG_H]])
        result = yolo_runner._simplify_polygon(pts, IMG_W, IMG_H, 2.0)
        for pt in result:
            assert 0.0 <= pt[0] <= 1.0
            assert 0.0 <= pt[1] <= 1.0

    def test_collinear_points_removed(self):
        # 100 collinear points along the top edge + 3 corners
        top = [[float(x), 0.0] for x in range(0, IMG_W + 1, 5)]
        pts = np.array(top + [[IMG_W, IMG_H], [0.0, IMG_H]])
        result = yolo_runner._simplify_polygon(pts, IMG_W, IMG_H, 2.0)
        assert len(result) < len(pts)
        assert len(result) >= 3

    def test_output_is_list_of_pairs(self):
        pts = np.array([[0.0, 0.0], [IMG_W, 0.0], [IMG_W, IMG_H], [0.0, IMG_H]])
        result = yolo_runner._simplify_polygon(pts, IMG_W, IMG_H, 2.0)
        assert isinstance(result, list)
        for pt in result:
            assert isinstance(pt, list)
            assert len(pt) == 2
