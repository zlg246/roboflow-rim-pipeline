"""
Tests for s3_loader.
All S3 calls are mocked — no real AWS credentials or network required.
"""
import base64
import pytest
from datetime import date, datetime, timezone
from unittest.mock import patch, MagicMock, call
from PIL import Image
from io import BytesIO
import s3_loader


# ── Helpers ───────────────────────────────────────────────────────────

def _make_jpeg_bytes(w=100, h=80):
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(120, 60, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def _rim_key(date_str="2026-05-06", wheel="left_rear_wheel", time_str="09-06-18"):
    """Build a wheel-scanner S3 key matching the real bucket structure."""
    return (
        f"wheel_scanner/"
        f"SCANNER_A_001608_{date_str}_{time_str}/"
        f"{wheel}_org_image.jpg"
    )


def _mock_rim_paginator(sessions_and_files: dict[str, list[str]]):
    """
    Simulate the two-pass paginator used by list_rim_image_keys.

    ``sessions_and_files`` maps session_prefix → list of filenames inside it.

    Pass 1 (Delimiter='/'): returns CommonPrefixes for each session prefix.
    Pass 2 (no Delimiter):   returns Contents for keys under the requested prefix.
    """
    def paginate(Bucket=None, Prefix="", Delimiter=None):
        if Delimiter == "/":
            # Pass 1: return session-level CommonPrefixes
            prefixes = [
                {"Prefix": sp}
                for sp in sessions_and_files
                if sp.startswith(Prefix)
            ]
            return [{"CommonPrefixes": prefixes}]
        else:
            # Pass 2: return file Contents under the given prefix
            contents = [
                {"Key": Prefix + fname}
                for sp, fnames in sessions_and_files.items()
                if Prefix == sp
                for fname in fnames
            ]
            return [{"Contents": contents}] if contents else [{}]

    mock_pag = MagicMock()
    mock_pag.paginate.side_effect = paginate
    return mock_pag


def _mock_client(paginator, get_object_response=None):
    """Return a mock S3 client that delegates pagination and get_object."""
    client = MagicMock()
    client.get_paginator.return_value = paginator
    if get_object_response:
        client.get_object.return_value = get_object_response
    return client


# ── list_rim_image_keys ───────────────────────────────────────────────

class TestListRimImageKeys:

    def test_returns_only_org_image_files(self):
        sessions = {
            "wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/": [
                "left_rear_wheel_org_image.jpg",
                "left_rear_wheel_scratch_bb_overlay.jpg",   # excluded
                "right_front_wheel_org_image.jpg",
                "right_rear_wheel_org_image.jpg",
            ],
        }
        pag    = _mock_rim_paginator(sessions)
        client = _mock_client(pag)
        with patch("s3_loader._get_client", return_value=client):
            keys = s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 6), date_to=date(2026, 5, 6),
            )
        assert len(keys) == 3
        assert all("_org_image.jpg" in k for k in keys)
        assert not any("_overlay" in k for k in keys)

    def test_date_range_filters_sessions(self):
        sessions = {
            "wheel_scanner/SCANNER_A_001608_2026-05-05_09-00-00/": ["left_rear_wheel_org_image.jpg"],  # before
            "wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/": ["left_rear_wheel_org_image.jpg"],  # in range
            "wheel_scanner/SCANNER_A_001608_2026-05-07_10-00-00/": ["left_rear_wheel_org_image.jpg"],  # after
        }
        pag    = _mock_rim_paginator(sessions)
        client = _mock_client(pag)
        with patch("s3_loader._get_client", return_value=client):
            keys = s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 6), date_to=date(2026, 5, 6),
            )
        assert len(keys) == 1
        assert "2026-05-06" in keys[0]

    def test_inclusive_date_boundaries(self):
        sessions = {
            "wheel_scanner/SCANNER_A_001608_2026-05-06_09-00-00/": ["left_rear_wheel_org_image.jpg"],
            "wheel_scanner/SCANNER_A_001608_2026-05-09_17-00-00/": ["right_rear_wheel_org_image.jpg"],
        }
        pag    = _mock_rim_paginator(sessions)
        client = _mock_client(pag)
        with patch("s3_loader._get_client", return_value=client):
            keys = s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 6), date_to=date(2026, 5, 9),
            )
        assert len(keys) == 2

    def test_multiple_sessions_same_day(self):
        sessions = {
            "wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/": ["left_rear_wheel_org_image.jpg"],
            "wheel_scanner/SCANNER_A_001608_2026-05-06_14-30-00/": ["right_front_wheel_org_image.jpg"],
        }
        pag    = _mock_rim_paginator(sessions)
        client = _mock_client(pag)
        with patch("s3_loader._get_client", return_value=client):
            keys = s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 6), date_to=date(2026, 5, 6),
            )
        assert len(keys) == 2

    def test_empty_bucket_returns_empty_list(self):
        pag    = _mock_rim_paginator({})
        client = _mock_client(pag)
        with patch("s3_loader._get_client", return_value=client):
            keys = s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 6), date_to=date(2026, 5, 6),
            )
        assert keys == []

    def test_returns_sorted_keys(self):
        sessions = {
            "wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/": [
                "right_rear_wheel_org_image.jpg",
                "left_rear_wheel_org_image.jpg",
                "right_front_wheel_org_image.jpg",
            ],
        }
        pag    = _mock_rim_paginator(sessions)
        client = _mock_client(pag)
        with patch("s3_loader._get_client", return_value=client):
            keys = s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 6), date_to=date(2026, 5, 6),
            )
        assert keys == sorted(keys)

    def test_invalid_date_range_raises_value_error(self):
        with pytest.raises(ValueError, match="date_from"):
            s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 9),
                date_to=date(2026, 5, 6),
            )

    def test_overlay_files_excluded_and_counted(self):
        """All non-org files (overlays, metadata, etc.) are silently skipped."""
        sessions = {
            "wheel_scanner/SCANNER_A_001608_2026-05-06_09-06-18/": [
                "left_rear_wheel_org_image.jpg",
                "left_rear_wheel_scratch_bb_overlay.jpg",
                "right_rear_wheel_org_image.jpg",
                "right_rear_wheel_scratch_bb_overlay.jpg",
                "meta.json",
            ],
        }
        pag    = _mock_rim_paginator(sessions)
        client = _mock_client(pag)
        with patch("s3_loader._get_client", return_value=client):
            keys = s3_loader.list_rim_image_keys(
                date_from=date(2026, 5, 6), date_to=date(2026, 5, 6),
            )
        assert len(keys) == 2


# ── upload_log_to_s3 — never writes to production ────────────────────

class TestUploadLogSafety:

    def test_skips_upload_when_log_bucket_empty(self, tmp_path, caplog):
        import logging
        log = tmp_path / "test.jsonl"
        log.write_text('{"s3_key": "test"}\n')
        import config as cfg
        original = cfg.LOG_BUCKET
        cfg.LOG_BUCKET = ""
        try:
            with caplog.at_level(logging.INFO, logger="s3_loader"):
                s3_loader.upload_log_to_s3(str(log))
        finally:
            cfg.LOG_BUCKET = original
        assert "local only" in caplog.text

    def test_refuses_to_write_to_production_bucket(self, tmp_path, caplog):
        import logging
        log = tmp_path / "test.jsonl"
        log.write_text('{"s3_key": "test"}\n')
        import config as cfg
        original_log = cfg.LOG_BUCKET
        cfg.LOG_BUCKET = cfg.S3_BUCKET    # same as production — should refuse
        try:
            with caplog.at_level(logging.WARNING, logger="s3_loader"):
                s3_loader.upload_log_to_s3(str(log))
        finally:
            cfg.LOG_BUCKET = original_log
        assert "Skipping" in caplog.text or "protect" in caplog.text


# ── load_image ────────────────────────────────────────────────────────

class TestLoadImage:

    def test_returns_correct_types(self):
        jpeg_bytes = _make_jpeg_bytes()
        mock_resp  = {"Body": MagicMock(read=MagicMock(return_value=jpeg_bytes))}
        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_resp
        with patch("s3_loader._get_client", return_value=mock_client):
            pil, b64, mime = s3_loader.load_image(_rim_key())
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"
        assert mime == "image/jpeg"
        assert base64.b64decode(b64) == jpeg_bytes

    def test_png_mime_type(self):
        buf = BytesIO()
        Image.new("RGB", (50, 50)).save(buf, format="PNG")
        raw = buf.getvalue()
        mock_resp   = {"Body": MagicMock(read=MagicMock(return_value=raw))}
        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_resp
        with patch("s3_loader._get_client", return_value=mock_client):
            _, _, mime = s3_loader.load_image("wheel_scanner/SESSION/car.png")
        assert mime == "image/png"
