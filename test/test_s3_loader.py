"""
Tests for s3_loader.
All S3 calls are mocked — no real AWS credentials or network required.
"""
import base64, pytest
from datetime import date, datetime, timezone
from unittest.mock import patch, MagicMock
from PIL import Image
from io import BytesIO
import s3_loader


# ── Helpers ───────────────────────────────────────────────────────────
def _make_jpeg_bytes(w=100, h=80):
    buf = BytesIO()
    Image.new("RGB", (w, h), color=(120, 60, 30)).save(buf, format="JPEG")
    return buf.getvalue()

def _dt(d: date):
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

def _obj(key, d=date(2026, 3, 16)):
    return {"Key": key, "LastModified": _dt(d)}

def _mock_paginator(objects):
    """
    Smart paginator mock for the two-pass list_image_keys.

    Pass 1 (Delimiter='/'):  returns CommonPrefixes derived from the object keys.
    Pass 2 (no Delimiter):   returns Contents filtered by the requested Prefix.
    """
    session_prefixes = sorted(set(
        "/".join(obj["Key"].split("/")[:2]) + "/"
        for obj in objects
        if len(obj["Key"].split("/")) >= 4
    ))

    def paginate(*args, **kwargs):
        if kwargs.get("Delimiter") == "/":
            prefix = kwargs.get("Prefix", "")
            common = [{"Prefix": p} for p in session_prefixes if p.startswith(prefix)]
            return [{"CommonPrefixes": common}]
        else:
            prefix = kwargs.get("Prefix", "")
            matching = [o for o in objects if o["Key"].startswith(prefix)]
            return [{"Contents": matching}] if matching else [{}]

    m = MagicMock()
    m.paginate.side_effect = paginate
    return m

# Convenience key builder matching the real bucket structure
def _key(date_str="2026-03-16", subfolder="3", n=1, time_str="14-10-49"):
    return (
        f"prior_condition/"
        f"SCANNER_A_0009bf_{date_str}_{time_str}/"
        f"{subfolder}/frame_{n:03d}.jpg"
    )


# ── _parse_session_info ───────────────────────────────────────────────
class TestParseSessionInfo:

    def test_standard_key_subfolder_3(self):
        session, sub, d = s3_loader._parse_session_info(_key("2026-03-16", "3"))
        assert session == "SCANNER_A_0009bf_2026-03-16_14-10-49"
        assert sub     == "3"
        assert d       == date(2026, 3, 16)

    def test_standard_key_subfolder_7(self):
        _, sub, _ = s3_loader._parse_session_info(_key("2026-03-16", "7"))
        assert sub == "7"

    def test_any_subfolder_is_parsed(self):
        for sf in ("1", "2", "4", "5", "6", "8"):
            _, sub, _ = s3_loader._parse_session_info(_key("2026-03-16", sf))
            assert sub == sf

    def test_different_device_id(self):
        key = "prior_condition/SCANNER_B_aabbcc_2026-05-15_09-00-00/3/img.jpg"
        _, _, d = s3_loader._parse_session_info(key)
        assert d == date(2026, 5, 15)

    def test_non_matching_key_returns_nones(self):
        result = s3_loader._parse_session_info("some/random/path.jpg")
        assert result == (None, None, None)

    def test_key_too_short_returns_nones(self):
        result = s3_loader._parse_session_info("frame.jpg")
        assert result == (None, None, None)

    def test_multiple_sessions_same_day_different_times(self):
        key1 = _key("2026-03-16", "3", time_str="09-00-00")
        key2 = _key("2026-03-16", "3", time_str="17-30-00")
        _, _, d1 = s3_loader._parse_session_info(key1)
        _, _, d2 = s3_loader._parse_session_info(key2)
        assert d1 == d2 == date(2026, 3, 16)


# ── list_image_keys ───────────────────────────────────────────────────
class TestListImageKeys:

    def test_returns_only_target_subfolders_3_and_7(self):
        objects = [
            _obj(_key("2026-03-16", "3")),
            _obj(_key("2026-03-16", "7")),
            _obj(_key("2026-03-16", "1")),   # excluded
            _obj(_key("2026-03-16", "5")),   # excluded
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 16),
                target_subfolders={"3", "7"},
            )
        assert len(keys) == 2
        subfolders_found = {k.split("/")[-2] for k in keys}
        assert subfolders_found == {"3", "7"}

    def test_date_range_filters_sessions(self):
        objects = [
            _obj(_key("2026-03-16", "3"), date(2026, 3, 16)),  # in range
            _obj(_key("2026-03-17", "3"), date(2026, 3, 17)),  # in range
            _obj(_key("2026-03-15", "3"), date(2026, 3, 15)),  # before
            _obj(_key("2026-03-20", "3"), date(2026, 3, 20)),  # after
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 17),
                target_subfolders={"3", "7"},
            )
        assert len(keys) == 2

    def test_inclusive_date_boundaries(self):
        objects = [
            _obj(_key("2026-03-16", "3"), date(2026, 3, 16)),  # from boundary
            _obj(_key("2026-03-20", "7"), date(2026, 3, 20)),  # to boundary
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 20),
                target_subfolders={"3", "7"},
            )
        assert len(keys) == 2

    def test_single_day_range(self):
        objects = [
            _obj(_key("2026-03-16", "3")),
            _obj(_key("2026-03-17", "3")),
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 16),
                target_subfolders={"3", "7"},
            )
        assert len(keys) == 1

    def test_multiple_sessions_same_day(self):
        objects = [
            _obj("prior_condition/SCANNER_A_0009bf_2026-03-16_09-00-00/3/f001.jpg"),
            _obj("prior_condition/SCANNER_A_0009bf_2026-03-16_14-10-49/3/f001.jpg"),
            _obj("prior_condition/SCANNER_A_0009bf_2026-03-16_17-30-00/7/f001.jpg"),
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 16),
                target_subfolders={"3", "7"},
            )
        assert len(keys) == 3

    def test_non_image_files_excluded(self):
        base = "prior_condition/SCANNER_A_0009bf_2026-03-16_14-10-49/3/"
        objects = [
            _obj(base + "frame.jpg"),
            _obj(base + "thumb.db"),
            _obj(base + "meta.json"),
            _obj(base + "frame2.png"),
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 16),
            )
        assert len(keys) == 2
        assert all(k.endswith((".jpg", ".png")) for k in keys)

    def test_returns_sorted_keys(self):
        objects = [
            _obj(_key("2026-03-16", "7", 3)),
            _obj(_key("2026-03-16", "3", 1)),
            _obj(_key("2026-03-16", "7", 1)),
            _obj(_key("2026-03-16", "3", 2)),
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 16),
            )
        assert keys == sorted(keys)

    def test_invalid_date_range_raises_value_error(self):
        with pytest.raises(ValueError, match="date_from"):
            s3_loader.list_image_keys(
                date_from=date(2026, 3, 20),
                date_to=date(2026, 3, 16),
            )

    def test_empty_bucket_returns_empty_list(self):
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator([])):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 20),
            )
        assert keys == []

    def test_custom_subfolders_override(self):
        objects = [
            _obj(_key("2026-03-16", "3")),
            _obj(_key("2026-03-16", "5")),
            _obj(_key("2026-03-16", "7")),
        ]
        with patch.object(s3_loader.s3, "get_paginator",
                          return_value=_mock_paginator(objects)):
            keys = s3_loader.list_image_keys(
                date_from=date(2026, 3, 16), date_to=date(2026, 3, 16),
                target_subfolders={"5"},
            )
        assert len(keys) == 1
        assert "/5/" in keys[0]


# ── upload_log_to_s3 — never writes to production ────────────────────
class TestUploadLogSafety:

    def test_skips_upload_when_log_bucket_empty(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        log.write_text('{"s3_key": "test"}\n')
        import config as cfg
        original = cfg.LOG_BUCKET
        cfg.LOG_BUCKET = ""
        try:
            s3_loader.upload_log_to_s3(str(log))
        finally:
            cfg.LOG_BUCKET = original
        out = capsys.readouterr().out
        assert "local only" in out

    def test_refuses_to_write_to_production_bucket(self, tmp_path, capsys):
        log = tmp_path / "test.jsonl"
        log.write_text('{"s3_key": "test"}\n')
        import config as cfg
        original_log = cfg.LOG_BUCKET
        cfg.LOG_BUCKET = cfg.S3_BUCKET    # same as production — should refuse
        try:
            s3_loader.upload_log_to_s3(str(log))
        finally:
            cfg.LOG_BUCKET = original_log
        out = capsys.readouterr().out
        assert "Skipping" in out or "protect" in out


# ── load_image ────────────────────────────────────────────────────────
class TestLoadImage:

    def test_returns_correct_types(self):
        jpeg_bytes = _make_jpeg_bytes()
        mock_resp  = {"Body": MagicMock(read=MagicMock(return_value=jpeg_bytes))}
        with patch.object(s3_loader.s3, "get_object", return_value=mock_resp):
            pil, b64, mime = s3_loader.load_image(_key())
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"
        assert mime == "image/jpeg"
        assert base64.b64decode(b64) == jpeg_bytes

    def test_png_mime_type(self):
        buf = BytesIO()
        Image.new("RGB", (50, 50)).save(buf, format="PNG")
        raw = buf.getvalue()
        mock_resp = {"Body": MagicMock(read=MagicMock(return_value=raw))}
        with patch.object(s3_loader.s3, "get_object", return_value=mock_resp):
            _, _, mime = s3_loader.load_image("frames/car.png")
        assert mime == "image/png"
