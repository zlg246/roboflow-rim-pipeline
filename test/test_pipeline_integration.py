"""
Integration tests — exercises the full pipeline orchestration logic.
No S3, no YOLO model, no Claude API, no Roboflow.
All external dependencies are mocked.
"""
import json
import pytest
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from PIL import Image
from io import BytesIO
import base64
import pipeline


# ── Fixtures / helpers ────────────────────────────────────────────────
YOLO_PRED = [{"x": 0.5, "y": 0.4, "width": 0.2, "height": 0.15,
               "class": "scratch", "confidence": 0.82}]


def _make_b64():
    buf = BytesIO()
    Image.new("RGB", (640, 480)).save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode()

def _mock_load(_key, **kwargs):
    return Image.new("RGB", (640, 480)), _make_b64(), "image/jpeg"

def _claude_result(decision, with_box=True):
    corrections = []
    if with_box and decision in ("scratch", "other"):
        corrections = [{
            "class": "scratch",
            "x": 0.5, "y": 0.4, "width": 0.2, "height": 0.15, "note": ""
        }]
    return {
        "decision":    decision,
        "confidence":  0.92,
        "reasoning":   "Mock.",
        "corrections": corrections,
    }

REAL_KEY = (
    "prior_condition/"
    "SCANNER_A_0009bf_2026-03-16_14-10-49/"
    "3/frame_001.jpg"
)


# ── Upload logic ──────────────────────────────────────────────────────
@pytest.mark.parametrize("decision,should_upload", [
    ("scratch", True),
    ("other",   True),
    ("null",    False),
])
def test_upload_triggered_only_for_damage_decisions(
    decision, should_upload, tmp_path, monkeypatch
):
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "log.jsonl")  # type: ignore

    with patch("pipeline.list_image_keys",  return_value=[REAL_KEY]), \
         patch("pipeline.load_image",       side_effect=_mock_load), \
         patch("pipeline.run_inference",    return_value=YOLO_PRED), \
         patch("pipeline.verify_image",     return_value=_claude_result(decision)), \
         patch("pipeline.upload",           return_value=True) as mock_upload, \
         patch("pipeline.upload_log_to_s3"):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
            limit=1,
        )

    assert mock_upload.called == should_upload


# ── Dry run ───────────────────────────────────────────────────────────
def test_dry_run_never_calls_roboflow(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "log.jsonl")  # type: ignore

    with patch("pipeline.list_image_keys",  return_value=[REAL_KEY]), \
         patch("pipeline.load_image",       side_effect=_mock_load), \
         patch("pipeline.run_inference",    return_value=YOLO_PRED), \
         patch("pipeline.verify_image",     return_value=_claude_result("scratch")), \
         patch("pipeline.upload",           return_value=True) as mock_upload, \
         patch("pipeline.upload_log_to_s3"):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
            dry_run=True,
            limit=1,
        )

    assert not mock_upload.called


# ── Resume / deduplication ────────────────────────────────────────────
def test_already_processed_keys_are_skipped(tmp_path, monkeypatch):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps({"s3_key": REAL_KEY}) + "\n")
    monkeypatch.setattr(pipeline, "LOG_PATH", log)  # type: ignore

    new_key = REAL_KEY.replace("frame_001", "frame_002")

    with patch("pipeline.list_image_keys",  return_value=[REAL_KEY, new_key]), \
         patch("pipeline.load_image",       side_effect=_mock_load), \
         patch("pipeline.run_inference",    return_value=YOLO_PRED), \
         patch("pipeline.verify_image",     return_value=_claude_result("scratch")), \
         patch("pipeline.upload",           return_value=True) as mock_upload, \
         patch("pipeline.upload_log_to_s3"):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
        )

    # Only new_key processed — REAL_KEY was already in the log
    assert mock_upload.call_count == 1


# ── Error handling ────────────────────────────────────────────────────
def test_json_decode_error_logged_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "log.jsonl")  # type: ignore

    with patch("pipeline.list_image_keys",  return_value=[REAL_KEY]), \
         patch("pipeline.load_image",       side_effect=_mock_load), \
         patch("pipeline.run_inference",    return_value=YOLO_PRED), \
         patch("pipeline.verify_image",     side_effect=json.JSONDecodeError("bad", "", 0)), \
         patch("pipeline.upload_log_to_s3"):

        pipeline.run(date_from=date(2026, 3, 16), date_to=date(2026, 3, 16), limit=1)

    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert "JSONDecodeError" in lines[0]["error"]


def test_s3_error_logged_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "log.jsonl")  # type: ignore

    with patch("pipeline.list_image_keys",  return_value=[REAL_KEY]), \
         patch("pipeline.load_image",       side_effect=Exception("S3 NoSuchKey")), \
         patch("pipeline.upload_log_to_s3"):

        pipeline.run(date_from=date(2026, 3, 16), date_to=date(2026, 3, 16), limit=1)

    lines = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert "S3 NoSuchKey" in lines[0]["error"]


def test_nothing_to_process_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "log.jsonl")  # type: ignore

    with patch("pipeline.list_image_keys",  return_value=[]), \
         patch("pipeline.upload_log_to_s3"):

        pipeline.run(date_from=date(2026, 3, 16), date_to=date(2026, 3, 16))

    assert "Nothing" in capsys.readouterr().out


# ── Limit flag ────────────────────────────────────────────────────────
def test_limit_caps_number_of_images_processed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "log.jsonl")  # type: ignore

    keys = [REAL_KEY.replace("frame_001", f"frame_{i:03d}") for i in range(10)]

    with patch("pipeline.list_image_keys",  return_value=keys), \
         patch("pipeline.load_image",       side_effect=_mock_load), \
         patch("pipeline.run_inference",    return_value=[]), \
         patch("pipeline.verify_image",     return_value=_claude_result("null")), \
         patch("pipeline.upload_log_to_s3"):

        pipeline.run(
            date_from=date(2026, 3, 16), date_to=date(2026, 3, 16),
            limit=3,
        )

    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 3


# ── OpenAI skip when no YOLO predictions ─────────────────────────────
def test_openai_skipped_when_no_yolo_predictions(tmp_path, monkeypatch):
    """When YOLO returns no predictions, OpenAI must NOT be called."""
    monkeypatch.setattr(pipeline, 'LOG_PATH', tmp_path / 'log.jsonl')

    with patch('pipeline.list_image_keys',  return_value=[REAL_KEY]),          patch('pipeline.load_image',       side_effect=_mock_load),          patch('pipeline.run_inference',    return_value=[]),                   patch('pipeline.verify_image',     return_value=_claude_result('null')) as mock_verify,          patch('pipeline.upload_log_to_s3'):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
            limit=1,
        )

    mock_verify.assert_not_called()


def test_openai_called_when_yolo_has_predictions(tmp_path, monkeypatch):
    """When YOLO returns predictions, OpenAI must be called."""
    monkeypatch.setattr(pipeline, 'LOG_PATH', tmp_path / 'log.jsonl')

    yolo_preds = [{'x': 0.5, 'y': 0.4, 'width': 0.2, 'height': 0.15,
                   'class': 'scratch', 'confidence': 0.82}]

    with patch('pipeline.list_image_keys',  return_value=[REAL_KEY]),          patch('pipeline.load_image',       side_effect=_mock_load),          patch('pipeline.run_inference',    return_value=yolo_preds),          patch('pipeline.verify_image',     return_value=_claude_result('scratch')) as mock_verify,          patch('pipeline.upload',           return_value=True),          patch('pipeline.upload_log_to_s3'):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
            limit=1,
        )

    mock_verify.assert_called_once()


def test_no_yolo_prediction_logged_as_null(tmp_path, monkeypatch):
    """Images with no YOLO predictions must be logged with decision=null."""
    monkeypatch.setattr(pipeline, 'LOG_PATH', tmp_path / 'log.jsonl')

    with patch('pipeline.list_image_keys',  return_value=[REAL_KEY]),          patch('pipeline.load_image',       side_effect=_mock_load),          patch('pipeline.run_inference',    return_value=[]),          patch('pipeline.upload_log_to_s3'):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
            limit=1,
        )

    lines = [json.loads(l) for l in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert lines[0]['decision'] == 'null'
    assert lines[0]['yolo_count'] == 0


# ── Bucket parameter ─────────────────────────────────────────────────
def test_batch_name_format(tmp_path, monkeypatch):
    """Batch name format: <dealership>_<date_from>_<date_to>
    prod- prefix stripped, hyphens replaced with underscores.
    e.g. prod-castle-hill-toyota + 2026-03-16 to 2026-03-20
      -> castle_hill_toyota_2026-03-16_2026-03-20
    """
    monkeypatch.setattr(pipeline, 'LOG_PATH', tmp_path / 'log.jsonl')

    with patch('pipeline.list_image_keys',  return_value=[REAL_KEY]),          patch('pipeline.load_image',       side_effect=_mock_load),          patch('pipeline.run_inference',    return_value=[]),          patch('pipeline.verify_image',     return_value=_claude_result('null')),          patch('pipeline.upload_log_to_s3'):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 20),
            bucket='prod-castle-hill-toyota',
            limit=1,
        )

    lines = [json.loads(l) for l in (tmp_path / 'log.jsonl').read_text().splitlines()]
    assert lines[0]['batch'] == 'castle_hill_toyota_2026-03-16_2026-03-20'


def test_batch_name_strips_prod_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'LOG_PATH', tmp_path / 'log.jsonl')

    with patch('pipeline.list_image_keys',  return_value=[REAL_KEY]),          patch('pipeline.load_image',       side_effect=_mock_load),          patch('pipeline.run_inference',    return_value=[]),          patch('pipeline.verify_image',     return_value=_claude_result('null')),          patch('pipeline.upload_log_to_s3'):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
            bucket='prod-chatswood-toyota',
            limit=1,
        )

    lines = [json.loads(l) for l in (tmp_path / 'log.jsonl').read_text().splitlines()]
    batch  = lines[0]['batch']
    assert not batch.startswith('prod-'), f"batch should not start with 'prod-': {batch}"
    assert 'chatswood_toyota' in batch
    assert '2026-03-16' in batch


def test_default_bucket_used_when_not_specified(tmp_path, monkeypatch):
    """When --bucket not passed, config.S3_BUCKET is used and prod- is stripped."""
    monkeypatch.setattr(pipeline, 'LOG_PATH', tmp_path / 'log.jsonl')

    with patch('pipeline.list_image_keys',  return_value=[REAL_KEY]),          patch('pipeline.load_image',       side_effect=_mock_load),          patch('pipeline.run_inference',    return_value=[]),          patch('pipeline.verify_image',     return_value=_claude_result('null')),          patch('pipeline.upload_log_to_s3'):

        pipeline.run(
            date_from=date(2026, 3, 16),
            date_to=date(2026, 3, 16),
            limit=1,
        )

    lines  = [json.loads(l) for l in (tmp_path / 'log.jsonl').read_text().splitlines()]
    batch  = lines[0]['batch']
    import config
    # prod- prefix stripped, hyphens → underscores, dates appended
    expected_dealership = config.S3_BUCKET.removeprefix('prod-').replace('-', '_')
    assert expected_dealership in batch
    assert '2026-03-16' in batch


# ── CLI date parser ───────────────────────────────────────────────────
class TestParseDateArg:
    def setup_method(self):
        from pipeline import _parse_date_arg
        self.parse = _parse_date_arg

    def test_iso_format(self):
        assert self.parse("2026-03-16") == date(2026, 3, 16)

    def test_today(self):
        assert self.parse("today") == date.today()

    def test_last_friday(self):
        result = self.parse("last-friday")
        assert result.weekday() == 4
        assert result < date.today()

    def test_monday_keyword(self):
        assert self.parse("monday").weekday() == 0

    def test_friday_keyword(self):
        assert self.parse("friday").weekday() == 4

    def test_invalid_raises(self):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            self.parse("not-a-date")
