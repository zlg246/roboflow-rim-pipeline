"""Tests that config loads correctly and required values are present."""
import pytest
import config


def test_required_api_keys_present():
    assert config.ROBOFLOW_API_KEY,   "ROBOFLOW_API_KEY missing from .env"
    assert config.ROBOFLOW_WORKSPACE, "ROBOFLOW_WORKSPACE missing from .env"


def test_production_bucket_is_set():
    assert config.S3_BUCKET, "S3_BUCKET must be non-empty"


def test_rim_s3_prefix():
    assert config.RIM_S3_ROOT_PREFIX == "wheel_scanner/"


def test_rim_roboflow_project_default():
    assert config.RIM_ROBOFLOW_PROJECT == "test_project-e4bw5"


def test_log_bucket_differs_from_prod():
    """Log bucket must not equal production bucket."""
    if config.LOG_BUCKET:
        assert config.LOG_BUCKET != config.S3_BUCKET, (
            "LOG_BUCKET must be different from S3_BUCKET (production)"
        )


def test_yolo_threshold_range():
    assert 0.0 < config.YOLO_CONF_THRESHOLD < 1.0


def test_upload_sleep_non_negative():
    assert config.UPLOAD_SLEEP >= 0.0


def test_default_date_range_is_monday_to_friday():
    from datetime import date
    d_from = config.default_date_from()
    d_to   = config.default_date_to()
    assert d_from.weekday() == 0, "default date_from should be Monday"
    assert d_to.weekday()   == 4, "default date_to should be Friday"
    assert d_to > d_from
