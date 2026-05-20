"""Tests that config loads correctly and required values are present."""
import pytest
import config


def test_required_api_keys_present():
    assert config.OPENAI_API_KEY,  "OPENAI_API_KEY missing from .env"
    assert config.ROBOFLOW_API_KEY,   "ROBOFLOW_API_KEY missing from .env"
    assert config.ROBOFLOW_WORKSPACE, "ROBOFLOW_WORKSPACE missing from .env"


def test_production_bucket_name():
    assert config.S3_BUCKET == "prod-castle-hill-toyota"


def test_target_subfolders_default():
    assert config.S3_TARGET_SUBFOLDERS == {"3", "7"}


def test_log_bucket_differs_from_prod():
    """Log bucket must not equal production bucket."""
    if config.LOG_BUCKET:
        assert config.LOG_BUCKET != config.S3_BUCKET, (
            "LOG_BUCKET must be different from S3_BUCKET (production)"
        )


def test_upload_decisions_are_valid():
    valid = {"correct", "partial", "missed", "other", "null", "bad_quality"}
    assert config.UPLOAD_DECISIONS.issubset(valid)
    assert config.SKIP_DECISIONS.issubset(valid)
    assert config.UPLOAD_DECISIONS.isdisjoint(config.SKIP_DECISIONS)


def test_yolo_threshold_range():
    assert 0.0 < config.YOLO_CONF_THRESHOLD < 1.0


def test_default_date_range_is_monday_to_friday():
    from datetime import date
    d_from = config.default_date_from()
    d_to   = config.default_date_to()
    assert d_from.weekday() == 0, "default date_from should be Monday"
    assert d_to.weekday()   == 4, "default date_to should be Friday"
    assert d_to > d_from
