"""
credentials.py — Resolve the right boto3 Session for a given S3 bucket.

Lookup order for each bucket:
  1. BUCKET_KEY_<BUCKET> + BUCKET_SECRET_<BUCKET> in .env  (dedicated credentials)
  2. Default session from the environment / aws configure

<BUCKET> is the bucket name uppercased with hyphens replaced by underscores.
Example: cmt-prod-ap-southeast-2-melton-toyota → CMT_PROD_AP_SOUTHEAST_2_MELTON_TOYOTA

Sessions are cached per bucket after the first lookup.
"""

import logging
import os

import boto3

import config

logger = logging.getLogger(__name__)

_session_cache: dict[str, boto3.Session] = {}
_default_session: boto3.Session | None = None


def _default() -> boto3.Session:
    global _default_session
    if _default_session is None:
        _default_session = boto3.Session()
    return _default_session


def _bucket_env_key(bucket: str) -> str:
    """Convert bucket name to env var prefix, e.g. prod-foo-bar → PROD_FOO_BAR."""
    return bucket.upper().replace("-", "_")


def get_session(bucket: str) -> boto3.Session:
    """
    Return a boto3 Session for the given bucket.

    Uses BUCKET_KEY_<BUCKET> / BUCKET_SECRET_<BUCKET> from .env if set,
    otherwise falls back to the default session from aws configure.
    """
    if bucket in _session_cache:
        return _session_cache[bucket]

    prefix = _bucket_env_key(bucket)
    key_id = os.environ.get(f"BUCKET_KEY_{prefix}")
    secret = os.environ.get(f"BUCKET_SECRET_{prefix}")

    if key_id and secret:
        logger.info("[credentials] Using .env credentials for bucket: %s", bucket)
        session = boto3.Session(
            aws_access_key_id     = key_id,
            aws_secret_access_key = secret,
            region_name           = config.AWS_REGION,
        )
    else:
        logger.info("[credentials] Using default credentials for bucket: %s", bucket)
        session = _default()

    _session_cache[bucket] = session
    return session
