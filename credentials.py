"""
credentials.py — Resolve the right boto3 Session for a given S3 bucket.

Lookup order for each bucket:
  1. .env vars  BUCKET_KEY_<BUCKET>  /  BUCKET_SECRET_<BUCKET>  (direct credentials)
  2. BUCKET_SECRET_ID_<BUCKET>  →  fetch that exact secret from Secrets Manager
  3. Auto-derived name {SECRET_PREFIX}{bucket}  →  fetch from Secrets Manager
  4. Default session from the environment / aws configure

For cases 2 and 3, if a secret name is explicitly configured but the fetch fails
(wrong secret name, missing IAM permission, bad secret format), an error is raised
immediately rather than silently falling back to the wrong credentials.
"""

import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

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


def _session_from_env(bucket: str) -> boto3.Session | None:
    """
    Build a session from BUCKET_KEY_<BUCKET> / BUCKET_SECRET_<BUCKET> env vars.
    Returns None if the vars are not set.
    """
    prefix = _bucket_env_key(bucket)
    key_id = os.environ.get(f"BUCKET_KEY_{prefix}")
    secret = os.environ.get(f"BUCKET_SECRET_{prefix}")
    if not (key_id and secret):
        return None
    logger.info("[credentials] Using .env credentials for bucket: %s", bucket)
    return boto3.Session(
        aws_access_key_id     = key_id,
        aws_secret_access_key = secret,
        region_name           = config.AWS_REGION,
    )


def _fetch_secret(secret_id: str, *, raise_on_error: bool) -> boto3.Session | None:
    """
    Fetch credentials from Secrets Manager by exact secret name or ARN.

    Args:
        secret_id:      Secret name or ARN to fetch.
        raise_on_error: If True, raise RuntimeError on any failure instead of
                        returning None. Use this when the secret name was
                        explicitly configured — a silent fallback would just
                        cause a confusing access-denied error on the S3 call.
    """
    try:
        sm = _default().client("secretsmanager", region_name=config.AWS_REGION)
        creds = json.loads(sm.get_secret_value(SecretId=secret_id)["SecretString"])
        logger.info("[credentials] Loaded Secrets Manager secret: %s", secret_id)
        # Support both AWS-style keys (AccessKeyId) and snake_case (aws_access_key_id)
        key_id = creds.get("AccessKeyId") or creds["aws_access_key_id"]
        secret = creds.get("SecretAccessKey") or creds["aws_secret_access_key"]
        return boto3.Session(
            aws_access_key_id     = key_id,
            aws_secret_access_key = secret,
            aws_session_token     = creds.get("SessionToken") or creds.get("aws_session_token"),
            region_name           = creds.get("region", config.AWS_REGION),
        )

    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg  = f"[credentials] Secrets Manager error fetching '{secret_id}': {code} — {e}"
        if raise_on_error:
            raise RuntimeError(msg) from e
        if code != "ResourceNotFoundException":
            logger.warning("%s — falling back to default credentials", msg)
        return None

    except (KeyError, json.JSONDecodeError) as e:
        msg = (
            f"[credentials] Secret '{secret_id}' has unexpected format: {e}\n"
            f"  Expected JSON with keys: AccessKeyId + SecretAccessKey  "
            f"or  aws_access_key_id + aws_secret_access_key"
        )
        if raise_on_error:
            raise RuntimeError(msg) from e
        logger.warning("%s — falling back to default credentials", msg)
        return None


def _session_from_secrets_manager(bucket: str) -> boto3.Session | None:
    """
    Look up Secrets Manager credentials for a bucket.

    If BUCKET_SECRET_ID_<BUCKET> is set, fetches that exact secret and raises
    on any failure (wrong name, missing permission, bad format).

    Otherwise tries the auto-derived name {SECRET_PREFIX}{bucket} with silent
    fallback — no secret just means the bucket uses default credentials.
    """
    prefix   = _bucket_env_key(bucket)
    explicit = os.environ.get(f"BUCKET_SECRET_ID_{prefix}")
    if explicit:
        logger.debug("[credentials] Using explicit secret '%s' for bucket: %s", explicit, bucket)
        return _fetch_secret(explicit, raise_on_error=True)
    return _fetch_secret(f"{config.SECRET_PREFIX}{bucket}", raise_on_error=False)


def get_session(bucket: str) -> boto3.Session:
    """
    Return a boto3 Session for the given bucket.

    Raises RuntimeError if an explicit credential config exists but fails.
    """
    if bucket in _session_cache:
        return _session_cache[bucket]

    session = (
        _session_from_env(bucket)
        or _session_from_secrets_manager(bucket)
        or _default()
    )

    if session is _default():
        logger.info("[credentials] Using default credentials for bucket: %s", bucket)

    _session_cache[bucket] = session
    return session
