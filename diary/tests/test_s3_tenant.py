"""Per-tenant S3 storage: X-Cowork-Storage header -> configured backend.

The web server forwards each user's encrypted storage connection as a base64
JSON header. This test exercises the real tenant-state resolution against a
real HTTP S3 server (moto server mode): the header must produce an
S3CorpusBackend with the right endpoint/bucket/prefix, and a logged exchange
must land as an object inside the user's bucket prefix.
"""
from __future__ import annotations

import base64
import json
import uuid
from datetime import date

import boto3
import pytest
from starlette.requests import Request

import agent.app as appmod
from agent.config import Config
from agent.s3_storage import S3CorpusBackend
from tests.test_s3_moto_e2e import moto_s3  # noqa: F401 — real HTTP S3 fixture


def _tenant_request(user_id: str, storage: dict | None) -> Request:
    headers = [(b"x-cowork-user-id", user_id.encode())]
    if storage is not None:
        encoded = base64.urlsafe_b64encode(json.dumps(storage).encode()).decode().rstrip("=")
        headers.append((b"x-cowork-storage", encoded.encode()))
    return Request({"type": "http", "headers": headers})


@pytest.fixture
def tenant_cfg(tmp_path, monkeypatch, moto_s3):
    cfg = Config({
        "corpus": {
            "backend": "local", "root": "", "monthly_prefix": "",
            "entry_layout": "daily", "entries_prefix": "Entries", "index_enabled": False,
            "local": {"root": str(tmp_path / "default-corpus")},
            "s3": {"endpoint_url": "", "bucket": "", "access_key": "", "secret_key": "",
                   "region": "us-east-1", "prefix": "", "timeout_s": 60},
        },
        "retrieval": {"db_path": str(tmp_path / "idx.db")},
        "llm": {"base_url": "http://stub", "api_key": "", "chat_model": "m", "embed_model": "e",
                "aux": {"base_url": "http://stub", "api_key": "", "model": "a"}},
        "ui": {"host": "127.0.0.1", "port": 8010},
    })
    monkeypatch.setattr(appmod, "_base_cfg", cfg)
    appmod._tenant_states.clear()
    yield cfg
    appmod._tenant_states.clear()


def test_s3_storage_header_builds_s3_backend_and_lands_object(moto_s3, tenant_cfg):
    user_id = str(uuid.uuid4())
    admin = boto3.client("s3", endpoint_url=moto_s3, aws_access_key_id="t", aws_secret_access_key="t",
                         region_name="us-east-1")
    admin.create_bucket(Bucket="tenant-bucket")

    storage = {"kind": "s3", "baseUrl": moto_s3, "bucket": "tenant-bucket",
               "username": "AKIAIOSFODNN7EXAMPLE", "secret": "secret-here", "corpusRoot": "tenant-a"}
    state = appmod._tenant_state(_tenant_request(user_id, storage))

    assert isinstance(state.backend, S3CorpusBackend)
    assert state.backend.bucket == "tenant-bucket"
    assert state.backend.prefix == "tenant-a"
    assert moto_s3.split("//", 1)[1] in state.backend.base

    state.store.log_exchange(date(2026, 9, 6), "Morning", "tenant hello", "tenant saved")

    obj = admin.get_object(Bucket="tenant-bucket", Key="tenant-a/Entries/2026/September/September 6, 2026.md")
    body = obj["Body"].read().decode()
    assert "**Me:** tenant hello" in body
    assert "xid" not in body.replace("xid:", "").replace("xid-", "")  # markers present but internal

    # Same header again must resolve from the cache, not rebuild state.
    again = appmod._tenant_state(_tenant_request(user_id, storage))
    assert again is state


def test_local_fallback_when_no_storage_header(moto_s3, tenant_cfg):
    from agent.local_storage import LocalCorpusBackend

    state = appmod._tenant_state(_tenant_request(str(uuid.uuid4()), None))
    assert isinstance(state.backend, LocalCorpusBackend)
