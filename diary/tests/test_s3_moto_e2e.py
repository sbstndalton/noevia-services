"""End-to-end S3-backend verification against a real S3 implementation.

Uses moto in *server mode*: a real HTTP S3 server on loopback, so the backend's
hand-rolled SigV4 signing, conditional headers, and listing XML are exercised
exactly as they will be against MinIO/B2/AWS — not against an in-process mock.
"""
from __future__ import annotations

import threading
from datetime import date

import boto3
import pytest
from moto.server import ThreadedMotoServer

from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from agent.s3_storage import S3CorpusBackend


@pytest.fixture
def moto_s3():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    admin = boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id="testing", aws_secret_access_key="testing",
        region_name="us-east-1",
    )
    admin.create_bucket(Bucket="diary-e2e")
    yield endpoint
    server.stop()


def make_backend(endpoint: str) -> S3CorpusBackend:
    return S3CorpusBackend(
        endpoint_url=endpoint, bucket="diary-e2e",
        access_key="testing", secret_key="testing",
    )


def make_store(endpoint: str, tmp_path, name="moto.db", layout="daily"):
    backend = make_backend(endpoint)
    cfg = Config({
        "corpus": {
            "root": "", "monthly_prefix": "", "index_file": "INDEX.md",
            "entry_layout": layout, "index_enabled": layout == "monthly",
            # Match the shipped config.yaml: the CorpusStore default template uses
            # {month:02d}, which list_months' filename regex does not parse.
            "month_file_template": "{year}-{month02}.md",
        }
    })
    return CorpusStore(cfg, backend, Journal(tmp_path / name))


def test_object_roundtrip_and_conditional_semantics_on_real_s3(moto_s3):
    backend = make_backend(moto_s3)
    ok, etag, status = backend.put("hello.md", b"first body")
    assert ok and status == 200 and etag
    assert backend.get_text("hello.md") == ("first body", etag)

    # Overwrite with a stale ETag must 412...
    ok, _, status = backend.put("hello.md", b"stale", if_match='"deadbeef"')
    assert not ok and status == 412
    # ...and with the current one must succeed.
    ok, etag2, status = backend.put("hello.md", b"second body", if_match=etag)
    assert ok and status == 200 and etag2 != etag

    # Create-only on an existing object must 412 (lost create race).
    ok, _, status = backend.put("hello.md", b"blind", if_none_match="*", max_retries=1)
    assert not ok and status == 412

    assert backend.exists("hello.md")
    assert not backend.exists("never-written.md")


def test_full_diary_flow_daily_layout_on_real_s3(moto_s3, tmp_path):
    store = make_store(moto_s3, tmp_path)
    store.log_exchange(date(2026, 9, 4), "Morning", "moto hello", "moto saved")
    store.log_exchange(date(2026, 9, 6), "Evening", "second day", "also saved")

    assert [m["id"] for m in store.list_months()] == ["2026-09"]
    assert "moto hello" in store.read_month_text(2026, 9)
    assert "second day" in store.read_month_text(2026, 9)
    assert "xid" not in store.read_month_text(2026, 9)
    assert "moto hello" in store.get_day_text(date(2026, 9, 4))
    assert store.journal.pending_count() == 0

    # Replayed journal entries must not duplicate content (crash-replay discipline).
    store.apply_pending()
    text, _ = store.read_month(date(2026, 9, 4))
    assert text.count("moto hello") == 1


def test_monthly_layout_with_index_on_real_s3(moto_s3, tmp_path):
    store = make_store(moto_s3, tmp_path, name="moto-monthly.db", layout="monthly")
    store.log_exchange(date(2026, 8, 31), "Evening", "august words", "assistant")
    store.log_exchange(date(2026, 9, 1), "Morning", "september words", "assistant")
    assert [m["id"] for m in store.list_months()] == ["2026-08", "2026-09"]
    assert "august words" in store.read_month_text(2026, 8)
    assert "september words" in store.read_month_text(2026, 9)
