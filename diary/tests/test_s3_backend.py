"""S3 backend contract tests against a local HTTP server (no external services).

Ports the WebDAV suite's patterns (test_webdav_llm.py): a minimal fake server,
conditional-write conflict tests, and CorpusStore-level replay/crash drills.
"""
from __future__ import annotations

import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import pytest

from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from agent.s3_storage import S3CorpusBackend


class FakeS3State:
    def __init__(self):
        self.lock = threading.Lock()
        self.objects = {}  # "bucket/key" -> {"body": bytes, "version": int}
        self.fail_next_puts = 0  # crash-drill control: emit 500 instead of storing


STATE = FakeS3State()


class FakeS3Handler(BaseHTTPRequestHandler):
    """Minimal S3-ish server: GET/PUT/HEAD objects, honors If-Match/If-None-Match."""

    server_version = "FakeS3/1.0"

    def log_message(self, *args):  # silence
        pass

    def _respond(self, code, body=b"", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _obj(self):
        """Percent-decoded bucket/key, matching real S3 server behavior."""
        parts = unquote(urlparse(self.path).path.lstrip("/"))
        return parts if "/" in parts else None

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        if "list-type" in query:
            self._do_list(query)
            return
        key = self._obj()
        if key is None:
            self._respond(404)
            return
        with STATE.lock:
            obj = STATE.objects.get(key)
        if obj is None:
            self._respond(404)
            return
        self._respond(200, obj["body"], {"ETag": f'"v{obj["version"]}"'})

    def do_HEAD(self):
        key = self._obj()
        with STATE.lock:
            obj = STATE.objects.get(key) if key else None
        if obj is None:
            self._respond(404)
        else:
            self._respond(200, b"", {"ETag": f'"v{obj["version"]}"'})

    def do_DELETE(self):
        with STATE.lock:
            STATE.objects.pop(self._obj(), None)
        self._respond(204)

    def do_PUT(self):
        key = self._obj()
        if key is None:
            self._respond(400)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        with STATE.lock:
            if STATE.fail_next_puts > 0:
                STATE.fail_next_puts -= 1
                self._respond(500)
                return
            obj = STATE.objects.get(key)
            if_match = self.headers.get("If-Match")
            if_none_match = self.headers.get("If-None-Match")
            if if_match is not None and (obj is None or f'"v{obj["version"]}"' != if_match):
                self._respond(412)
                return
            if if_match is None and if_none_match == "*" and obj is not None:
                self._respond(412)
                return
            version = (obj["version"] + 1) if obj else 1
            STATE.objects[key] = {"body": body, "version": version}
            self._respond(200, b"", {"ETag": f'"v{version}"'})

    def _do_list(self, query):
        prefix = query.get("prefix", [""])[0]
        delimiter = query.get("delimiter", [""])[0]
        bucket = urlparse(self.path).path.lstrip("/").split("/", 1)[0]
        with STATE.lock:
            keys = [k[len(bucket) + 1:] for k in STATE.objects if k.startswith(bucket + "/")]
        contents, common = [], set()
        for key in keys:
            if prefix and not key.startswith(prefix):
                continue
            if delimiter and delimiter in key[len(prefix):]:
                common.add(prefix + key[len(prefix):].split(delimiter)[0] + delimiter)
            else:
                contents.append(key)
        xml = ['<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>']
        for key in sorted(contents):
            version = STATE.objects[f"{bucket}/{key}"]["version"]
            xml.append(f"<Contents><Key>{key}</Key><ETag>&quot;v{version}&quot;</ETag>"
                       f"<LastModified>2026-09-06T00:00:00Z</LastModified></Contents>")
        for cp in sorted(common):
            xml.append(f"<CommonPrefixes><Prefix>{cp}</Prefix></CommonPrefixes>")
        xml.append("</ListBucketResult>")
        self._respond(200, "".join(xml).encode(), {"Content-Type": "application/xml"})


@pytest.fixture
def s3_url():
    STATE.objects.clear()
    STATE.fail_next_puts = 0
    server = HTTPServer(("127.0.0.1", 0), FakeS3Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def make_backend(s3_url) -> S3CorpusBackend:
    return S3CorpusBackend(
        endpoint_url=s3_url, bucket="diary", access_key="test", secret_key="test-secret"
    )


# ---------------- object-level contract (ported from the WebDAV suite) ----------------


def test_get_returns_etag_and_missing_is_none(s3_url):
    backend = make_backend(s3_url)
    assert backend.get_text("notes/a.md") == (None, None)
    ok, _, _ = backend.put("notes/a.md", b"hello")
    assert ok
    text, etag = backend.get_text("notes/a.md")
    assert text == "hello" and etag == '"v1"'


def test_put_with_stale_etag_conflicts_then_fresh_succeeds(s3_url):
    backend = make_backend(s3_url)
    backend.put("file.md", b"one")
    text, etag = backend.get_text("file.md")
    # another writer bumps the object behind our backs
    STATE.objects["diary/file.md"]["version"] += 100
    ok, new_etag, status = backend.put("file.md", b"stale write", if_match=etag, max_retries=1)
    assert not ok and status == 412
    ok, new_etag, status = backend.put("file.md", b"fresh write", if_match=STATE.objects["diary/file.md"]["version"] and f'"v{STATE.objects["diary/file.md"]["version"]}"')
    assert ok and status == 200
    assert backend.get_text("file.md")[0] == "fresh write"


def test_put_create_only_loses_race_with_412(s3_url):
    backend = make_backend(s3_url)
    ok, _, status = backend.put("new.md", b"first", if_none_match="*")
    assert ok and status == 200
    ok, _, status = backend.put("new.md", b"blind create", if_none_match="*", max_retries=1)
    assert not ok and status == 412


def test_exists_head_semantics(s3_url):
    backend = make_backend(s3_url)
    assert not backend.exists("x.md")
    backend.put("x.md", b"data")
    assert backend.exists("x.md")


def test_list_dir_returns_direct_children_only(s3_url):
    backend = make_backend(s3_url)
    backend.put("2026-09.md", b"month")
    backend.put("Entries/2026/September/September 4, 2026.md", b"day")
    entries = backend.list_dir("")
    # S3 has no real directories: "Entries/" surfaces as a CommonPrefix.
    assert [(e["name"], e["is_dir"]) for e in entries] == [
        ("2026-09.md", False),
        ("Entries", True),
    ]
    month_entries = backend.list_dir("Entries/2026/September")
    assert [e["name"] for e in month_entries] == ["September 4, 2026.md"]
    assert month_entries[0]["etag"] == '"v1"'


# ---------------- store-level durability (port of journal-replay drills) ----------------


def make_store(s3_url, tmp_path, name="s3.db", layout="monthly"):
    backend = make_backend(s3_url)
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


def test_journal_replay_with_s3_backend(s3_url, tmp_path):
    store = make_store(s3_url, tmp_path)
    store.log_exchange(date(2026, 9, 3), "S3", "hello", "saved")
    store.apply_pending()
    text, _ = store.read_month(date(2026, 9, 3))
    assert text.count("**Me:** hello") == 1
    assert "**Assistant:** saved" in text
    assert store.journal.pending_count() == 0


def test_mid_write_failure_then_restart_loses_nothing(s3_url, tmp_path):
    """Crash drill: PUT fails mid-apply (server 500), process 'dies', replay on
    restart must produce exactly one copy of the exchange."""
    store = make_store(s3_url, tmp_path, name="crash.db")

    class HalfDead(S3CorpusBackend):
        """Same object; PUTs fail at the server while the journal is armed."""
        def put(self, *args, **kwargs):
            return False, None, 500

    dead = HalfDead(
        endpoint_url="http://unused", bucket="diary", access_key="t", secret_key="t"
    )
    dead._client = store.backend._client  # reuse the live client for gets
    real = store.backend
    store.backend = dead
    store.log_exchange(date(2026, 9, 3), "Crash", "durable words", "assistant words")
    # PUTs failed, so nothing was stored but the journal entry is pending
    # (exchange + the INDEX.md month-registration entry it also enqueues):
    assert real.get_text("2026-09.md")[0] is None
    assert store.journal.pending_count() == 2

    # "Restart": fresh store over the same journal db + a healthy backend.
    recovered = make_store(s3_url, tmp_path, name="crash.db")
    recovered.apply_pending()
    assert store.journal.pending_count() == 0
    text, _ = recovered.read_month(date(2026, 9, 3))
    assert text.count("**Me:** durable words") == 1
    assert text.count("durable words") == 1  # no duplicate from replay


def test_daily_layout_writes_and_lists_through_fake(s3_url, tmp_path):
    store = make_store(s3_url, tmp_path, name="daily.db", layout="daily")
    store.log_exchange(date(2026, 8, 31), "Evening", "August note", "Saved")
    store.log_exchange(date(2026, 9, 4), "Later", "September note", "Saved")
    assert [m["id"] for m in store.list_months()] == ["2026-08", "2026-09"]
    assert "September note" in store.read_month_text(2026, 9)
    assert "August note" not in store.read_month_text(2026, 9)
    assert "September note" in store.get_day_text(date(2026, 9, 4))


def test_rejects_store_ignoring_conditional_headers(monkeypatch):
    import httpx
    backend = S3CorpusBackend('https://example.invalid', 'bucket')
    calls = []
    def request(method, key, **kwargs):
        calls.append((method, key))
        return httpx.Response(204 if method == 'DELETE' else 200, request=httpx.Request(method, 'https://example.invalid'))
    monkeypatch.setattr(backend, '_request', request)
    with pytest.raises(RuntimeError, match='atomic conditional'):
        backend.put('real-diary.md', b'precious content')
    assert all('.cowork-probes/' in key for _, key in calls)
    assert calls[-1][0] == 'DELETE'
    backend.close()
