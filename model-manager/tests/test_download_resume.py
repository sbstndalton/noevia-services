"""#799: a resumed single-stream download must never append a full body to a partial, must check
the 206 Content-Range offset, and must not install a file of the wrong size. HTTP is mocked."""
import asyncio

import httpx
import pytest

from app import downloader, hf

PAYLOAD = b"synthetic-gguf-bytes-0123456789"   # 31 bytes, below PARALLEL_MIN_SIZE
REAL_CLIENT = httpx.AsyncClient
if_ranges: list = []   # If-Range sent with each ranged request of the last run


VALIDATOR = '"etag-v1"'


def _run(monkeypatch, tmp_path, respond, partial: bytes | None = None, total: int = 0,
         validator: str | None = VALIDATOR):
    seen = []
    if_ranges.clear()

    def wrapped(req):
        seen.append((req.method, req.headers.get("range")))
        if req.headers.get("range"):
            if_ranges.append(req.headers.get("if-range"))
        return respond(req)

    monkeypatch.setattr(downloader.httpx, "AsyncClient",
                        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(wrapped), **kw))
    monkeypatch.setattr(hf, "get_token", lambda: None)
    monkeypatch.setattr(downloader.db, "record_download", lambda **kw: None)
    dest = tmp_path / "model.gguf"
    temp = tmp_path / "model.gguf.download"
    if partial is not None:
        temp.write_bytes(partial)
        if validator:
            (tmp_path / "model.gguf.download.validator").write_text(validator)
    job = downloader.DownloadJob("t", "fixture", "model.gguf", "https://example.test/model.gguf", dest, temp, total)
    asyncio.run(downloader.DownloadManager()._run(job))
    return job, seen


def _head(req, size=len(PAYLOAD)):
    # No accept-ranges: forces the single-stream path that resumes partials.
    return httpx.Response(200, headers={"content-length": str(size)})


def test_server_ignoring_range_restarts_instead_of_appending(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        return httpx.Response(200, content=PAYLOAD)          # ignores the Range header
    job, seen = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10])
    assert ("GET", "bytes=10-") in seen
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD
    assert job.downloaded_bytes == len(PAYLOAD)
    assert not job.temp_path.exists()


def test_valid_206_resume_appends_the_rest(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        start = int(req.headers["range"].removeprefix("bytes=").rstrip("-"))
        return httpx.Response(206, content=PAYLOAD[start:],
                              headers={"content-range": f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"})
    job, seen = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10])
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD
    assert seen[-1] == ("GET", "bytes=10-")
    assert if_ranges == [VALIDATOR], "a resume names the upload it continues"


@pytest.mark.parametrize("content_range", ["bytes 0-30/31", "bytes 12-30/31", "", "garbage"])
def test_206_for_the_wrong_offset_installs_nothing(monkeypatch, tmp_path, content_range):
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        headers = {"content-range": content_range} if content_range else {}
        return httpx.Response(206, content=PAYLOAD[10:], headers=headers)
    job, _ = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10])
    assert job.status == "error" and "Content-Range" in job.error
    assert not job.dest_path.exists()
    assert job.temp_path.read_bytes() == PAYLOAD[:10], "the good partial is kept for the next attempt"


def test_206_for_a_changed_size_discards_the_partial_and_starts_over(monkeypatch, tmp_path):
    # HEAD says 31 bytes, the resume answers with a 50-byte file: never splice, fetch afresh.
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        if req.headers.get("range"):
            return httpx.Response(206, content=b"x" * 40, headers={"content-range": "bytes 10-49/50"})
        return httpx.Response(200, content=PAYLOAD)
    job, seen = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10])
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD
    assert [r for m, r in seen if m == "GET"] == ["bytes=10-", None]


def test_short_body_is_not_installed_and_partial_is_kept(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        return httpx.Response(200, content=PAYLOAD[:20])     # connection ended early
    job, _ = _run(monkeypatch, tmp_path, respond)
    assert job.status == "error" and "incomplete" in job.error and "nothing was installed" in job.error
    assert not job.dest_path.exists()
    assert job.temp_path.read_bytes() == PAYLOAD[:20]


def test_oversized_body_is_not_installed_and_is_discarded(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        return httpx.Response(200, content=PAYLOAD + b"extra")
    job, _ = _run(monkeypatch, tmp_path, respond)
    assert job.status == "error" and "expected 31" in job.error
    assert not job.dest_path.exists() and not job.temp_path.exists()


def test_size_known_only_from_the_request_is_still_enforced(monkeypatch, tmp_path):
    # HEAD fails; the size Hugging Face listed when the job was queued is what is checked.
    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(200, content=PAYLOAD[:20])
    job, _ = _run(monkeypatch, tmp_path, respond, total=len(PAYLOAD))
    assert job.status == "error" and not job.dest_path.exists()


def test_partial_at_full_size_is_fetched_again_not_trusted(monkeypatch, tmp_path):
    # A file already the expected size cannot be told apart from a preallocated parallel file.
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        assert "range" not in req.headers
        return httpx.Response(200, content=PAYLOAD)
    job, _ = _run(monkeypatch, tmp_path, respond, partial=b"\0" * len(PAYLOAD))
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD


def test_416_is_never_taken_as_complete(monkeypatch, tmp_path):
    # Review of #827: a preallocated, mostly-zero file the size of the model, a HEAD that fails
    # (so the size is unknown up front) and a 416 for the resume must not install anything.
    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(405)
        return httpx.Response(416, headers={"content-range": f"bytes */{len(PAYLOAD)}"})
    job, _ = _run(monkeypatch, tmp_path, respond, partial=b"\0" * len(PAYLOAD))
    assert job.status == "error"
    assert not job.dest_path.exists()


def test_416_restart_fetches_the_real_bytes(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(405)
        if req.headers.get("range"):
            return httpx.Response(416, headers={"content-range": f"bytes */{len(PAYLOAD)}"})
        return httpx.Response(200, content=PAYLOAD)
    job, seen = _run(monkeypatch, tmp_path, respond, partial=b"\0" * len(PAYLOAD))
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD
    assert [r for m, r in seen if m == "GET"] == [f"bytes={len(PAYLOAD)}-", None]


def test_failed_parallel_attempt_leaves_no_preallocated_file(monkeypatch, tmp_path):
    # The parallel path preallocates the whole file; it must be gone after a failure so a
    # single-stream retry can never resume (or install) it.
    attempt = {"n": 0}

    def respond(req):
        if req.method == "HEAD":
            if attempt["n"] == 0:
                return httpx.Response(200, headers={"content-length": str(len(PAYLOAD)), "accept-ranges": "bytes"})
            return httpx.Response(405)
        if attempt["n"] == 0:
            return httpx.Response(503)
        if req.headers.get("range"):
            return httpx.Response(416, headers={"content-range": f"bytes */{len(PAYLOAD)}"})
        return httpx.Response(200, content=PAYLOAD)
    monkeypatch.setattr(downloader, "PARALLEL_MIN_SIZE", 1)
    monkeypatch.setattr(downloader, "PARALLEL_CHUNKS", 2)
    first, _ = _run(monkeypatch, tmp_path, respond)
    assert first.status == "error" and not first.temp_path.exists() and not first.dest_path.exists()
    attempt["n"] = 1
    second, seen = _run(monkeypatch, tmp_path, respond)
    assert second.status == "done", second.error
    assert second.dest_path.read_bytes() == PAYLOAD
    assert [r for m, r in seen if m == "GET"] == [None]


def test_partial_of_a_replaced_upload_is_not_extended(monkeypatch, tmp_path):
    # Review of #827: a stale .download of an older upload with the same name. The stored
    # validator goes out as If-Range; the server sees the file changed and answers 200.
    new = b"a-newer-upload-of-the-same-name-and-longer"

    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(405)
        if req.headers.get("if-range") == '"etag-v2"':
            return httpx.Response(206, content=new[10:], headers={"content-range": f"bytes 10-{len(new) - 1}/{len(new)}"})
        return httpx.Response(200, content=new, headers={"etag": '"etag-v2"'})
    job, seen = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10])
    assert ("GET", "bytes=10-") in seen
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == new


def test_head_reporting_a_different_validator_skips_the_resume(monkeypatch, tmp_path):
    new = b"replacement-bytes-0123456789-xyz"

    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(new)), "etag": '"etag-v2"'})
        assert "range" not in req.headers
        return httpx.Response(200, content=new, headers={"etag": '"etag-v2"'})
    job, _ = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10])
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == new


def test_partial_without_a_stored_validator_is_not_resumed(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        assert "range" not in req.headers
        return httpx.Response(200, content=PAYLOAD)
    job, _ = _run(monkeypatch, tmp_path, respond, partial=b"from-an-unknown-upload", validator=None)
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD


def test_validator_is_stored_while_partial_and_removed_once_installed(monkeypatch, tmp_path):
    def short(req):
        if req.method == "HEAD":
            return _head(req)
        return httpx.Response(200, content=PAYLOAD[:12], headers={"etag": '"etag-v1"', "content-length": "12"})
    job, _ = _run(monkeypatch, tmp_path, short)
    sidecar = tmp_path / "model.gguf.download.validator"
    assert job.status == "error" and sidecar.read_text() == '"etag-v1"\t31'

    def rest(req):
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(PAYLOAD)), "etag": '"etag-v1"'})
        assert req.headers["range"] == "bytes=12-" and req.headers["if-range"] == '"etag-v1"'
        return httpx.Response(206, content=PAYLOAD[12:], headers={"content-range": f"bytes 12-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"})
    job, _ = _run(monkeypatch, tmp_path, rest, partial=None)
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD and not sidecar.exists()


def test_weak_etag_falls_back_to_last_modified():
    assert downloader._validator({"etag": 'W/"x"', "last-modified": "Mon, 05 Oct 2026 10:00:00 GMT"}) == "Mon, 05 Oct 2026 10:00:00 GMT"
    assert downloader._validator({"etag": '"x"', "last-modified": "y"}) == '"x"'
    assert downloader._validator({}) == ""


def test_416_for_a_longer_partial_starts_over(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(405)                        # size unknown up front
        if req.headers.get("range"):
            return httpx.Response(416, headers={"content-range": "bytes */20"})
        return httpx.Response(200, content=PAYLOAD[:20])
    job, seen = _run(monkeypatch, tmp_path, respond, partial=b"stale-partial-that-is-too-long-for-the-file")
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD[:20]
    assert [r for m, r in seen if m == "GET"] == ["bytes=43-", None]


def test_partial_larger_than_the_known_size_is_not_resumed(monkeypatch, tmp_path):
    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        assert "range" not in req.headers
        return httpx.Response(200, content=PAYLOAD)
    job, _ = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD + b"junk")
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == PAYLOAD


# ---------- second review of #827 ----------

def test_new_validator_is_written_only_after_the_partial_is_truncated(monkeypatch, tmp_path):
    sizes = []
    real = downloader._write_validator

    def spy(job, value, total):
        sizes.append(job.temp_path.stat().st_size)
        real(job, value, total)
    monkeypatch.setattr(downloader, "_write_validator", spy)

    def respond(req):
        if req.method == "HEAD":
            return _head(req)
        return httpx.Response(200, content=PAYLOAD, headers={"etag": '"etag-v2"'})
    job, _ = _run(monkeypatch, tmp_path, respond, partial=b"old-upload-bytes", validator=None)
    assert job.status == "done", job.error
    assert sizes == [0], "the new validator appears only beside an emptied file"


def test_crash_before_truncating_never_leaves_old_bytes_beside_a_new_validator(monkeypatch, tmp_path):
    real_open = open

    def crashing_open(path, mode="r", *a, **kw):
        if mode == "wb" and str(path).endswith(".download"):
            raise OSError("simulated crash before the truncate")
        return real_open(path, mode, *a, **kw)
    monkeypatch.setattr(downloader, "open", crashing_open, raising=False)

    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": "40", "etag": '"etag-v2"'})
        return httpx.Response(200, content=b"n" * 40, headers={"etag": '"etag-v2"'})
    # An old partial and its validator; HEAD now reports a different upload, so it restarts.
    job, _ = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10])
    assert job.status == "error"
    assert job.temp_path.read_bytes() == PAYLOAD[:10]
    sidecar = tmp_path / "model.gguf.download.validator"
    assert not (sidecar.exists() and "etag-v2" in sidecar.read_text()), "old bytes must never carry the new validator"


def test_replaced_upload_of_another_size_is_not_spliced_when_if_range_is_ignored(monkeypatch, tmp_path):
    # The server honours Range but ignores If-Range, and HEAD fails, so only the size the partial
    # was recorded with can tell the uploads apart.
    new = b"a-replacement-upload-with-a-different-size!"

    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(405)
        if req.headers.get("range"):
            start = int(req.headers["range"].removeprefix("bytes=").rstrip("-"))
            return httpx.Response(206, content=new[start:], headers={"content-range": f"bytes {start}-{len(new) - 1}/{len(new)}"})
        return httpx.Response(200, content=new, headers={"etag": '"etag-v2"'})
    job, seen = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10], validator=f'"etag-v1"\t{len(PAYLOAD)}')
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == new
    assert [r for m, r in seen if m == "GET"] == ["bytes=10-", None]


def test_head_size_disagreeing_with_the_recorded_size_skips_the_resume(monkeypatch, tmp_path):
    new = b"x" * 44

    def respond(req):
        if req.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(new))})
        assert "range" not in req.headers
        return httpx.Response(200, content=new)
    job, _ = _run(monkeypatch, tmp_path, respond, partial=PAYLOAD[:10], validator=f'"etag-v1"\t{len(PAYLOAD)}')
    assert job.status == "done", job.error
    assert job.dest_path.read_bytes() == new


def test_validator_file_format():
    class J:
        temp_path = None
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        j = J()
        j.temp_path = Path(d) / "m.gguf.download"
        downloader._write_validator(j, '"e"', 31)
        assert downloader._read_validator(j) == ('"e"', 31)
        downloader._validator_path(j).write_text('"legacy"')
        assert downloader._read_validator(j) == ('"legacy"', 0)
        downloader._write_validator(j, "bad\tvalue", 31)
        assert downloader._read_validator(j) == ("", 0)
