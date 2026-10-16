"""model-files is the only listing implementation (#964; noevia#1071 retired MODEL_FILES_IMPL and
removed the Python reference). It FAILS CLOSED (ModelFilesError, answered as 502) on every
failure, and the app refuses to start without the binary. A fake binary (a small script) stands
in for model-files; the real one is covered by test_model_files_differential.py. Synthetic
listings only."""
from __future__ import annotations

import json
import stat
import sys
import textwrap

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, hf, model_files
from app.main import app

LISTING = [
    {"type": "file", "path": "m-Q4_K_M-00001-of-00002.gguf", "lfs": {"size": 5, "oid": "x"},
     "lastCommit": {"title": "x" * 10_000}, "securityFileStatus": {"status": "safe"}},
    {"type": "file", "path": "m-Q4_K_M-00002-of-00002.gguf", "size": 6},
    {"type": "directory", "path": "sub"},
    {"type": "file", "path": "README.md", "size": 1},
]


@pytest.fixture(autouse=True)
def _fresh_log_state():
    model_files._LOGGED.clear()
    yield
    model_files._LOGGED.clear()


def _fake_bin(tmp_path, body: str) -> str:
    """An executable script standing in for model-files: it records its argv and stdin, then
    runs `body` with `listing` = the parsed stdin."""
    body_file = tmp_path / "fake_model_files.py"
    body_file.write_text(textwrap.dedent(f"""\
        import json, sys, time
        raw = sys.stdin.read()
        open({str(tmp_path / "calls.log")!r}, "a").write(json.dumps([sys.argv[1:], raw]) + "\\n")
        listing = json.loads(raw)
    """) + textwrap.dedent(body))
    # A sh wrapper rather than a #!python line: the interpreter path may contain spaces.
    script = tmp_path / "fake-model-files"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body_file}" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _calls(tmp_path) -> list:
    log = tmp_path / "calls.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _use_rust(monkeypatch, binary: str) -> None:
    monkeypatch.setattr(config.settings, "model_files_bin", binary)


# A fake that answers the GGUF entries of a listing in model-files' shape, marked so tests can tell.
_ECHO = """
    import re
    files = []
    for e in listing:
        if e.get("type") != "file" or not e["path"].endswith(".gguf"):
            continue
        m = re.search(r"-(\\d{5})-of-(\\d{5})\\.gguf$", e["path"])
        files.append({"path": e["path"], "size": (e.get("lfs") or {}).get("size") or e.get("size") or 0,
                      "quant": "FROM-RUST", "shard_base": e["path"][:m.start()] if m else e["path"],
                      "shard_index": int(m.group(1)) if m else None,
                      "shard_total": int(m.group(2)) if m else None})
    print(json.dumps({"files": files}))
"""


def test_there_is_no_python_fallback_and_the_retired_switch_is_ignored(tmp_path, monkeypatch):
    # noevia#1071: the reference and the selector are gone; a leftover MODEL_FILES_IMPL=python in
    # the environment still goes through the binary.
    for gone in ("files_from_tree_py", "impl_choice", "IMPLS"):
        assert not hasattr(model_files, gone), gone
    assert "model_files_impl" not in config.Settings.model_fields
    monkeypatch.setenv("MODEL_FILES_IMPL", "python")
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO))
    assert {f["quant"] for f in model_files.files_from_tree(LISTING)} == {"FROM-RUST"}
    assert len(_calls(tmp_path)) == 1


def test_startup_refuses_without_the_binary_before_any_other_hook(tmp_path, monkeypatch):
    # noevia#1071: fail closed at startup, not on the first listing or models.ini write.
    from app import db, main
    hooks = []
    monkeypatch.setattr(db, "init", lambda: hooks.append("db"))
    for missing in (str(tmp_path / "nowhere" / "model-files"), "model-files-not-on-path"):
        _use_rust(monkeypatch, missing)
        with pytest.raises(model_files.ModelFilesError, match="missing_binary"):
            with TestClient(app):
                pass
        with pytest.raises(model_files.ModelFilesError, match="missing_binary"):
            main._startup()
    assert hooks == []
    not_executable = tmp_path / "model-files"
    not_executable.write_text("#!/bin/sh\n")
    _use_rust(monkeypatch, str(not_executable))
    with pytest.raises(model_files.ModelFilesError, match="missing_binary"):
        model_files.require_binary()
    assert hooks == []


def test_startup_proceeds_with_the_binary(tmp_path, monkeypatch):
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO))
    model_files.require_binary()
    with TestClient(app) as client:
        assert client.get("/api/v1/sections").status_code == 200
    assert _calls(tmp_path) == [], "the startup check does not run the binary"


def test_rust_result_is_used_and_input_is_projected(tmp_path, monkeypatch):
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO))
    files = model_files.files_from_tree(LISTING)
    assert [f["quant"] for f in files] == ["FROM-RUST", "FROM-RUST"]
    assert [f["shard_index"] for f in files] == [1, 2]
    (argv, raw), = _calls(tmp_path)
    assert argv == ["tree"]
    sent = json.loads(raw)
    # Only the fields the function reads cross the boundary; the 10 KB commit title does not.
    assert sent[0] == {"type": "file", "path": "m-Q4_K_M-00001-of-00002.gguf", "lfs": {"size": 5}}
    assert "lastCommit" not in raw and len(raw) < 1000


@pytest.mark.parametrize("reason,body", [
    ("rejected", "sys.stderr.write('bad'); sys.exit(1)"),
    ("rejected", "print(json.dumps({'files': []})); sys.exit(3)"),
    ("malformed_output", "print('not json')"),
    ("malformed_output", "print(json.dumps([1, 2]))"),
    ("malformed_output", "print(json.dumps({'files': [{'path': 'a'}]}))"),
    ("malformed_output", "print(json.dumps({'files': [{'path': 1, 'size': 0, 'quant': None, 'shard_base': 'a',"
                         " 'shard_index': None, 'shard_total': None}]}))"),
    ("malformed_output", "print(json.dumps({'files': [{'path': 'a', 'size': True, 'quant': None,"
                         " 'shard_base': 'a', 'shard_index': None, 'shard_total': None}] * 1}))"),
    ("malformed_output", "print(json.dumps({'files': [{'path': 'a', 'size': 1, 'quant': None,"
                         " 'shard_base': 'a', 'shard_index': None, 'shard_total': None}] * 99}))"),
    ("timeout", "time.sleep(30)"),
])
def test_rust_failures_fail_closed(tmp_path, monkeypatch, reason, body):
    _use_rust(monkeypatch, _fake_bin(tmp_path, body))
    monkeypatch.setattr(model_files, "MODEL_FILES_TIMEOUT_S", 1.0)
    with pytest.raises(model_files.ModelFilesError) as exc:
        model_files.files_from_tree(LISTING)
    assert reason in str(exc.value)


def test_missing_binary_fails_closed(tmp_path, monkeypatch):
    _use_rust(monkeypatch, str(tmp_path / "nowhere" / "model-files"))
    with pytest.raises(model_files.ModelFilesError, match="missing_binary"):
        model_files.files_from_tree(LISTING)
    _use_rust(monkeypatch, "model-files-not-on-path")
    with pytest.raises(model_files.ModelFilesError, match="missing_binary"):
        model_files.files_from_tree(LISTING)


def test_oversized_and_unencodable_input_fail_closed(tmp_path, monkeypatch):
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO))
    monkeypatch.setattr(model_files, "MODEL_FILES_STDIN_CAP", 100)
    with pytest.raises(model_files.ModelFilesError, match="input_too_large"):
        model_files.files_from_tree(LISTING)
    monkeypatch.setattr(model_files, "MODEL_FILES_STDIN_CAP", 16 * 1024 * 1024)
    with pytest.raises(model_files.ModelFilesError, match="unencodable_input"):
        model_files.files_from_tree([{"type": "file", "path": "a.gguf", "size": object()}])
    assert _calls(tmp_path) == []


def _hub(monkeypatch, listing):
    """repo_detail against a fake hub that serves `listing` as one tree page."""
    real = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=listing)

    monkeypatch.setattr(hf.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_repo_detail_goes_through_the_binary(tmp_path, monkeypatch):
    from app import db
    db.init()  # repo_detail reads the saved hub token
    _hub(monkeypatch, LISTING)
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO))
    rs = await hf.repo_detail("synthetic/repo-GGUF")
    assert [(f.path, f.size, f.shard_index) for f in rs.files] == [
        ("m-Q4_K_M-00001-of-00002.gguf", 5, 1), ("m-Q4_K_M-00002-of-00002.gguf", 6, 2)]
    assert {f.quant for f in rs.files} == {"FROM-RUST"}


def test_api_answers_502_when_rust_refuses(tmp_path, monkeypatch):
    _hub(monkeypatch, LISTING)
    from app import api
    queued = []
    monkeypatch.setattr(api.manager, "enqueue", lambda **kw: queued.append(kw))
    _use_rust(monkeypatch, _fake_bin(tmp_path, "sys.exit(1)"))
    with TestClient(app) as client:
        r = client.post("/api/v1/downloads", json={"repo": "synthetic/repo-GGUF",
                                                   "path": "m-Q4_K_M-00001-of-00002.gguf"})
    assert r.status_code == 502
    assert "could not be checked" in r.json()["detail"]
    assert queued == []


def test_child_gets_a_minimal_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_LOADER_TOKEN", "synthetic-secret-not-for-children")
    monkeypatch.setenv("HF_TOKEN", "hf_synthetic")
    out = tmp_path / "env.json"
    _use_rust(monkeypatch, _fake_bin(tmp_path, f"""
        import os
        open({str(out)!r}, "w").write(json.dumps(dict(os.environ)))
        print(json.dumps({{"files": []}}))
    """))
    model_files.files_from_tree([])
    env = json.loads(out.read_text())
    assert "PATH" in env
    assert "MODEL_LOADER_TOKEN" not in env and "HF_TOKEN" not in env
    # Whatever the OS, sh or interpreter adds itself, nothing of the service's environment leaks.
    assert not set(env) - {"PATH", "__CF_USER_TEXT_ENCODING", "LC_CTYPE", "PWD", "SHLVL", "_"}


@pytest.mark.asyncio
async def test_a_slow_binary_does_not_block_the_event_loop(tmp_path, monkeypatch):
    import asyncio
    import time
    from app import db
    db.init()
    other = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    _hub(monkeypatch, LISTING)
    _use_rust(monkeypatch, _fake_bin(tmp_path, "time.sleep(2)\n" + textwrap.dedent(_ECHO)))
    slow = asyncio.create_task(hf.repo_detail("synthetic/repo-GGUF"))
    await asyncio.sleep(0.2)  # the listing is now waiting on the binary
    t0 = time.monotonic()
    r = await other.get("/api/v1/sections")
    assert r.status_code == 200 and time.monotonic() - t0 < 2.0
    assert not slow.done()
    detail = await slow
    assert {f.quant for f in detail.files} == {"FROM-RUST"}
    await other.aclose()
