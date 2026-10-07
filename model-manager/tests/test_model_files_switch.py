"""MODEL_FILES_IMPL switch (#964): python stays the default and never spawns anything; rust runs
the model-files binary and FAILS CLOSED (ModelFilesError, answered as 502) on every failure.
A fake binary (a small script) stands in for model-files; the real one is covered by
test_model_files_differential.py. Synthetic listings only."""
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
def _fresh_switch_state(monkeypatch):
    model_files._LOGGED.clear()
    monkeypatch.setattr(config.settings, "model_files_impl", "python")
    monkeypatch.setattr(config.settings, "model_files_bin", "model-files")
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
    monkeypatch.setattr(config.settings, "model_files_impl", "rust")
    monkeypatch.setattr(config.settings, "model_files_bin", binary)


# A fake that answers with the Python reference's own result, marked so tests can tell.
_ECHO_PYTHON = f"""
    sys.path.insert(0, {str(__import__("pathlib").Path(__file__).resolve().parents[1])!r})
    from app.model_files import files_from_tree_py
    files = files_from_tree_py(listing)
    for f in files:
        f["quant"] = "FROM-RUST"
    print(json.dumps({{"files": files}}))
"""


def test_default_is_python_and_never_spawns(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "model_files_bin", _fake_bin(tmp_path, _ECHO_PYTHON))
    assert model_files.impl_choice() == "python"
    assert model_files.files_from_tree(LISTING) == model_files.files_from_tree_py(LISTING)
    assert _calls(tmp_path) == []


def test_python_path_keeps_python_errors(monkeypatch):
    with pytest.raises(ValueError):
        model_files.files_from_tree([{"type": "file", "path": "a.gguf", "size": "nope"}])
    with pytest.raises(AttributeError):
        model_files.files_from_tree([5])


def test_invalid_setting_means_python_with_one_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config.settings, "model_files_impl", "fortran")
    monkeypatch.setattr(config.settings, "model_files_bin", _fake_bin(tmp_path, _ECHO_PYTHON))
    with caplog.at_level("WARNING"):
        model_files.files_from_tree(LISTING)
        model_files.files_from_tree(LISTING)
    assert model_files.impl_choice() == "python"
    assert sum("MODEL_FILES_IMPL" in r.message for r in caplog.records) == 1
    assert _calls(tmp_path) == []


def test_rust_result_is_used_and_input_is_projected(tmp_path, monkeypatch):
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO_PYTHON))
    files = model_files.files_from_tree(LISTING)
    assert [f["quant"] for f in files] == ["FROM-RUST", "FROM-RUST"]
    assert [f["shard_index"] for f in files] == [1, 2]
    (argv, raw), = _calls(tmp_path)
    assert argv == ["tree"]
    sent = json.loads(raw)
    # Only the fields the function reads cross the boundary; the 10 KB commit title does not.
    assert sent[0] == {"type": "file", "path": "m-Q4_K_M-00001-of-00002.gguf", "lfs": {"size": 5}}
    assert "lastCommit" not in raw and len(raw) < 1000


def test_projection_is_faithful():
    odd = [5, "x", None, [], {"type": "file", "path": "a.gguf", "lfs": "str"},
           {"type": "file", "path": "b.gguf", "lfs": {"oid": "x"}, "size": 3},
           {"type": "file", "path": "c.gguf", "lfs": [], "size": 4}, {}]
    for e in odd:
        a = _try(lambda: model_files.files_from_tree_py([e]))
        b = _try(lambda: model_files.files_from_tree_py([model_files._project(e)]))
        assert a == b, e


def _try(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


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
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO_PYTHON))
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
async def test_repo_detail_goes_through_the_switch(tmp_path, monkeypatch):
    from app import db
    db.init()  # repo_detail reads the saved hub token
    _hub(monkeypatch, LISTING)
    py = await hf.repo_detail("synthetic/repo-GGUF")
    assert [f.path for f in py.files] == ["m-Q4_K_M-00001-of-00002.gguf", "m-Q4_K_M-00002-of-00002.gguf"]
    _use_rust(monkeypatch, _fake_bin(tmp_path, _ECHO_PYTHON))
    rs = await hf.repo_detail("synthetic/repo-GGUF")
    assert [(f.path, f.size, f.shard_base) for f in rs.files] == [(f.path, f.size, f.shard_base) for f in py.files]
    assert {f.quant for f in rs.files} == {"FROM-RUST"}


def test_api_answers_502_when_rust_refuses(tmp_path, monkeypatch):
    _hub(monkeypatch, LISTING)
    _use_rust(monkeypatch, _fake_bin(tmp_path, "sys.exit(1)"))
    from app import api
    queued = []
    monkeypatch.setattr(api.manager, "enqueue", lambda **kw: queued.append(kw))
    with TestClient(app) as client:
        r = client.post("/api/v1/downloads", json={"repo": "synthetic/repo-GGUF",
                                                   "path": "m-Q4_K_M-00001-of-00002.gguf"})
    assert r.status_code == 502
    assert "could not be checked" in r.json()["detail"]
    assert queued == []
