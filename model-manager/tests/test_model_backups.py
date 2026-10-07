"""models.ini recovery copies (#1021): python (default) keeps the behaviour before #1021 except
that a write with noevia-core's `backup: false` hint (#1003) makes no copies; MODEL_FILES_IMPL=rust
lets `model-files backups` decide which copies to make and which old ones to remove, and fails
closed. The shared table (tests/fixtures/model-backups.v1.json, byte-identical to noevia-rs's
crates/model-files/tests/fixtures/) is replayed through the real binary when MODEL_FILES_BIN
points at one (CI builds it at the Dockerfile's NOEVIA_RS_REF). Synthetic files only."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app import config, ini, model_files

FIXTURES = Path(__file__).parent / "fixtures" / "model-backups.v1.json"
BIN = os.environ.get("MODEL_FILES_BIN", "")
needs_bin = pytest.mark.skipif(not BIN, reason="MODEL_FILES_BIN not set: no model-files binary to run")
BASE = "a" * 64


@pytest.fixture(autouse=True)
def _python_by_default(monkeypatch):
    model_files._LOGGED.clear()
    monkeypatch.setattr(config.settings, "model_files_impl", "python")
    monkeypatch.setattr(config.settings, "model_files_bin", "model-files")
    yield
    model_files._LOGGED.clear()


def _dir(tmp_path, revisions: int = 0) -> Path:
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "models.ini").write_text("version = 1\n")
    for i in range(revisions):
        p = d / f"models.ini.noevia-backup-{i:064x}"
        p.write_text(f"old {i}\n")
        os.utime(p, ns=(1_700_000_000_000_000_000 + i, 1_700_000_000_000_000_000 + i))
    for n in ("models.ini.noevia-backup-notes", "other.ini.noevia-backup-" + "b" * 64):
        (d / n).write_text("operator\n")
    return d


def _names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def _revisions(d: Path) -> set[str]:
    return {n for n in _names(d) if n.startswith("models.ini.noevia-backup-") and len(n) == len("models.ini.noevia-backup-") + 64}


def test_python_default_keeps_both_copies_as_before(tmp_path):
    d = _dir(tmp_path, revisions=12)
    ini._replace_file(d / "models.ini", "version = 2\n", base_revision=BASE)
    assert (d / "models.ini").read_text() == "version = 2\n"
    assert len(_revisions(d)) == 13, "python does not prune revision copies"
    assert (d / f"models.ini.noevia-backup-{BASE}").read_text() == "version = 1\n"
    assert any(n.startswith("models.ini.bak-") for n in _names(d))


def test_python_honours_the_hint_with_no_copies(tmp_path):
    d = _dir(tmp_path, revisions=2)
    before = _names(d)
    ini._replace_file(d / "models.ini", "version = 2\n", base_revision=BASE, backup=False)
    assert (d / "models.ini").read_text() == "version = 2\n"
    assert _names(d) == before


def test_api_threads_only_an_explicit_false(monkeypatch):
    seen = []
    monkeypatch.setattr(ini, "write_raw_text", lambda text, base_revision=None, backup=True: seen.append(backup))
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        base = c.get("/api/v1/sections").json()["revision"]
        text = ini.raw_text()
        for hint in (False, True, "false", 0, None):
            assert c.put("/api/v1/models-ini", json={"baseRevision": base, "text": text, "backup": hint}).status_code == 200
        assert c.put("/api/v1/models-ini", json={"baseRevision": base, "text": text}).status_code == 200
    assert seen == [False, True, True, True, True, True]


def _fake(tmp_path, reply: str, code: int = 0) -> str:
    body = tmp_path / "fake.py"
    body.write_text(textwrap.dedent(f"""\
        import sys
        sys.stdin.read()
        sys.stdout.write({reply!r})
        sys.stderr.write("model-files: refused: input" if {code} else "")
        sys.exit({code})
    """))
    script = tmp_path / "fake-model-files"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body}" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _rust(monkeypatch, binary: str) -> None:
    monkeypatch.setattr(config.settings, "model_files_impl", "rust")
    monkeypatch.setattr(config.settings, "model_files_bin", binary)


@pytest.mark.parametrize("reply,code", [
    ("", 1),  # refused
    ("not json", 0),
    ('{"rotating":null,"revision":null,"prune":["models.ini"]}', 0),  # the preset itself
    ('{"rotating":null,"revision":null,"prune":["models.ini.noevia-backup-notes"]}', 0),  # operator copy
    ('{"rotating":null,"revision":null,"prune":["models.ini.bak-unlisted"]}', 0),  # not listed
    ('{"rotating":"models.ini.bak-x","revision":null,"prune":[]}', 0),  # a name it was not given
])
def test_rust_fails_closed_and_touches_nothing(tmp_path, monkeypatch, reply, code):
    d = _dir(tmp_path, revisions=3)
    before = {n: (d / n).read_bytes() for n in _names(d)}
    _rust(monkeypatch, _fake(tmp_path, reply, code))
    with pytest.raises(OSError):
        ini._replace_file(d / "models.ini", "version = 2\n", base_revision=BASE)
    assert {n: (d / n).read_bytes() for n in _names(d)} == before


def test_rust_missing_binary_fails_closed(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    _rust(monkeypatch, str(tmp_path / "nope"))
    with pytest.raises(OSError):
        ini._replace_file(d / "models.ini", "version = 2\n")
    assert (d / "models.ini").read_text() == "version = 1\n"


@needs_bin
def test_rust_bounds_revision_copies_and_keeps_this_writes(tmp_path, monkeypatch):
    d = _dir(tmp_path, revisions=15)
    _rust(monkeypatch, BIN)
    ini._replace_file(d / "models.ini", "version = 2\n", base_revision=BASE)
    revs = _revisions(d)
    assert len(revs) == ini.REVISION_BACKUPS_TO_KEEP
    assert f"models.ini.noevia-backup-{BASE}" in revs
    assert (d / f"models.ini.noevia-backup-{BASE}").read_text() == "version = 1\n"
    # The newest nine old ones stay; operator files are untouched.
    assert {f"models.ini.noevia-backup-{i:064x}" for i in range(6, 15)} <= revs
    assert {"models.ini.noevia-backup-notes", "other.ini.noevia-backup-" + "b" * 64} <= _names(d)
    assert (d / "models.ini").read_text() == "version = 2\n"


@needs_bin
def test_rust_hinted_write_makes_no_copy_but_prunes(tmp_path, monkeypatch):
    d = _dir(tmp_path, revisions=15)
    _rust(monkeypatch, BIN)
    ini._replace_file(d / "models.ini", "version = 2\n", base_revision=BASE, backup=False)
    revs = _revisions(d)
    assert len(revs) == ini.REVISION_BACKUPS_TO_KEEP
    assert f"models.ini.noevia-backup-{BASE}" not in revs
    assert not any(n.startswith("models.ini.bak-") for n in _names(d))
    assert f"models.ini.noevia-backup-{14:064x}" in revs, "the newest pre-tune copy stays"


@needs_bin
def test_rust_refuses_a_bad_base_revision_like_python(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    _rust(monkeypatch, BIN)
    with pytest.raises(ValueError):
        ini._replace_file(d / "models.ini", "version = 2\n", base_revision="nothex")
    assert (d / "models.ini").read_text() == "version = 1\n"


@needs_bin
def test_shared_table_through_the_binary():
    f = json.loads(FIXTURES.read_text())
    assert f["version"] == 1 and f["limits"]["keepRotating"] == ini.BACKUPS_TO_KEEP
    assert len(f["cases"]) >= 180
    for c in f["cases"]:
        text = c["input"] if c["input"] is not None else c["inputBase"] + " " * c["pad"]
        proc = subprocess.run([BIN, "backups"], input=text.encode(), capture_output=True, timeout=30, check=False)
        if "error" in c["expect"]:
            assert proc.returncode == 1 and proc.stdout == b"", c["name"]
            assert proc.stderr.decode().strip() == f"model-files: refused: {c['expect']['error']}", c["name"]
        else:
            assert proc.returncode == 0, (c["name"], proc.stderr)
            assert json.loads(proc.stdout) == c["expect"], c["name"]
