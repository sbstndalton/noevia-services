"""GGUF_PARSER switch (#909): rust runs the gguf-meta binary, python stays the default, and
every Rust failure falls back to the Python parser. A fake binary (a small script) stands in
for gguf-meta; the real binary is covered by test_gguf_meta_differential.py. Synthetic only."""
from __future__ import annotations

import json
import logging
import os
import stat
import sys
import textwrap

import pytest

from conftest import _gguf
from app import config, gguf_meta

TINY = {"general.architecture": "llama", "general.name": "tiny", "llama.context_length": 4096,
        "tokenizer.chat_template": "{{ messages }}"}


@pytest.fixture(autouse=True)
def _fresh_switch_state(monkeypatch):
    gguf_meta._LOGGED.clear()
    gguf_meta._RUST_CACHE.clear()
    monkeypatch.setattr(config.settings, "gguf_parser", "python")
    monkeypatch.setattr(config.settings, "gguf_meta_bin", "gguf-meta")
    yield
    gguf_meta._LOGGED.clear()
    gguf_meta._RUST_CACHE.clear()


@pytest.fixture
def model(tmp_path):
    p = tmp_path / "tiny.gguf"
    p.write_bytes(_gguf(TINY))
    return p


def _fake_bin(tmp_path, body: str) -> str:
    """An executable script standing in for gguf-meta. It appends one line to calls.log per run
    (so tests can count spawns) and then runs `body` with `path` = its single argument."""
    script = tmp_path / "fake-gguf-meta"
    log = tmp_path / "calls.log"
    script.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""\
        import json, sys, time
        open({str(log)!r}, "a").write(repr(sys.argv[1:]) + "\\n")
        path = sys.argv[1]
    """) + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _calls(tmp_path) -> list[str]:
    log = tmp_path / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _use_rust(monkeypatch, binary: str) -> None:
    monkeypatch.setattr(config.settings, "gguf_parser", "rust")
    monkeypatch.setattr(config.settings, "gguf_meta_bin", binary)


def _python(path):
    return gguf_meta.summarize(gguf_meta.read_raw(path))


def _rust_summary(name: str) -> str:
    """Body printing a well-formed summary whose general.name marks it as the Rust result."""
    return f"""
        s = {{"arch": "llama", "general": {{"name": {name!r}}}, "model": {{"arch": "llama",
             "rope_freq_base": float("nan")}}, "tokenizer": {{}}, "chat_template": None}}
        print(json.dumps(s))
    """


def test_default_is_python_and_never_spawns(tmp_path, model, monkeypatch):
    monkeypatch.setattr(config.settings, "gguf_meta_bin", _fake_bin(tmp_path, _rust_summary("rust")))
    assert gguf_meta.parser_choice() == "python"
    assert gguf_meta.summarize_path(model) == _python(model)
    assert _calls(tmp_path) == []


def test_rust_success_is_used_sanitised_and_cached(tmp_path, model, monkeypatch):
    _use_rust(monkeypatch, _fake_bin(tmp_path, _rust_summary("from-rust")))
    first = gguf_meta.summarize_path(model)
    assert first["general"]["name"] == "from-rust"
    assert first["model"]["rope_freq_base"] is None  # NaN in the output is nulled like #901
    first["general"]["name"] = "mutated by a caller"
    again = gguf_meta.summarize_path(model)
    assert again["general"]["name"] == "from-rust"  # callers get copies, the cache is intact
    calls = _calls(tmp_path)
    assert len(calls) == 1  # cached on (path, mtime, size)
    assert calls[0] == repr([str(model.absolute())])  # the path is exactly one argv element


def test_path_with_shell_metacharacters_is_one_argument(tmp_path, monkeypatch):
    p = tmp_path / "a b; touch pwned $(id).gguf"
    p.write_bytes(_gguf(TINY))
    _use_rust(monkeypatch, _fake_bin(tmp_path, _rust_summary("from-rust")))
    assert gguf_meta.summarize_path(p)["general"]["name"] == "from-rust"
    assert _calls(tmp_path) == [repr([str(p)])]
    assert not (tmp_path / "pwned").exists()


@pytest.mark.parametrize("reason,body", [
    ("nonzero_exit", "sys.stderr.write('boom'); sys.exit(1)"),
    ("bad_json", "print('{not json')"),
    ("bad_json", "print(json.dumps([1, 2, 3]))"),
    ("bad_json", "sys.stdout.buffer.write(b'\\xff\\xfe')"),
])
def test_rust_failures_fall_back_to_python(tmp_path, model, monkeypatch, caplog, reason, body):
    _use_rust(monkeypatch, _fake_bin(tmp_path, body))
    with caplog.at_level(logging.WARNING, logger=gguf_meta.__name__):
        assert gguf_meta.summarize_path(model) == _python(model)
    assert f"rust:{reason}" in gguf_meta._LOGGED
    assert any(reason in r.getMessage() for r in caplog.records)


def test_timeout_kills_the_binary_and_falls_back(tmp_path, model, monkeypatch):
    monkeypatch.setattr(gguf_meta, "GGUF_META_TIMEOUT_S", 0.5)
    _use_rust(monkeypatch, _fake_bin(tmp_path, "time.sleep(30)"))
    import time
    t0 = time.monotonic()
    assert gguf_meta.summarize_path(model) == _python(model)
    assert time.monotonic() - t0 < 10
    assert "rust:timeout" in gguf_meta._LOGGED


def test_oversized_output_is_capped_and_falls_back(tmp_path, model, monkeypatch):
    monkeypatch.setattr(gguf_meta, "GGUF_META_STDOUT_CAP", 1024)
    _use_rust(monkeypatch, _fake_bin(tmp_path, "sys.stdout.write('x' * 10_000_000); sys.stdout.flush()"))
    assert gguf_meta.summarize_path(model) == _python(model)
    assert "rust:output_too_large" in gguf_meta._LOGGED


@pytest.mark.parametrize("binary", ["no-such-gguf-meta-binary", "/nonexistent/dir/gguf-meta"])
def test_missing_binary_falls_back(model, monkeypatch, binary):
    _use_rust(monkeypatch, binary)
    assert gguf_meta.summarize_path(model) == _python(model)
    assert "rust:missing_binary" in gguf_meta._LOGGED


def test_non_executable_binary_counts_as_missing(tmp_path, model, monkeypatch):
    path = _fake_bin(tmp_path, _rust_summary("from-rust"))
    os.chmod(path, 0o644)
    _use_rust(monkeypatch, path)
    assert gguf_meta.summarize_path(model) == _python(model)
    assert "rust:missing_binary" in gguf_meta._LOGGED


def test_failure_is_logged_once_and_not_respawned_for_the_same_file(tmp_path, model, monkeypatch, caplog):
    _use_rust(monkeypatch, _fake_bin(tmp_path, "sys.exit(3)"))
    other = tmp_path / "other.gguf"
    other.write_bytes(_gguf({**TINY, "general.name": "other"}))
    with caplog.at_level(logging.WARNING, logger=gguf_meta.__name__):
        for _ in range(3):
            assert gguf_meta.summarize_path(model) == _python(model)
        assert gguf_meta.summarize_path(other) == _python(other)
    assert len(_calls(tmp_path)) == 2  # once per file version, not per call
    assert sum("nonzero_exit" in r.getMessage() for r in caplog.records) == 1


def test_python_errors_survive_the_fallback(tmp_path, monkeypatch):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    _use_rust(monkeypatch, _fake_bin(tmp_path, "sys.exit(1)"))
    with pytest.raises(gguf_meta.GgufMetaError):
        gguf_meta.summarize_path(bad)
    with pytest.raises(FileNotFoundError):
        gguf_meta.summarize_path(tmp_path / "missing.gguf")


@pytest.mark.parametrize("value", ["RUST", " rust "])
def test_setting_is_case_and_space_insensitive(tmp_path, model, monkeypatch, value):
    _use_rust(monkeypatch, _fake_bin(tmp_path, _rust_summary("from-rust")))
    monkeypatch.setattr(config.settings, "gguf_parser", value)
    assert gguf_meta.parser_choice() == "rust"


def test_invalid_setting_means_python_with_one_warning(tmp_path, model, monkeypatch, caplog):
    monkeypatch.setattr(config.settings, "gguf_meta_bin", _fake_bin(tmp_path, _rust_summary("rust")))
    monkeypatch.setattr(config.settings, "gguf_parser", "golang")
    with caplog.at_level(logging.WARNING, logger=gguf_meta.__name__):
        for _ in range(3):
            assert gguf_meta.summarize_path(model) == _python(model)
    assert _calls(tmp_path) == []
    assert sum("GGUF_PARSER" in r.getMessage() for r in caplog.records) == 1


def test_summarize_bytes_uses_rust_and_cleans_its_temp_file(tmp_path, monkeypatch):
    _use_rust(monkeypatch, _fake_bin(tmp_path, _rust_summary("from-rust")))
    buf = _gguf(TINY)
    assert gguf_meta.summarize_bytes(buf)["general"]["name"] == "from-rust"
    (arg,) = eval(_calls(tmp_path)[0])  # noqa: S307 - our own repr() of argv
    assert not os.path.exists(arg)
    monkeypatch.setattr(config.settings, "gguf_meta_bin", _fake_bin(tmp_path, "sys.exit(1)"))
    assert gguf_meta.summarize_bytes(buf) == gguf_meta.summarize(gguf_meta.read_raw_bytes(buf))


def test_env_var_reaches_the_settings(monkeypatch):
    monkeypatch.setenv("GGUF_PARSER", "rust")
    monkeypatch.setenv("GGUF_META_BIN", "/opt/x/gguf-meta")
    s = config.Settings()
    assert s.gguf_parser == "rust" and s.gguf_meta_bin == "/opt/x/gguf-meta"
    monkeypatch.delenv("GGUF_PARSER")
    assert config.Settings().gguf_parser == "python"


def test_model_detail_endpoint_goes_through_the_switch(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    _use_rust(monkeypatch, _fake_bin(tmp_path, _rust_summary("from-rust")))
    with TestClient(app) as c:
        r = c.get("/api/v1/models/detail?key=tiny/tiny-Q4_K_M.gguf")
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["general"]["name"] == "from-rust"
    assert json.dumps(r.json(), allow_nan=False)
