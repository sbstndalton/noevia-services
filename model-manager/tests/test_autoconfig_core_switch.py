"""MODEL_AUTOCONFIG switch: python stays the default and never spawns anything; rust runs the
model-autoconfig binary BESIDE the Python size core, which stays authoritative. The Python plan
is used when Rust agrees exactly, or when Python's is the conservative one (same backend and
placement mode, no larger context, GPU layer count or prompt cache); any other disagreement and
every Rust fault refuses the recommendation. A fake binary (a small script) stands in for
model-autoconfig; the real one is covered by test_autoconfig_core_differential.py. Synthetic
models only - nothing loads a model or starts llama.cpp."""
from __future__ import annotations

import dataclasses
import json
import logging
import random
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from app import autoconfig, autoconfig_core, config

MM = Path(__file__).resolve().parents[1]


def _summary(layers=32, experts=None, native=131072):
    m = {"block_count": layers, "attention_head_count": 32, "embedding_length": 4096,
         "attention_head_count_kv": 8, "context_length": native}
    if experts:
        m["expert_count"] = experts
    return {"arch": "llama", "model": m, "chat_template": "x"}


def _backends(vram=24.0, host=64.0):
    return [{"name": "llama-cuda", "vendor": "cuda", "vram_gb": vram, "gpu_count": 1,
             "host_ram_gb": host, "baseline": {}}]


def _analyze(**over):
    kw = dict(summary=_summary(), file_size=int(4.7 * 2**30), backends=_backends(), n_sessions=1)
    kw.update(over)
    return autoconfig.analyze(**kw)


@pytest.fixture(autouse=True)
def _fresh_switch_state(monkeypatch):
    autoconfig_core._LOGGED.clear()
    monkeypatch.setattr(config.settings, "model_autoconfig", "python")
    monkeypatch.setattr(config.settings, "model_autoconfig_bin", "model-autoconfig")
    yield
    autoconfig_core._LOGGED.clear()


def _fake_bin(tmp_path, body: str) -> str:
    """An executable standing in for model-autoconfig: records argv, stdin and environment, then
    runs `body` with `req` = the parsed request and `plan` = the Python plan for it."""
    body_file = tmp_path / "fake_model_autoconfig.py"
    body_file.write_text(textwrap.dedent(f"""\
        import json, os, sys, time
        sys.path.insert(0, {str(MM)!r})
        from app.autoconfig_core import size_plan
        raw = sys.stdin.read()
        open({str(tmp_path / "calls.log")!r}, "a").write(json.dumps([sys.argv[1:], sorted(os.environ)]) + "\\n")
        req = json.loads(raw)
        plan = size_plan(req)
    """) + textwrap.dedent(body))
    script = tmp_path / "fake-model-autoconfig"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body_file}" "$@"\n')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _calls(tmp_path) -> list:
    log = tmp_path / "calls.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _use_rust(monkeypatch, binary: str) -> None:
    monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
    monkeypatch.setattr(config.settings, "model_autoconfig_bin", binary)


ECHO = "print(json.dumps(plan))\n"


def test_default_is_python_and_never_spawns(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "model_autoconfig_bin", _fake_bin(tmp_path, ECHO))
    assert autoconfig_core.impl_choice() == "python"
    rec = _analyze()
    assert rec.error == "" and rec.recommended_ctx > 0
    assert _calls(tmp_path) == []


def test_invalid_value_means_python_with_one_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config.settings, "model_autoconfig", "Rusty")
    monkeypatch.setattr(config.settings, "model_autoconfig_bin", _fake_bin(tmp_path, ECHO))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        first = dataclasses.asdict(_analyze())
        second = dataclasses.asdict(_analyze())
    assert first == second and first["error"] == ""
    warnings = [r for r in caplog.records if "MODEL_AUTOCONFIG" in r.getMessage()]
    assert len(warnings) == 1 and "using python" in warnings[0].getMessage()
    assert _calls(tmp_path) == []


def test_upper_case_rust_is_rust(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "model_autoconfig", " RUST ")
    assert autoconfig_core.impl_choice() == "rust"


def test_rust_agreeing_gives_exactly_the_python_recommendation(tmp_path, monkeypatch):
    py = dataclasses.asdict(_analyze(summary=_summary(experts=64), file_size=60 * 2**30))
    fake = _fake_bin(tmp_path, ECHO)
    _use_rust(monkeypatch, fake)
    rs = dataclasses.asdict(_analyze(summary=_summary(experts=64), file_size=60 * 2**30))
    assert rs == py
    (argv, env), = _calls(tmp_path)
    assert argv == ["size"]
    # The child gets a minimal environment: nothing of the service's (tokens, settings).
    # (conftest sets MODELS_DIR etc. in this process; the shell and interpreter add their own.)
    assert "PATH" in env and not {"MODELS_DIR", "DATA_DIR", "MODELS_INI_PATH"} & set(env)


@pytest.mark.parametrize("body,reason", [
    ("sys.exit(3)\n", "rejected"),
    ("print('not json')\n", "malformed_output"),
    ("print(json.dumps([plan]))\n", "malformed_output"),
    ("time.sleep(30)\n", "timeout"),
    ("sys.stdout.write(' ' * (17 * 1024 * 1024))\n", "output_too_large"),
])
def test_rust_faults_refuse_the_recommendation(tmp_path, monkeypatch, body, reason, caplog):
    monkeypatch.setattr(autoconfig_core, "CORE_TIMEOUT_S", 2.0)
    _use_rust(monkeypatch, _fake_bin(tmp_path, body))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        rec = _analyze()
    assert rec.recommended_ctx == 0 and rec.values == {} and rec.plans == []
    assert "MODEL_AUTOCONFIG=rust" in rec.error and reason in rec.error
    assert any(reason in r.getMessage() for r in caplog.records)


def test_missing_binary_refuses(tmp_path, monkeypatch):
    _use_rust(monkeypatch, str(tmp_path / "nope"))
    rec = _analyze()
    assert rec.recommended_ctx == 0 and "missing_binary" in rec.error


def test_python_exceptions_still_raise_under_rust(tmp_path, monkeypatch):
    """Python runs first; a request it raises on raises exactly as in python mode, and the
    binary is never asked."""
    fake = _fake_bin(tmp_path, ECHO)
    b = _backends()
    b[0].update(gpu_count=3, card_vram_gb=[12.0])     # fewer cards than GPUs: IndexError
    with pytest.raises(IndexError):
        _analyze(backends=b, file_size=30 * 2**30)
    _use_rust(monkeypatch, fake)
    with pytest.raises(IndexError):
        _analyze(backends=b, file_size=30 * 2**30)
    assert _calls(tmp_path) == []


# ---- the mismatch policy: Python's plan only when it is the conservative one ----

MUTATIONS = {
    # Rust larger than Python: Python is conservative, so its plan is used.
    "rust ctx larger": ("plan['ctx'] *= 2; plan['initial_ctx'] *= 2", True),
    "rust cache-ram larger": ("plan['cache_ram'] = (plan['cache_ram'] or 0) + 512", True),
    "rust cache-ram unset": ("plan['cache_ram'] = None", True),
    "rust only differs in the table": ("plan['plans'][0]['rows'][0]['kv_gb'] += 1.0", True),
    # Rust smaller, or a different backend / placement mode: refuse.
    "rust ctx smaller": ("plan['ctx'] //= 2; plan['initial_ctx'] //= 2", False),
    "rust cache-ram smaller": ("plan['cache_ram'] = (plan['cache_ram'] or 2) // 2", False),
    "rust ngl smaller": ("plan['ngl'] = 3", False),
    "rust other backend": ("plan['recommended'] = None", False),
    "rust fit flipped": ("plan['fit'] = not plan['fit']", False),
    "rust malformed field": ("plan['ctx'] = 'big'", False),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_mismatch_policy(tmp_path, monkeypatch, name, caplog):
    mutation, python_used = MUTATIONS[name]
    py = dataclasses.asdict(_analyze())
    _use_rust(monkeypatch, _fake_bin(tmp_path, mutation + "\n" + ECHO))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        rs = dataclasses.asdict(_analyze())
    if python_used:
        assert rs == py
        assert any("more conservative" in r.getMessage() for r in caplog.records)
    else:
        assert rs["recommended_ctx"] == 0 and rs["values"] == {}
        assert "mismatch" in rs["error"]


def test_python_unset_cache_ram_is_not_conservative():
    py = {"recommended": 0, "fit": False, "sized": True, "ctx": 4096, "initial_ctx": 4096,
          "ngl": None, "cache_ram": None}
    rs = dict(py, cache_ram=1024)
    assert not autoconfig_core.python_is_conservative(py, rs, 1)
    assert autoconfig_core.python_is_conservative(rs, py, 1)


def test_never_larger_than_rust_property(monkeypatch):
    """Whatever the Rust side answers, plan_sizes returns Python's plan or raises, and when it
    returns, no written setting is larger than Rust's (seeded random perturbations)."""
    r = random.Random(1133)
    reqs = []
    real = autoconfig_core.plan_sizes
    monkeypatch.setattr(autoconfig_core, "plan_sizes", lambda req, model="": reqs.append(req) or real(req, model))
    for experts in (None, 8, 64):
        for vram in (8.0, 24.0, 48.0):
            for preset in ("", "fast", "long-ctx"):
                _analyze(summary=_summary(experts=experts), file_size=int(20 * 2**30),
                         backends=_backends(vram=vram), preset=preset)
    monkeypatch.setattr(autoconfig_core, "plan_sizes", real)
    monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
    used = refused = 0
    for _ in range(600):
        req = r.choice(reqs)
        py = autoconfig_core.size_plan(req)
        rs = json.loads(json.dumps(py))
        for key in r.sample(["ctx", "initial_ctx", "cache_ram", "ngl", "fit", "recommended"], r.randint(1, 3)):
            if key == "fit":
                rs[key] = r.random() < 0.5
            elif key == "recommended":
                rs[key] = r.choice([None, 0, 1])
            else:
                rs[key] = r.choice([None, 0, 1, 512, 999, 4096, 8192, 131072, 10**6, (rs.get(key) or 0) + r.randint(-5, 5)])
        monkeypatch.setattr(autoconfig_core, "size_plan_rust", lambda _req, _rs=rs: _rs)
        try:
            got = autoconfig_core.plan_sizes(req)
        except autoconfig_core.AutoconfigCoreError:
            refused += 1
            continue
        used += 1
        assert got == py
        p, q = autoconfig_core._written(py, req["n_sessions"]), autoconfig_core._written(rs, req["n_sessions"])
        assert p[:2] == q[:2] and all(a <= b for a, b in zip(p[2:], q[2:]))
    assert used and refused


# ---- bounded output, default binary path, per-shape mismatch logging (noevia#1138, #1139) ----

def _pid_alive(pid: int) -> bool:
    import os
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A zombie still answers signal 0; treat it as gone.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def test_endless_stdout_is_cut_at_the_cap_and_the_child_killed(tmp_path, monkeypatch, caplog):
    import time
    monkeypatch.setattr(autoconfig_core, "CORE_STDOUT_CAP", 256 * 1024)
    monkeypatch.setattr(autoconfig_core, "CORE_TIMEOUT_S", 30.0)
    pidfile = tmp_path / "pid"
    body = (f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
            "chunk = b' ' * 65536\n"
            "while True:\n"
            "    sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()\n")
    _use_rust(monkeypatch, _fake_bin(tmp_path, body))
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        rec = _analyze()
    assert time.monotonic() - t0 < 10          # cut by the cap, not the 30 s timeout
    assert rec.recommended_ctx == 0 and "output_too_large" in "".join(r.getMessage() for r in caplog.records)
    pid = int(pidfile.read_text())
    for _ in range(100):
        if not _pid_alive(pid):
            break
        time.sleep(0.05)
    assert not _pid_alive(pid)


def test_endless_stdout_with_a_stalled_reader_times_out_and_kills(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(autoconfig_core, "CORE_TIMEOUT_S", 1.0)
    pidfile = tmp_path / "pid"
    body = f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n"
    _use_rust(monkeypatch, _fake_bin(tmp_path, body))
    t0 = time.monotonic()
    rec = _analyze()
    assert time.monotonic() - t0 < 8 and rec.recommended_ctx == 0
    pid = int(pidfile.read_text())
    for _ in range(100):
        if not _pid_alive(pid):
            break
        time.sleep(0.05)
    assert not _pid_alive(pid)


def test_default_binary_is_the_installed_path_and_the_env_override_still_works(monkeypatch):
    assert config.Settings().model_autoconfig_bin == "/usr/local/bin/model-autoconfig"
    assert autoconfig_core.DEFAULT_BINARY == "/usr/local/bin/model-autoconfig"
    monkeypatch.setattr(config.settings, "model_autoconfig_bin", "")
    monkeypatch.setattr(autoconfig_core.os.path, "isfile", lambda p: p == autoconfig_core.DEFAULT_BINARY)
    monkeypatch.setattr(autoconfig_core.os, "access", lambda p, m: True)
    assert autoconfig_core._rust_binary() == "/usr/local/bin/model-autoconfig"


def test_mismatch_logs_once_per_request_shape_with_the_model_name(tmp_path, monkeypatch, caplog):
    _use_rust(monkeypatch, _fake_bin(tmp_path, "plan['ctx'] *= 2; plan['initial_ctx'] *= 2\n" + ECHO))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        for _ in range(3):
            _analyze(model_rel="alpha/model-a.gguf")
        _analyze(model_rel="beta/model-b.gguf", file_size=9 * 2**30)   # a different shape
        _analyze(model_rel="beta/model-b.gguf", file_size=9 * 2**30)
    msgs = [r.getMessage() for r in caplog.records if "more conservative" in r.getMessage()]
    assert len(msgs) == 2
    assert "alpha/model-a.gguf" in msgs[0] and "beta/model-b.gguf" in msgs[1]


def test_refused_mismatch_logs_once_per_shape_with_the_model_name(tmp_path, monkeypatch, caplog):
    _use_rust(monkeypatch, _fake_bin(tmp_path, "plan['ctx'] //= 2; plan['initial_ctx'] //= 2\n" + ECHO))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        for _ in range(3):
            _analyze(model_rel="alpha/model-a.gguf")
    msgs = [r.getMessage() for r in caplog.records if "mismatch" in r.getMessage()]
    assert len(msgs) == 1 and "alpha/model-a.gguf" in msgs[0]
