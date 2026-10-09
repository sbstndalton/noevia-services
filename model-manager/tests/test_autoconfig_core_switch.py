"""MODEL_AUTOCONFIG switch: python stays the default and never spawns anything; rust runs the
model-autoconfig binary (`check`, once per analyze()) BESIDE the Python input prep, size core and
values assembly, which stay authoritative. Prep must agree exactly; the plan and the values are
used when Rust agrees exactly, or when Python's are the conservative ones (same backend, placement
mode and other values; no larger context, GPU layer count, prompt cache, batch, ubatch or image
tokens); any other disagreement and every Rust fault refuses the recommendation. A Python prep
refusal stands either way. A fake binary (a small script) stands in for model-autoconfig; the
real one is covered by test_autoconfig_core_differential.py. Synthetic models only - nothing
loads a model or starts llama.cpp."""
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
    runs `body` with `parts` = the parsed request, `out` = the Python reference's answer to it,
    `plan` = out["size"], `values` = out["values"] and `prep` = out["prep"] (same objects)."""
    body_file = tmp_path / "fake_model_autoconfig.py"
    body_file.write_text(textwrap.dedent(f"""\
        import json, os, sys, time
        sys.path.insert(0, {str(MM)!r})
        from app.autoconfig_core import check_reference
        raw = sys.stdin.read()
        open({str(tmp_path / "calls.log")!r}, "a").write(json.dumps([sys.argv[1:], sorted(os.environ)]) + "\\n")
        parts = json.loads(raw)
        out = check_reference(parts)
        plan, values, prep = out.get("size"), out.get("values"), out.get("prep")
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


ECHO = "print(json.dumps(out))\n"


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
    assert argv == ["check"]
    # The child gets a minimal environment: nothing of the service's (tokens, settings).
    # (conftest sets MODELS_DIR etc. in this process; the shell and interpreter add their own.)
    assert "PATH" in env and not {"MODELS_DIR", "DATA_DIR", "MODELS_INI_PATH"} & set(env)


@pytest.mark.parametrize("body,reason", [
    ("sys.exit(3)\n", "rejected"),
    ("print('not json')\n", "malformed_output"),
    ("print(json.dumps([out]))\n", "malformed_output"),
    ("del out['values']; print(json.dumps(out))\n", "malformed_output"),
    ("out['values'] = [['parallel', '1'], ['parallel', '1']]; print(json.dumps(out))\n", "malformed_output"),
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


def _captured_parts(monkeypatch, calls) -> list[dict]:
    """The confirm() arguments analyze() builds for `calls` (python mode: nothing is spawned)."""
    seen: list[dict] = []
    real = autoconfig_core.confirm
    monkeypatch.setattr(autoconfig_core, "confirm",
                        lambda **kw: seen.append(json.loads(json.dumps(kw))) or real(**kw))
    for kw in calls:
        _analyze(**kw)
    monkeypatch.setattr(autoconfig_core, "confirm", real)
    return seen


def _confirm_with(monkeypatch, parts: dict, rust_answer: dict) -> bool:
    """confirm() under rust with `rust_answer` as the binary's output: True if it accepts."""
    monkeypatch.setattr(autoconfig_core, "check_rust", lambda _parts, _a=rust_answer: _a)
    try:
        autoconfig_core.confirm(**parts)
    except autoconfig_core.AutoconfigCoreError:
        return False
    return True


def _request(parts: dict) -> dict:
    """The `check` request confirm() sends for its captured arguments."""
    req = {"prep": parts["prep_in"], "size": parts["size_req"], "values": parts["values_in"]}
    req.update({k: v for k, v in (parts.get("extra_in") or {}).items() if v is not None})
    return req


def _answer(parts: dict) -> dict:
    return json.loads(json.dumps(autoconfig_core.check_reference(_request(parts))))


def _calls_for_property() -> list[dict]:
    out = []
    for experts in (None, 8, 64):
        for vram in (8.0, 24.0, 48.0):
            for preset in ("", "fast", "long-ctx"):
                for extra in ({}, {"n_sessions": 3}, {"mmproj_gb_override": 0.9, "section_name": "s"}):
                    out.append(dict(summary=_summary(experts=experts), file_size=int(20 * 2**30),
                                    backends=_backends(vram=vram), preset=preset, **extra))
    return out


def test_never_larger_than_rust_property(monkeypatch):
    """Whatever the Rust side answers for the size plan, confirm() accepts only Python's plan
    when no written setting is larger than Rust's (seeded random perturbations)."""
    r = random.Random(1133)
    all_parts = _captured_parts(monkeypatch, _calls_for_property())
    monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
    used = refused = 0
    for _ in range(600):
        parts = r.choice(all_parts)
        ans = _answer(parts)
        py, rs = parts["plan"], ans["size"]
        for key in r.sample(["ctx", "initial_ctx", "cache_ram", "ngl", "fit", "recommended"], r.randint(1, 3)):
            if key == "fit":
                rs[key] = r.random() < 0.5
            elif key == "recommended":
                rs[key] = r.choice([None, 0, 1])
            else:
                rs[key] = r.choice([None, 0, 1, 512, 999, 4096, 8192, 131072, 10**6, (rs.get(key) or 0) + r.randint(-5, 5)])
        if not _confirm_with(monkeypatch, parts, ans):
            refused += 1
            continue
        used += 1
        n = parts["size_req"]["n_sessions"]
        p, q = autoconfig_core._written(py, n), autoconfig_core._written(rs, n)
        assert p[:2] == q[:2] and all(a <= b for a, b in zip(p[2:], q[2:]))
    assert used and refused


# ---- input prep and values (slice 2) ----

def test_values_never_larger_than_rust_property(monkeypatch):
    """Whatever the Rust side answers for the values, confirm() accepts Python's only when every
    value outside VALUE_LIMITS is identical (same keys, same order) and no batch, ubatch, image
    tokens, context, ngl or prompt cache of Python's is larger than Rust's."""
    r = random.Random(1137)
    all_parts = _captured_parts(monkeypatch, _calls_for_property())
    monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
    keys = sorted(autoconfig_core.VALUE_LIMITS) + ["parallel", "jinja", "mmproj-offload", "fit", "split-mode",
                                                    "reasoning", "rope-scale", "flash-attn"]
    used = refused = 0
    for _ in range(1500):
        parts = r.choice(all_parts)
        ans = _answer(parts)
        rs = dict(ans["values"])
        for key in r.sample(keys, r.randint(1, 3)):
            choice = r.random()
            if choice < 0.2:
                rs.pop(key, None)
            elif choice < 0.3:
                rs[key] = r.choice(["on", "x", "1.5", "", "-1"])
            else:
                cur = rs.get(key)
                base = int(cur) if cur and cur.lstrip("-").isdigit() else 1024
                rs[key] = str(max(0, base + r.choice([-4096, -1, 0, 1, 512, 10**6])))
        items = list(rs.items())
        if r.random() < 0.1 and len(items) > 2:
            i = r.randrange(len(items) - 1)
            items[i], items[i + 1] = items[i + 1], items[i]
        ans["values"] = [[k, v] for k, v in items]
        if not _confirm_with(monkeypatch, parts, ans):
            refused += 1
            continue
        used += 1
        py = parts["values"]
        got = dict(items)
        for k in set(py) | set(got):
            if k in autoconfig_core.VALUE_LIMITS:
                lim = autoconfig_core.VALUE_LIMITS[k]
                assert (int(py[k]) if k in py else lim) <= (int(got[k]) if k in got else lim), (k, py, got)
            else:
                assert py.get(k) == got.get(k), (k, py, got)
        assert [k for k in py if k in got] == [k for k in got if k in py]
    assert used > 50 and refused > 50, (used, refused)


VALUE_MUTATIONS = {
    # Rust larger (or unbounded) than Python: Python is conservative, so its values are used.
    "rust batch larger": ("values[:] = [[k, '8192' if k == 'batch-size' else v] for k, v in values]", True),
    "rust image tokens unset": ("values[:] = [p for p in values if p[0] != 'image-max-tokens']", True),
    "rust ubatch larger": ("values[:] = [[k, '4096' if k == 'ubatch-size' else v] for k, v in values]", True),
    # Rust smaller, or any other value different: refuse.
    "rust ubatch smaller": ("values[:] = [[k, '512' if k == 'ubatch-size' else v] for k, v in values]", False),
    "rust batch unset (2048 < 4096)": ("values[:] = [p for p in values if p[0] != 'batch-size']", False),
    "rust reasoning differs": ("values.append(['reasoning', 'on'])", False),
    "rust split-mode differs": ("values.append(['split-mode', 'row'])", False),
    "rust order differs": ("values[0], values[1] = values[1], values[0]", False),
    "rust number malformed": ("values[:] = [[k, 'big' if k == 'ubatch-size' else v] for k, v in values]", False),
}


@pytest.mark.parametrize("name", sorted(VALUE_MUTATIONS))
def test_values_mismatch_policy(tmp_path, monkeypatch, name, caplog):
    mutation, python_used = VALUE_MUTATIONS[name]
    kw = dict(n_sessions=2, section_name="s", mmproj_gb_override=0.9, model_rel="/models/s/m.gguf")
    py = dataclasses.asdict(_analyze(**kw))
    assert py["values"]["ubatch-size"] == "1024" and py["values"]["batch-size"] == "4096"
    _use_rust(monkeypatch, _fake_bin(tmp_path, mutation + "\n" + ECHO))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        rs = dataclasses.asdict(_analyze(**kw))
    if python_used:
        assert rs == py
        assert any("more conservative" in r.getMessage() for r in caplog.records)
    else:
        assert rs["recommended_ctx"] == 0 and rs["values"] == {}
        assert "mismatch" in rs["error"]


@pytest.mark.parametrize("mutation", [
    "prep['layers'] += 1",
    "prep['shape']['kv_heads'] = 1",
    "prep['model_gb_raw'] *= 0.5",
    "prep['backends'][0]['gpu_count'] = 2",
    "prep['refuse'] = 'kv'",
])
def test_prep_must_agree_exactly(tmp_path, monkeypatch, mutation):
    """Input prep feeds everything after it, so any difference refuses (even a smaller model)."""
    _use_rust(monkeypatch, _fake_bin(tmp_path, mutation + "\n" + ECHO))
    rec = _analyze()
    assert rec.recommended_ctx == 0 and rec.values == {} and "mismatch" in rec.error


REFUSALS = {
    "block_count": dict(summary=_summary(layers=5000)),
    "no_backends": dict(backends=[]),
    "ram": dict(file_size=500 * 2**30),
    "unsized_vram": dict(backends=_backends(vram=0)),
    "kv": dict(summary={"arch": "llama", "model": {"block_count": 32}, "chat_template": "x"}),
}


@pytest.mark.parametrize("kind", sorted(REFUSALS))
@pytest.mark.parametrize("body", [ECHO, "prep.clear(); prep['refuse'] = None\n" + ECHO, "sys.exit(3)\n"])
def test_a_python_prep_refusal_stands_under_rust(tmp_path, monkeypatch, caplog, kind, body):
    """Python's early refusal is returned unchanged whatever the Rust side says (agreeing,
    disagreeing or failing); a disagreement is logged, and only prep is sent."""
    py = dataclasses.asdict(_analyze(**REFUSALS[kind]))
    assert py["error"] and py["recommended_ctx"] == 0
    _use_rust(monkeypatch, _fake_bin(tmp_path, body))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        rs = dataclasses.asdict(_analyze(**REFUSALS[kind]))
    assert rs == py
    (argv, _env), = _calls(tmp_path)
    assert argv == ["check"]
    msgs = " ".join(r.getMessage() for r in caplog.records)
    if body == ECHO:
        assert "model-autoconfig" not in msgs
    elif "exit" in body:
        assert "rejected" in msgs
    else:
        assert "the refusal stands" in msgs


def test_refusal_path_sends_only_prep(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
    monkeypatch.setattr(autoconfig_core, "check_rust",
                        lambda parts: seen.append(parts) or autoconfig_core.check_reference(parts))
    _analyze(backends=[])
    assert len(seen) == 1 and set(seen[0]) == {"prep"}
    seen.clear()
    _analyze()
    # No backend carries its command, so the baseline part is not asked for; files always holds
    # at least the projector resolution.
    assert len(seen) == 1 and set(seen[0]) == {"prep", "size", "values", "spec", "files", "present"}


def test_values_are_conservative_edges():
    c = autoconfig_core.values_are_conservative
    assert c({"batch-size": "1024"}, {})              # absent = llama-server's 2048
    assert not c({"batch-size": "4096"}, {})
    assert c({"ubatch-size": "512"}, {})
    assert not c({"ubatch-size": "1024"}, {})
    assert c({"ctx-size": "8192"}, {})                # absent = unbounded
    assert not c({}, {"ctx-size": "8192"})
    assert not c({"ubatch-size": "1" * 50}, {"ubatch-size": "1" * 50})   # malformed: too long
    assert not c({"a": "1", "b": "2"}, {"b": "2", "a": "1"})            # order counts
    assert c({"a": "1", "ngl": "10", "b": "2"}, {"a": "1", "b": "2"})


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
    monkeypatch.delenv("MODEL_AUTOCONFIG_BIN", raising=False)
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


# ---- slices 3-6: speculative profiles, companion files, presentation, baseline ----

def _head_kw(tmp_path) -> dict:
    """A model folder with a draft head beside it and the Coding profile picked, so the spec part
    writes every limit (n-max 8, n-min 1, head ngl 999)."""
    models = tmp_path / "models"
    (models / "s" / "MTP").mkdir(parents=True)
    (models / "s" / "s.gguf").write_bytes(b"\0" * 16)
    with open(models / "s" / "MTP" / "mtp-Q8_0.gguf", "wb") as f:
        f.truncate(4096)
    return dict(models_dir=models, section_name="s", model_subdir="s", spec_profile="coding")


def _set_spec(key: str, value: str) -> str:
    return (f"for pair in out['spec']['values']:\n"
            f"    if pair[0] == {key!r}: pair[1] = {value!r}\n") + ECHO


SPEC_MUTATIONS = {
    # Rust's draft limits larger than Python's: Python is conservative, so its answer is used.
    "rust deeper drafts": ("spec-draft-n-max", "16", True),
    "rust larger draft minimum": ("spec-draft-n-min", "2", True),
    "rust more head layers on the GPU": ("spec-draft-ngl", "1000", True),
    # Smaller, unset, or any other spec value differing: refuse.
    "rust shallower drafts": ("spec-draft-n-max", "4", False),
    "rust fewer head layers": ("spec-draft-ngl", "12", False),
    "rust unset draft depth": ("spec-draft-n-max", "", False),
    "rust other confidence gate": ("spec-draft-p-min", "0.5", False),
    "rust other spec type": ("spec-type", "draft-mtp", False),
    "rust other head": ("spec-draft-model", "/models/x.gguf", False),
}


@pytest.mark.parametrize("name", sorted(SPEC_MUTATIONS))
def test_spec_mismatch_policy(tmp_path, monkeypatch, name, caplog):
    key, value, python_used = SPEC_MUTATIONS[name]
    kw = _head_kw(tmp_path)
    py = dataclasses.asdict(_analyze(**kw))
    assert dict(py["values"])["spec-draft-n-max"] == "8" and py["spec_head_rel"].endswith("mtp-Q8_0.gguf")
    _use_rust(monkeypatch, _fake_bin(tmp_path, _set_spec(key, value)))
    with caplog.at_level(logging.WARNING, logger="app.autoconfig_core"):
        rs = dataclasses.asdict(_analyze(**kw))
    if python_used:
        assert rs == py
        assert any("more conservative" in r.getMessage() for r in caplog.records)
    else:
        assert rs["recommended_ctx"] == 0 and rs["values"] == {} and "mismatch" in rs["error"]


PART_MUTATIONS = {
    "spec profile": "out['spec']['key'] = 'off'",
    "spec saved profile": "out['spec']['saved'] = 'custom'",
    "spec value order": "out['spec']['values'].reverse()",
    "spec budgeted head": "out['spec']['mtp_rel'] = ''",
    "files projector": "out['files'][-1] = {'mmproj_rel': '', 'mmproj_gb': 0.0, 'available': 'x'}",
    "files head": "out['files'][0] = '/models/s/other-mtp-.gguf'",
    "present diff": "out['present']['current_diff'].append('ngl: unset')",
    "present quirk order": "out['present']['quirks'].reverse()",
    "present displaced": "out['present']['displaced'] = out['present']['displaced'][1:]",
    "present minimal": "out['present']['minimal'] = out['present']['minimal'][:-1]",
    "present warning": "out['present']['warnings'] = []",
    "present saved preset": "out['present']['current_preset'] = 'fast'",
    "baseline": "out['baseline'][0].append(['keep', '1'])",
    "a part missing": "del out['present']",
}


@pytest.mark.parametrize("name", sorted(PART_MUTATIONS))
def test_any_other_part_disagreeing_refuses(tmp_path, monkeypatch, name):
    kw = _head_kw(tmp_path)
    b = _backends()
    b[0]["baseline_args"] = ["--models-dir", "/models", "-np", "1"]
    b[0]["baseline"] = autoconfig.parse_baseline(b[0]["baseline_args"])
    kw.update(backends=b, n_sessions=2, current_section={"ngl": "30", "keep": "64"},
              model_rel="/models/s/s-Q2_K.gguf")
    py = dataclasses.asdict(_analyze(**kw))
    assert py["error"] == "" and len(py["quirks"]) >= 2 and py["warnings"] and py["current_diff"]
    _use_rust(monkeypatch, _fake_bin(tmp_path, PART_MUTATIONS[name] + "\n" + ECHO))
    rs = dataclasses.asdict(_analyze(**kw))
    assert rs["recommended_ctx"] == 0 and rs["values"] == {} and "MODEL_AUTOCONFIG=rust" in rs["error"]
    assert ("mismatch" if name != "a part missing" else "malformed_output") in rs["error"]


def test_every_part_is_sent_and_agreement_is_exact(tmp_path, monkeypatch):
    kw = _head_kw(tmp_path)
    b = _backends()
    b[0]["baseline_args"] = ["-c", "8192", "--jinja"]
    b[0]["baseline"] = autoconfig.parse_baseline(b[0]["baseline_args"])
    kw["backends"] = b
    py = dataclasses.asdict(_analyze(**kw))
    seen = []
    monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
    monkeypatch.setattr(autoconfig_core, "check_rust",
                        lambda parts: seen.append(parts) or json.loads(json.dumps(autoconfig_core.check_reference(parts))))
    assert dataclasses.asdict(_analyze(**kw)) == py
    (parts,) = seen
    assert set(parts) == {"prep", "size", "values", "spec", "files", "present", "baseline"}
    assert [c["rule"] for c in parts["files"]] == ["mmproj_subdir", "projector", "mtp_folder"]
    assert parts["baseline"][0]["args"] == ["-c", "8192", "--jinja"]
    assert parts["baseline"][0]["known"] == sorted(autoconfig.ini.ALL_KNOWN_KEYS)


def test_a_baseline_that_its_command_does_not_give_refuses(tmp_path, monkeypatch):
    """The baseline analyze() used must be what parse_baseline makes of the command it carries."""
    b = _backends()
    b[0]["baseline_args"] = ["-np", "1"]
    b[0]["baseline"] = {}                 # not what "-np 1" parses to
    py = _analyze(backends=b)
    assert py.error == ""                 # python mode never looks
    _use_rust(monkeypatch, _fake_bin(tmp_path, ECHO))
    rs = _analyze(backends=b)
    assert rs.recommended_ctx == 0 and "mismatch" in rs.error


def test_spec_never_larger_than_rust_property(monkeypatch):
    """Whatever the Rust side answers for the spec part, confirm() accepts Python's only when every
    field and value is identical (same keys, same order) except a draft limit (n-max, n-min, head
    ngl) that is a plain number no larger in Python's."""
    r = random.Random(1152)
    calls = []
    for profile in ("coding", "balanced", "writing", "ngram", "off", "custom", ""):
        for cur in (None, {"spec-type": "draft-mtp", "spec-draft-n-max": "6", "spec-draft-ngl": "40",
                           "spec-draft-model": "/models/s/h.gguf"}):
            calls.append(dict(section_name="s", spec_profile=profile, current_section=cur,
                              summary=dict(_summary(), model=dict(_summary()["model"], nextn_predict_layers=1))))
    all_parts = _captured_parts(monkeypatch, calls)
    monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
    used = refused = 0
    for _ in range(1500):
        parts = r.choice(all_parts)
        ans = _answer(parts)
        rs = ans["spec"]
        for _ in range(r.randint(1, 2)):
            if r.random() < 0.15 or not rs["values"]:
                rs[r.choice(["key", "saved", "head", "mtp_rel"])] = r.choice(["", "off", "custom", "/models/x.gguf"])
                continue
            pair = r.choice(rs["values"])
            pair[1] = r.choice(["", "0", "1", "2", "4", "8", "16", "999", "1000", "0.05", "x", "-1", " 8",
                                str(int(pair[1]) + r.randint(-3, 3)) if pair[1].isdigit() else pair[1]])
        if not _confirm_with(monkeypatch, parts, ans):
            refused += 1
            continue
        used += 1
        py = parts["extra"]["spec"]
        assert all(py[k] == rs[k] for k in ("key", "saved", "head", "mtp_rel"))
        assert [k for k, _ in py["values"]] == [k for k, _ in rs["values"]]
        for (k, p), (_, q) in zip(py["values"], rs["values"]):
            if p != q:
                assert k in autoconfig_core.SPEC_LIMITS and int(p) <= int(q), (k, p, q)
    assert used and refused


def test_backend_list_carries_the_command_it_parsed(monkeypatch):
    from app import helpers
    monkeypatch.setattr(helpers.services, "_effective_container_names", lambda: ["llama-cuda"])
    monkeypatch.setattr(helpers.services, "_docker_client", lambda: None)
    monkeypatch.setattr(helpers.hw, "vram_gb_for", lambda name: 24.0)
    monkeypatch.setattr(helpers.hw, "gpu_count_for", lambda name: 1)
    monkeypatch.setattr(helpers.hw, "card_vram_gb_for", lambda name: [24.0])
    monkeypatch.setattr(helpers.hw, "host_ram_gb", lambda: 64.0)
    monkeypatch.setattr(helpers, "_container_baseline", lambda name: ["-np", "2", "--jinja"])
    (b,) = helpers._backend_list()
    assert b["baseline_args"] == ["-np", "2", "--jinja"]
    assert b["baseline"] == {"parallel": "2", "jinja": "true"}
