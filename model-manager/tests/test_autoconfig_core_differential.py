"""Differential test (MODEL_AUTOCONFIG): the model-autoconfig binary (sbstndalton/noevia-rs) and
the Python reference must agree exactly - every int an int, every float the same float - on the
shared fixture tables (tests/fixtures/model-autoconfig.v1.json for the size core `size_plan`, and
model-autoconfig-check.v1.json for input prep and values assembly through `check_reference`;
the same files as noevia-rs's crates/model-autoconfig/tests/fixtures/) and on seeded random
corpora generated here; where Python raises, the binary must name the same exception type.
Fixture cases named "[stricter] ..." are inputs the port refuses on purpose: the binary must
refuse them (never answer differently).

The Python half (the reference still produces every committed expectation) always runs. The
binary half runs only when MODEL_AUTOCONFIG_BIN points at a built model-autoconfig (CI builds it
at the Dockerfile's NOEVIA_RS_REF); skipped otherwise. Synthetic only: invented model shapes and
backends; nothing loads a model or starts llama.cpp."""
from __future__ import annotations

import dataclasses
import json
import os
import random
import subprocess
from pathlib import Path

import pytest

from app import autoconfig, autoconfig_core, config
from app.autoconfig_core import canonical, kv_shape, size_plan

FIXTURES = Path(__file__).parent / "fixtures" / "model-autoconfig.v1.json"
BIN = os.environ.get("MODEL_AUTOCONFIG_BIN", "")
needs_bin = pytest.mark.skipif(not BIN, reason="MODEL_AUTOCONFIG_BIN not set: no model-autoconfig binary to compare")


def _cases() -> list[dict]:
    return json.loads(FIXTURES.read_text())["cases"]


CHECK_FIXTURES = FIXTURES.with_name("model-autoconfig-check.v1.json")


def _check_cases() -> list[dict]:
    return json.loads(CHECK_FIXTURES.read_text())["cases"]


def _python_check(text: str) -> dict:
    try:
        return autoconfig_core.check_reference(json.loads(text))
    except Exception as e:  # noqa: BLE001 - any exception is the reference refusing the input
        return {"error": type(e).__name__}


def _rust_check(text: str) -> dict:
    proc = subprocess.run([BIN, "check"], input=text.encode(), capture_output=True, timeout=60, check=False)
    if proc.returncode != 0:
        assert proc.stdout == b"", "an error must leave stdout empty"
        code = proc.stderr.decode(errors="replace").split(": ", 2)[1]
        return {"error": code.removeprefix("python:"), "python": code.startswith("python:")}
    return json.loads(proc.stdout)


def _python(text: str) -> dict:
    try:
        return size_plan(json.loads(text))
    except Exception as e:  # noqa: BLE001 - any exception is the reference refusing the input
        return {"error": type(e).__name__}


def _rust(text: str) -> dict:
    proc = subprocess.run([BIN, "size"], input=text.encode(), capture_output=True, timeout=60, check=False)
    if proc.returncode != 0:
        assert proc.stdout == b"", "an error must leave stdout empty"
        # "model-autoconfig: python:IndexError: ..." -> IndexError; other codes stay as they are.
        code = proc.stderr.decode(errors="replace").split(": ", 2)[1]
        return {"error": code.removeprefix("python:")}
    return json.loads(proc.stdout)


def test_python_reference_still_produces_every_expectation():
    cases = _cases()
    assert len(cases) >= 150
    assert any("error" in c["expect"] for c in cases)
    for c in cases:
        assert canonical(_python(c["input"])) == canonical(c["expect"]), c["name"]


@needs_bin
def test_binary_agrees_on_every_fixture():
    cases = _cases()
    agreed = sum(canonical(_rust(c["input"])) == canonical(c["expect"]) for c in cases)
    assert agreed == len(cases)
    print(f"model-autoconfig fixtures: {agreed}/{len(cases)} agree")


def test_python_reference_still_produces_every_check_expectation():
    cases = _check_cases()
    assert len(cases) >= 400
    assert any("error" in c["expect"] for c in cases)
    assert sum(c["name"].startswith("[stricter]") for c in cases) >= 5
    for c in cases:
        assert canonical(_python_check(c["input"])) == canonical(c["expect"]), c["name"]


@needs_bin
def test_binary_agrees_on_every_check_fixture():
    agreed = stricter = 0
    for c in _check_cases():
        got = _rust_check(c["input"])
        if c["name"].startswith("[stricter]"):
            assert "error" in got and not got["python"], (c["name"], got)
            stricter += 1
            continue
        got.pop("python", None)
        assert canonical(got) == canonical(c["expect"]), c["name"]
        agreed += 1
    print(f"model-autoconfig check fixtures: {agreed} agree, {stricter} refused on purpose")


# ---- a seeded random corpus of analyze() inputs, built here (not committed) ----

def _summary(r: random.Random) -> dict:
    layers = r.choice([1, 2, 6, 12, 24, 32, 40, 48, 64, 80, r.randint(1, 160)])
    m: dict = {"block_count": layers, "attention_head_count": r.choice([8, 16, 32, 40, 64]),
               "embedding_length": r.choice([2048, 3072, 4096, 5120, 8192]),
               "context_length": r.choice([0, 4096, 8192, 32768, 40960, 131072, 262144, 1048576,
                                           100000, -5, 2**40, r.randint(1, 300000)]),
               "attention_head_count_kv": r.choice([8, 4, 2, 1, {"_array": True, "count": layers,
                                                    "sample": [r.choice([8, 1, 2]) for _ in range(min(layers, 12))]}])}
    if r.random() < 0.4:
        m["expert_count"] = r.choice([2, 4, 8, 16, 32, 64, 128])
    if r.random() < 0.3:
        m.update(key_length=r.choice([128, 256, 512]), value_length=r.choice([128, 256, 512]))
    if r.random() < 0.15:
        m.update(full_attention_interval=r.choice([0, 1, 2, 4]), ssm_state_size=r.choice([0, 128]))
    if r.random() < 0.3:
        per = r.choice([2, 3, 6])
        pat = [(i % per) != per - 1 for i in range(min(layers, 14))]
        m.update(sliding_window=r.choice([512, 1024, 4096]),
                 sliding_window_pattern=r.choice([{"_array": True, "count": layers, "sample": pat}, pat, None]),
                 key_length_swa=r.choice([None, 128, 256]), shared_kv_layers=r.choice([None, 0, 4, 18]))
    if r.random() < 0.3:
        m.update(arch=r.choice(["gemma3", "llama", ""]), rope_scaling_type=r.choice([None, "none", "linear", " NONE "]),
                 rope_scaling_factor=r.choice([None, 0.0, 4.0]))
    if r.random() < 0.03:
        m["block_count"] = r.choice([5000, -1])          # an early refusal
    feats = r.choice([None, {}, {k: r.random() < 0.5 for k in (
        "accepts_enable_thinking", "accepts_reasoning_effort", "uses_think_tags", "uses_channel_thought",
        "accepts_preserve_thinking")}])
    return {"arch": r.choice(["llama", "qwen3", "gemma3", "gemma4", ""]), "model": m,
            "chat_template": r.choice(["", "x"]), "chat_template_features": feats}


def _backends(r: random.Random) -> list[dict]:
    out = []
    for i in range(r.choice([1, 1, 2, 3])):
        gc = r.choice([1, 1, 2, 3])
        b = {"name": r.choice(["a", "b", f"x{i}"]), "vendor": "cuda",
             "vram_gb": r.choice([4, 8, 12, 16, 23.9, 24, 32, 48, 80, round(r.uniform(1, 100), 3)]),
             "gpu_count": gc, "host_ram_gb": r.choice([0, 16, 32, 64, 125.5]), "baseline": {}}
        if gc > 1 and r.random() < 0.7:
            b["card_vram_gb"] = [r.choice([8, 12, 11.6, 16, 24]) for _ in range(r.choice([gc, gc, gc - 1, gc + 1]))]
        out.append(b)
    if r.random() < 0.03:
        out = r.choice([[], [dict(out[0], vram_gb=0)]])  # an early refusal
    return out


def _section(r: random.Random) -> dict | None:
    if r.random() < 0.5:
        return None
    return {k: r.choice(v) for k, v in {
        "ubatch-size": ["", "512", "2048", " 4096 ", "1_024", "x"], "batch-size": ["", "256", "8192"],
        "mmproj": ["", "/models/s/mmproj-F16.gguf"], "ngl": ["999", "30"], "n-cpu-moe": ["4"]}.items()
        if r.random() < 0.4}


def _analyze_kwargs(r: random.Random) -> dict:
    return dict(summary=_summary(r), section_name=r.choice(["", "s"]), current_section=_section(r),
                model_rel=r.choice(["", "/models/s/m.gguf"]), vision=r.random() < 0.9,
                file_size=int(r.choice([0.5, 2, 4.7, 9, 14, 20, 27, 40, 70, 90]) * 2**30) + r.randint(0, 2**20),
                backends=_backends(r), preset=r.choice(["", "fast", "balanced", "long-ctx"]),
                n_sessions=r.choice([1, 1, 2, 4, 8]), mmproj_gb_override=r.choice([None, None, 0.8]),
                prompt_tps=r.choice([0.0, 0.0, 500.0, 1737.4]), prompt_budget_s=r.choice([120.0, 10.0, 3600.0]),
                verified_ctx=r.choice([0, 0, 16384, 50000]))


def _parts(n: int, seed: int) -> list[dict]:
    """The `check` requests analyze() builds for random inputs (captured, not re-derived): all
    three parts for a recommendation, prep alone for an early refusal."""
    seen: list[dict] = []

    def capture(**kw):
        seen.append(json.loads(json.dumps({"prep": kw["prep_in"], "size": kw["size_req"], "values": kw["values_in"]})))

    def capture_refusal(**kw):
        seen.append(json.loads(json.dumps({"prep": kw["prep_in"]})))
    r = random.Random(seed)
    orig = autoconfig_core.confirm, autoconfig_core.confirm_refusal
    autoconfig_core.confirm, autoconfig_core.confirm_refusal = capture, capture_refusal
    try:
        while len(seen) < n:
            try:
                autoconfig.analyze(**_analyze_kwargs(r))
            except Exception:  # noqa: BLE001 - raising inputs simply produce no request
                pass
    finally:
        autoconfig_core.confirm, autoconfig_core.confirm_refusal = orig
    return seen


def _requests(n: int, seed: int) -> list[dict]:
    """The size requests analyze() builds for random inputs."""
    return [p["size"] for p in _parts(10 * n, seed) if "size" in p][:n]


@needs_bin
def test_binary_agrees_on_a_seeded_random_corpus():
    reqs = _requests(800, seed=1008)
    disagree = []
    for i, req in enumerate(reqs):
        text = json.dumps(req)
        if canonical(_rust(text)) != canonical(_python(text)):
            disagree.append(i)
    assert disagree == [], f"{len(disagree)} of {len(reqs)} disagree, first {disagree[:5]}"


@needs_bin
def test_binary_check_agrees_on_a_seeded_random_corpus():
    """Input prep, size plan and values for what analyze() really sends, part by part."""
    parts = _parts(800, seed=1137)
    assert sum("values" in p for p in parts) > 300 and sum("values" not in p for p in parts) > 20
    disagree = []
    for i, p in enumerate(parts):
        text = json.dumps(p)
        if canonical(_rust_check(text)) != canonical(_python_check(text)):
            disagree.append(i)
    assert disagree == [], f"{len(disagree)} of {len(parts)} disagree, first {disagree[:5]}"


@needs_bin
def test_rust_mode_never_gives_a_larger_plan_than_python(monkeypatch):
    """The property the switch exists for: under MODEL_AUTOCONFIG=rust a recommendation is
    either exactly the Python one or refused - never anything else, so never larger."""
    r = random.Random(20261008)
    kwargs = [_analyze_kwargs(r) for _ in range(300)]
    refused = 0
    for kw in kwargs:
        monkeypatch.setattr(config.settings, "model_autoconfig", "python")
        try:
            py = dataclasses.asdict(autoconfig.analyze(**json.loads(json.dumps(kw))))
        except Exception as e:  # noqa: BLE001
            py = {"raised": type(e).__name__}
        monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
        monkeypatch.setattr(config.settings, "model_autoconfig_bin", BIN)
        try:
            rs = dataclasses.asdict(autoconfig.analyze(**json.loads(json.dumps(kw))))
        except Exception as e:  # noqa: BLE001
            rs = {"raised": type(e).__name__}
        if rs != py:
            refused += 1
            assert "MODEL_AUTOCONFIG=rust" in rs.get("error", ""), (kw, rs)
            assert rs["values"] == {} and rs["recommended_ctx"] == 0
    assert refused == 0, f"{refused} refusals on in-domain inputs"


def test_kv_shape_matches_the_raw_kv_cache_bytes():
    """kv_cache_bytes (the raw per-layer arrays) and kv_shape_bytes (the shape the size core
    and the Rust port read) are one computation."""
    r = random.Random(7)
    for _ in range(2000):
        m = _summary(r)["model"]
        layers = m["block_count"]
        heads = m["attention_head_count"]
        kv = autoconfig._kv_first_int(m["attention_head_count_kv"], default=heads)
        args = (r.choice(["gemma3", "llama"]), layers, kv, m["embedding_length"] // heads)
        swa = {k: m.get(k) for k in ("sliding_window", "sliding_window_pattern", "key_length_swa",
                                     "value_length_swa", "shared_kv_layers")}
        kw = dict(key_length=m.get("key_length"), value_length=m.get("value_length"),
                  full_attention_interval=m.get("full_attention_interval"),
                  ssm_state_size=m.get("ssm_state_size"), kv_heads_pattern=m["attention_head_count_kv"], **swa)
        shape = kv_shape(*args, **kw)
        for ctx in (0, 4096, 131072 * 8):
            assert autoconfig.kv_cache_bytes(args[0], ctx, *args[1:], 1.0625, **kw) == \
                autoconfig_core.kv_shape_bytes(shape, ctx, 1.0625)
