"""Hostile or corrupt GGUF metadata must degrade to an `_error`, never to memory exhaustion,
a hung event loop or a 500 (#869, #870). Synthetic bytes only."""
from __future__ import annotations

import io
import struct
import time
import tracemalloc

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT
from app import autoconfig, gguf_meta
from app.main import app

U32, STRING, ARRAY, U64 = 4, 8, 9, 10


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _header(kv_count: int, tensors: int = 0) -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensors) + struct.pack("<Q", kv_count)


def _kv_u32(key: str, v: int) -> bytes:
    return _s(key) + struct.pack("<II", U32, v)


def _peak_of(fn):
    tracemalloc.start()
    try:
        out = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak


MB2 = 2 * 1024 * 1024


@pytest.mark.parametrize("n", [1 << 33, 1 << 40, (1 << 63), (1 << 64) - 1])
def test_huge_string_length_in_a_local_file_sets_error_without_reading_it(tmp_path, n):
    p = tmp_path / "evil.gguf"
    # A real file: a buffered reader asked for n bytes reserves n bytes up front.
    p.write_bytes(_header(2) + _s("general.architecture") + struct.pack("<I", STRING)
                  + struct.pack("<Q", n) + b"x" * 64)
    raw, peak = _peak_of(lambda: gguf_meta.read_raw(p))
    assert "_error" in raw
    assert peak < MB2


@pytest.mark.parametrize("count", [1 << 40, 1 << 63, (1 << 64) - 1])
@pytest.mark.parametrize("subtype", [U32, STRING])
def test_huge_array_count_is_not_walked_or_seeked_past_the_end(count, subtype):
    buf = (_header(2) + _s("tokenizer.ggml.tokens") + struct.pack("<I", ARRAY)
           + struct.pack("<IQ", subtype, count)
           + (b"\0" * 8 * 4 if subtype == STRING else b"\0" * 4 * 16))
    t0 = time.monotonic()
    raw, peak = _peak_of(lambda: gguf_meta.read_raw_bytes(buf))
    assert time.monotonic() - t0 < 1.0 and peak < MB2
    assert "_error" in raw


def test_deeply_nested_arrays_do_not_raise_recursion_error():
    depth = 5000
    body = b"".join(struct.pack("<IQ", ARRAY, 1) for _ in range(depth))
    buf = _header(1) + _s("evil") + struct.pack("<I", ARRAY) + body
    raw = gguf_meta.read_raw_bytes(buf)
    assert "nested" in raw["_error"]


def test_one_level_of_nested_arrays_is_still_accepted():
    inner = struct.pack("<IQ", U32, 2) + struct.pack("<II", 7, 8)
    buf = _header(1) + _s("k") + struct.pack("<I", ARRAY) + struct.pack("<IQ", ARRAY, 1) + inner
    raw = gguf_meta.read_raw_bytes(buf)
    assert raw["k"] == [[7, 8]] and "_error" not in raw


def test_implausible_kv_count_is_refused_up_front():
    raw, peak = _peak_of(lambda: gguf_meta.read_raw_bytes(_header((1 << 64) - 1)))
    assert "implausible kv_count" in raw["_error"] and peak < MB2


def test_truncated_fixed_header_is_a_meta_error_not_a_struct_error():
    with pytest.raises(gguf_meta.GgufMetaError):
        gguf_meta.read_raw_bytes(b"GGUF\x03\x00")


def test_oversized_but_present_string_is_truncated_and_parsing_continues():
    big = b"a" * (gguf_meta.MAX_STRING_LEN + 5000)
    buf = (_header(2) + _s("tokenizer.chat_template") + struct.pack("<I", STRING)
           + struct.pack("<Q", len(big)) + big + _kv_u32("llama.block_count", 32))
    raw = gguf_meta.read_raw_bytes(buf)
    assert raw["tokenizer.chat_template"].endswith("…[truncated]")
    assert len(raw["tokenizer.chat_template"]) < gguf_meta.MAX_STRING_LEN + 50
    assert raw["llama.block_count"] == 32 and "_error" not in raw


def test_a_range_cut_tokenizer_array_keeps_its_count_for_vocab_size():
    # A remote header is the first MB only; the tokens array is normally cut off mid-way. The
    # array must still report its declared count (that is what vocab_size comes from).
    strings = b"".join(_s(f"t{i}") for i in range(20))
    buf = (_header(3) + _kv_u32("llama.block_count", 32) + _s("tokenizer.ggml.tokens")
           + struct.pack("<I", ARRAY) + struct.pack("<IQ", STRING, 150_000) + strings)
    raw = gguf_meta.read_raw_bytes(buf)
    assert raw["llama.block_count"] == 32
    assert raw["tokenizer.ggml.tokens"]["count"] == 150_000
    assert "_error" in raw
    assert gguf_meta.summarize(raw)["model"]["vocab_size"] == 150_000


def test_error_surfaces_in_the_summary():
    raw = gguf_meta.read_raw_bytes(_header(5) + _kv_u32("a", 1))
    assert gguf_meta.summarize(raw)["general"]["header_error"]


def test_models_detail_on_a_hostile_file_never_500s():
    import shutil
    d = ROOT / "models" / "evil"
    d.mkdir(exist_ok=True)
    try:
        (d / "evil-Q4_K_M.gguf").write_bytes(
            _header(1) + _s("general.architecture") + struct.pack("<I", STRING)
            + struct.pack("<Q", (1 << 64) - 1) + b"x" * 64)
        (d / "notgguf-Q4_K_M.gguf").write_bytes(b"NOPE" + b"\0" * 64)
        with TestClient(app) as c:
            r = c.get("/api/v1/models/detail?key=evil/evil-Q4_K_M.gguf")
            assert r.status_code == 200 and r.json()["summary"]["general"]["header_error"]
            assert c.get("/api/v1/models/detail?key=evil/notgguf-Q4_K_M.gguf").status_code == 422
            assert c.get("/api/v1/models").status_code == 200
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --- #901: non-finite float metadata ---------------------------------------------------------

F32, F64 = 6, 12


def _nonfinite_gguf() -> bytes:
    """Synthetic header: NaN float32, +Inf float64 and -Inf float32 array element, plus a finite float."""
    return (
        _header(7)
        + _s("general.architecture") + struct.pack("<I", STRING) + _s("llama")
        + _s("llama.rope.freq_base") + struct.pack("<If", F32, float("nan"))
        + _s("llama.rope.scaling.factor") + struct.pack("<Id", F64, float("inf"))
        + _s("llama.expert_count") + struct.pack("<Id", F64, float("inf"))
        + _s("llama.context_length") + struct.pack("<Id", F64, float("nan"))
        + _s("llama.attention.head_count_kv") + struct.pack("<IIQ", ARRAY, F32, 2)
        + struct.pack("<ff", float("-inf"), 8.0)
        + _s("general.parameter_count") + struct.pack("<If", F32, float("nan"))
    )


def _assert_json_safe(value) -> None:
    import json
    json.dumps(value, allow_nan=False)  # what Starlette's JSONResponse does


def test_non_finite_floats_become_null_in_the_summary():
    summary = gguf_meta.summarize(gguf_meta.read_raw_bytes(_nonfinite_gguf()))
    m = summary["model"]
    assert m["rope_freq_base"] is None
    assert m["rope_scaling_factor"] is None
    assert m["expert_count"] is None and m["context_length"] is None
    assert summary["general"]["params_raw"] is None and summary["general"]["params"] is None
    assert m["attention_head_count_kv"] == [None, 8.0]
    assert not summary["general"]["header_error"]
    _assert_json_safe(summary)


def test_finite_floats_are_untouched_by_the_sanitiser():
    buf = (_header(2) + _s("general.architecture") + struct.pack("<I", STRING) + _s("llama")
           + _s("llama.rope.freq_base") + struct.pack("<If", F32, 500000.0))
    assert gguf_meta.summarize(gguf_meta.read_raw_bytes(buf))["model"]["rope_freq_base"] == 500000.0


def test_models_detail_with_non_finite_floats_returns_200_with_nulls():
    import shutil
    d = ROOT / "models" / "nonfinite"
    d.mkdir(exist_ok=True)
    try:
        (d / "nan-Q4_K_M.gguf").write_bytes(_nonfinite_gguf())
        with TestClient(app) as c:
            r = c.get("/api/v1/models/detail?key=nonfinite/nan-Q4_K_M.gguf")
            assert r.status_code == 200
            m = r.json()["summary"]["model"]
            assert m["rope_freq_base"] is None and m["rope_scaling_factor"] is None
            assert c.get("/api/v1/models").status_code == 200
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_autoconfig_and_suggest_defaults_survive_non_finite_metadata():
    from app import ini
    summary = gguf_meta.summarize(gguf_meta.read_raw_bytes(_nonfinite_gguf()))
    ini.suggest_defaults(summary)
    rec = autoconfig.analyze(summary=summary, file_size=4 * 1024 ** 3, backends=_BACKEND, vision=False)
    _assert_json_safe(__import__("dataclasses").asdict(rec))


def test_kv_first_int_ignores_nulled_entries():
    assert autoconfig._kv_first_int({"_array": True, "count": 20, "sample": [None, 8, 8, None]}) == 8
    assert autoconfig._kv_first_int({"_array": True, "count": 20, "sample": [None, None]}, default=3) == 3
    assert autoconfig._kv_first_int([None, 8], default=3) == 3


def test_model_shape_survives_infinite_expert_count(tmp_path):
    from app import services
    p = tmp_path / "inf.gguf"
    p.write_bytes(_nonfinite_gguf())
    assert services.model_shape(p).expert_count == 0


# --- #870: implausible block_count -------------------------------------------------------

_BACKEND = [{"name": "engine", "vendor": "unknown", "vram_gb": 14.0, "gpu_count": 1,
             "card_vram_gb": [14.0], "host_ram_gb": 29.0, "baseline": {}}]


def _summary(block_count: int) -> dict:
    return {"arch": "llama", "model": {
        "arch": "llama", "context_length": 262144, "embedding_length": 4096,
        "block_count": block_count, "attention_head_count": 32, "attention_head_count_kv": 8}}


@pytest.mark.parametrize("layers", [10 ** 9, 0xFFFFFFFF, autoconfig.MAX_BLOCK_COUNT + 1])
def test_analyze_rejects_an_implausible_block_count_quickly(layers):
    t0 = time.monotonic()
    rec = autoconfig.analyze(summary=_summary(layers), file_size=4 * 1024 ** 3,
                             backends=_BACKEND, vision=False)
    assert time.monotonic() - t0 < 1.0
    assert rec.error and "block_count" in rec.error and not rec.recommended_ctx


def test_analyze_still_accepts_the_deepest_real_models():
    for layers in (32, 126, 512):
        rec = autoconfig.analyze(summary=_summary(layers), file_size=4 * 1024 ** 3,
                                 backends=_BACKEND, vision=False)
        assert not rec.error


def test_search_repo_runs_estimates_off_the_event_loop_and_survives_a_hostile_header(monkeypatch):
    import threading
    from app import helpers, hf, hw, services
    detail = hf.HfRepoDetail(id="o/r-GGUF", files=[
        hf.HfFile(path="m-Q4_K_M.gguf", size=int(2.6e9), quant="Q4_K_M",
                  shard_base="m-Q4_K_M.gguf", shard_index=None, shard_total=None)], readme_snippet=None)

    loop_threads = []

    async def repo_detail(repo):
        # Runs on the event loop: TestClient serves the app on a portal thread, not the main
        # thread, so "not the main thread" proves nothing. Compare against the loop's own thread.
        loop_threads.append(threading.current_thread())
        return detail

    async def header(repo, path):
        return _summary(0xFFFFFFFF)
    monkeypatch.setattr(hf, "repo_detail", repo_detail)
    monkeypatch.setattr(hf, "gguf_header", header)
    monkeypatch.setattr(services, "_fit_backends", lambda: {"engine": 14.0})
    monkeypatch.setattr(hw, "gpu_count_for", lambda n: 1)
    monkeypatch.setattr(hw, "card_vram_gb_for", lambda n: [14.0])
    monkeypatch.setattr(hw, "host_ram_gb", lambda: 29.0)
    real = helpers._preset_estimates
    threads = []

    def spy(*a, **kw):
        threads.append(threading.current_thread())
        return real(*a, **kw)
    monkeypatch.setattr(helpers, "_preset_estimates", spy)
    with TestClient(app) as c:
        t0 = time.monotonic()
        r = c.get("/api/v1/search/repo", params={"repo": "o/r-GGUF"})
    assert r.status_code == 200 and time.monotonic() - t0 < 2.0
    assert r.json()["groups"][0]["estimates"] == []
    assert threads and loop_threads
    assert all(t is not loop_threads[0] for t in threads)
