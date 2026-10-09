"""noevia#1159: Discover had no context estimate for LiquidAI/d1-3B-GGUF (arch lfm2). Its GGUF does
declare attention.head_count_kv, but per layer, with 0 for the short-conv layers that hold no KV
cache. The representative value was then 0 and autoconfig refused as if the key were missing.

llama.cpp's rule, which autoconfig now follows: an absent head_count_kv means head_count (MHA);
per-layer entries are read per layer, and a layer whose entry is 0 has no KV cache (llama.cpp
marks it recurrent). Only attention layers are sized. When the summary holds only a prefix of the
array (it keeps 8 entries of a long one), which layers attend is unknown, so it still refuses,
now saying why, and the reason reaches Discover as estimatesReason.

Synthetic shapes only; nothing loads a model."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import autoconfig, autoconfig_core
from app.main import app

BACKENDS = [{"name": "llama-cuda", "vram_gb": 16.0, "gpu_count": 1, "host_ram_gb": 32.0, "baseline": {}}]
BASE = {"block_count": 30, "attention_head_count": 32, "embedding_length": 2048, "context_length": 32768}
LFM2_HEADS = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0]


def _prep(model: dict, arch: str = "lfm2") -> dict:
    return autoconfig_core.prepare({"n_sessions": 1, "arch": arch, "model": model,
                                    "file_size": 1_600_000_000, "backends": BACKENDS})


def _kv(p: dict, ctx: int = 32768) -> int:
    return autoconfig_core.kv_shape_bytes(p["shape"], ctx, 1.0625)


def test_absent_kv_heads_is_mha():
    absent = _prep(dict(BASE), arch="llama")
    explicit = _prep(dict(BASE, attention_head_count_kv=32), arch="llama")
    assert absent["refuse"] is None and absent["shape"] == explicit["shape"]
    assert absent["shape"]["kv_heads"] == 32 and absent["shape"]["layers"] == 30


def test_gqa_scalar_is_unchanged():
    p = _prep(dict(BASE, attention_head_count_kv=8), arch="llama")
    assert p["shape"]["kv_heads"] == 8 and p["shape"]["layers"] == 30


def test_hybrid_per_layer_heads_size_only_the_attention_layers():
    p = _prep(dict(BASE, attention_head_count_kv=LFM2_HEADS))
    n_attn = sum(1 for h in LFM2_HEADS if h)
    assert p["refuse"] is None
    assert p["layers"] == 30, "the model's own layer count (offload, split) is untouched"
    assert p["shape"]["layers"] == n_attn and p["shape"]["kv_heads"] == 8
    # The same bytes as a dense stack of just those layers at 8 KV heads.
    dense = _prep(dict(BASE, block_count=n_attn, attention_head_count_kv=8), arch="llama")
    assert _kv(p) == _kv(dense) > 0


def test_mixed_head_counts_take_the_largest():
    heads = [0, 4, 0, 8, 0, 2]
    p = _prep(dict(BASE, block_count=6, attention_head_count_kv=heads))
    assert p["shape"]["layers"] == 3 and p["shape"]["kv_heads"] == 8


def test_a_prefix_of_a_hybrid_array_refuses_with_its_reason():
    # What the live d1-3B summary looks like: count 30, the first 8 entries.
    model = dict(BASE, attention_head_count_kv={"_array": True, "count": 30, "sample": LFM2_HEADS[:8]})
    p = _prep(model)
    assert p == {"refuse": "kv_layers", "known": 8, "count": 30, "layers": 30}
    rec = autoconfig.analyze(summary={"arch": "lfm2", "model": model}, file_size=1_600_000_000,
                             backends=BACKENDS, preset="fast")
    assert rec.recommended_ctx == 0
    assert "hybrid model" in rec.error and "only the first 8 of its 30 entries" in rec.error
    assert "missing/zero" not in rec.error


def test_wrong_length_refuses():
    p = _prep(dict(BASE, block_count=6, attention_head_count_kv=[0, 8, 0]))
    assert p["refuse"] == "kv_layers"
    err = autoconfig._prep_refusal(p).error
    assert "it has 3 entries for 6 layers" in err


@pytest.mark.parametrize("kv", [0, [0, 0, 0, 0]])
def test_zero_heads_still_refuse_as_missing(kv):
    p = _prep(dict(BASE, block_count=4, attention_head_count_kv=kv))
    assert p == {"refuse": "kv", "missing": ["attention_head_count_kv"]}


def test_full_hybrid_array_gives_an_estimate_end_to_end():
    model = dict(BASE, attention_head_count_kv=LFM2_HEADS)
    rec = autoconfig.analyze(summary={"arch": "lfm2", "model": model}, file_size=1_600_000_000,
                             backends=BACKENDS, preset="fast")
    assert not rec.error and rec.recommended_ctx > 0


def _fit(monkeypatch):
    from app import hw, services
    monkeypatch.setattr(services, "_fit_backends", lambda: {"llama-cuda": 16.0})
    monkeypatch.setattr(hw, "gpu_count_for", lambda n: 1)
    monkeypatch.setattr(hw, "card_vram_gb_for", lambda n: [16.0])
    monkeypatch.setattr(hw, "host_ram_gb", lambda: 32.0)


def test_estimates_carry_the_refusal_reason(monkeypatch):
    from app import helpers
    _fit(monkeypatch)
    summary = {"arch": "lfm2", "model": dict(BASE, attention_head_count_kv=
                                             {"_array": True, "count": 30, "sample": LFM2_HEADS[:8]})}
    est, reason = helpers._preset_estimates_with_reason(summary, 1_600_000_000)
    assert est == [] and "hybrid model" in reason and len(reason) <= helpers._REASON_MAX
    assert helpers._preset_estimates(summary, 1_600_000_000) == []
    ok, why = helpers._preset_estimates_with_reason({"arch": "lfm2", "model": dict(BASE, attention_head_count_kv=LFM2_HEADS)},
                                                    1_600_000_000)
    assert ok and why == ""


def test_estimates_reason_without_a_gpu_backend(monkeypatch):
    from app import helpers, services
    monkeypatch.setattr(services, "_fit_backends", lambda: {})
    monkeypatch.setattr(services, "_cpu_backends", lambda: [])
    est, reason = helpers._preset_estimates_with_reason({"model": dict(BASE)}, 1)
    assert est == [] and "No GPU backend" in reason
    assert helpers._preset_estimates_with_reason({}, 1) == ([], "")


def test_search_repo_returns_estimates_reason(monkeypatch):
    from app import api, hf
    _fit(monkeypatch)
    detail = hf.HfRepoDetail(id="synthetic/d1-3B-GGUF", files=[
        hf.HfFile(path="d1-Q4_K_M.gguf", size=1_600_000_000, quant="Q4_K_M", shard_base="d1-Q4_K_M.gguf",
                  shard_index=None, shard_total=None)], readme_snippet=None)
    sample = {"value": {"_array": True, "count": 30, "sample": LFM2_HEADS[:8]}}

    async def repo_detail(repo):
        return detail

    async def header(repo, path):
        await asyncio.sleep(0)
        return {"arch": "lfm2", "model": dict(BASE, attention_head_count_kv=sample["value"])}
    monkeypatch.setattr(hf, "repo_detail", repo_detail)
    monkeypatch.setattr(hf, "gguf_header", header)
    monkeypatch.setattr(api.manager, "enqueue", lambda **kw: None)
    with TestClient(app) as c:
        g = c.get("/api/v1/search/repo", params={"repo": "synthetic/d1-3B-GGUF"}).json()["groups"][0]
        assert g["estimates"] == [] and "hybrid model" in g["estimatesReason"]
        sample["value"] = LFM2_HEADS
        g = c.get("/api/v1/search/repo", params={"repo": "synthetic/d1-3B-GGUF"}).json()["groups"][0]
        assert g["estimates"] and "estimatesReason" not in g


_BIN = __import__("os").environ.get("MODEL_AUTOCONFIG_BIN", "")


@pytest.mark.skipif(not _BIN, reason="MODEL_AUTOCONFIG_BIN not set: no model-autoconfig binary to compare")
@pytest.mark.parametrize("preset", ["fast", "balanced", "long-ctx"])
def test_rust_mode_gives_the_same_hybrid_plan_and_refusal(monkeypatch, preset):
    """MODEL_AUTOCONFIG=rust must keep agreeing: the attention-only KV shape (fewer shape layers
    than the stack) is a valid size request, and the prefix refusal is the same refusal."""
    import dataclasses
    from app import config
    for heads in (LFM2_HEADS, {"_array": True, "count": 30, "sample": LFM2_HEADS[:8]}):
        kw = dict(summary={"arch": "lfm2", "model": dict(BASE, attention_head_count_kv=heads)},
                  file_size=1_600_000_000, backends=BACKENDS, preset=preset)
        monkeypatch.setattr(config.settings, "model_autoconfig", "python")
        py = dataclasses.asdict(autoconfig.analyze(**kw))
        monkeypatch.setattr(config.settings, "model_autoconfig", "rust")
        monkeypatch.setattr(config.settings, "model_autoconfig_bin", _BIN)
        rs = dataclasses.asdict(autoconfig.analyze(**kw))
        assert rs == py
