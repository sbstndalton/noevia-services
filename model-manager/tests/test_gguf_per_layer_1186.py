"""noevia#1186: the GGUF header reader kept only the first 8 entries of every array, so a hybrid
model's per-layer attention.head_count_kv (lfm2: 30 layers, 0 on the short-conv layers) reached
autoconfig as a prefix and autoconfig refused with kv_layers. A top-level numeric array whose
length equals the header's own <arch>.block_count (at most MAX_PER_LAYER_KEPT) is now kept whole;
every other long array keeps the 8-entry sample and its count. Synthetic headers only."""
from __future__ import annotations

import struct

import pytest

from app import autoconfig, autoconfig_core, gguf_meta

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f",
        BOOL: "<B", U64: "<Q", I64: "<q", F64: "<d"}
HEADS = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0]
BACKENDS = [{"name": "llama-cuda", "vram_gb": 16.0, "gpu_count": 1, "host_ram_gb": 32.0, "baseline": {}}]


def _s(x: str) -> bytes:
    return struct.pack("<Q", len(x.encode())) + x.encode()


def _kv(key: str, t: int, v) -> bytes:
    if t == STRING:
        return _s(key) + struct.pack("<I", STRING) + _s(v)
    return _s(key) + struct.pack("<I", t) + struct.pack(_FMT[t], v)


def _arr(key: str, sub: int, items: list, count: int | None = None) -> bytes:
    out = _s(key) + struct.pack("<I", ARRAY) + struct.pack("<IQ", sub, len(items) if count is None else count)
    return out + b"".join(_s(i) if sub == STRING else struct.pack(_FMT[sub], i) for i in items)


def _gguf(*pairs: bytes) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)) + b"".join(pairs)


def _lfm2(*extra: bytes, block_count: int = 30, bc_type: int = U32) -> bytes:
    return _gguf(_kv("general.architecture", STRING, "lfm2"), _kv("lfm2.block_count", bc_type, block_count),
                 _kv("lfm2.attention.head_count", U32, 32), _kv("lfm2.embedding_length", U32, 2048),
                 _kv("lfm2.context_length", U32, 32768), *extra)


def test_per_layer_head_count_kv_is_kept_whole_and_autoconfig_sizes_it():
    raw = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.attention.head_count_kv", U32, HEADS),
                                         _arr("tokenizer.ggml.token_type", I32, [1] * 30 + [2])))
    assert raw["lfm2.attention.head_count_kv"] == HEADS
    assert raw["tokenizer.ggml.token_type"] == {"_array": True, "count": 31, "sample": [1] * 8}
    summary = gguf_meta.summarize(raw)
    assert summary["model"]["attention_head_count_kv"] == HEADS
    p = autoconfig_core.prepare({"n_sessions": 1, "arch": "lfm2", "model": summary["model"],
                                 "file_size": 1_600_000_000, "backends": BACKENDS})
    assert p["refuse"] is None and p["shape"]["layers"] == sum(1 for h in HEADS if h)
    assert p["shape"]["recurrent_bytes"] == HEADS.count(0) * autoconfig_core._SSM_STATE_BYTES
    rec = autoconfig.analyze(summary=summary, file_size=1_600_000_000, backends=BACKENDS, preset="fast")
    assert not rec.error and rec.recommended_ctx > 0


@pytest.mark.parametrize("sub,items", [(F32, [0.5] * 30), (I64, list(range(30))), (U8, [1] * 30)])
def test_every_numeric_type_of_block_count_length_is_kept(sub, items):
    raw = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.x", sub, items)))
    assert raw["lfm2.x"] == items


def test_the_bound_is_max_per_layer_kept():
    n = gguf_meta.MAX_PER_LAYER_KEPT
    at = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.x", U8, [1] * n), block_count=n))
    assert at["lfm2.x"] == [1] * n
    over = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.x", U8, [1] * (n + 1)), block_count=n + 1, bc_type=U64))
    assert over["lfm2.x"] == {"_array": True, "count": n + 1, "sample": [1] * 8}


@pytest.mark.parametrize("data", [
    _lfm2(_arr("lfm2.x", U32, HEADS[:29])),                    # length differs from block_count
    _lfm2(_arr("lfm2.x", BOOL, [1] * 30)),                     # bool is not a per-layer number
    _gguf(_kv("general.architecture", STRING, "lfm2"),         # block_count only after the array
          _arr("lfm2.x", U32, HEADS), _kv("lfm2.block_count", U32, 30)),
    _gguf(_kv("general.architecture", STRING, "lfm2"),         # another arch's block_count
          _kv("llama.block_count", U32, 30), _arr("lfm2.x", U32, HEADS)),
    _gguf(_kv("lfm2.block_count", U32, 30), _arr("lfm2.x", U32, HEADS)),  # no architecture
    _lfm2(_arr("lfm2.x", U32, HEADS), block_count=30.0, bc_type=F32),     # non-int block_count
    _lfm2(_arr("lfm2.x", U32, HEADS), block_count=1, bc_type=BOOL),
])
def test_other_long_arrays_keep_the_sample(data):
    v = gguf_meta.read_raw_bytes(data)["lfm2.x"]
    assert isinstance(v, dict) and v["_array"] and len(v["sample"]) == 8


def test_strings_of_block_count_length_keep_the_sample():
    raw = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.x", STRING, [f"l{i}" for i in range(30)])))
    assert raw["lfm2.x"]["count"] == 30 and len(raw["lfm2.x"]["sample"]) == 8


def test_a_cut_off_per_layer_array_keeps_count_and_sample():
    raw = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.attention.head_count_kv", U32, HEADS[:12], count=30)))
    assert raw["lfm2.attention.head_count_kv"] == {"_array": True, "count": 30, "sample": HEADS[:8]}
    assert raw["_error"] == "stopped at KV read: array runs past the end of the data"


def test_whole_lists_read_like_summaries_for_scalar_fields():
    raw = gguf_meta.read_raw_bytes(_lfm2(
        _arr("lfm2.attention.head_count", U32, [1] + [8] * 29),
        _arr("lfm2.feed_forward_length", U32, [5] * 4 + [9] * 13 + [7] * 13),  # tie: first seen
        _arr("lfm2.attention.key_length", I32, [64, 128] * 15)))
    m = gguf_meta.summarize(raw)["model"]
    assert m["attention_head_count"] == 8 and m["feed_forward_length"] == 9
    # a KV dimension takes the largest entry: never below the sample's most common value
    assert m["key_length"] == 128
    # lists no longer than a sample keep their first element, as before
    assert gguf_meta._scalar_int([1] + [8] * 7) == 1


@pytest.mark.parametrize("heads", [
    [1] + [8] * 29,                                       # one odd layer must not shrink KV
    [2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8] + [8] * 14,  # OpenELM-style increasing
])
def test_whole_kv_list_never_sizes_below_the_old_sample(heads):
    old = autoconfig_core._kv_first_int({"_array": True, "count": len(heads), "sample": heads[:8]})
    new = autoconfig_core._kv_first_int(heads)
    assert new == max(heads) >= old
    model = {"block_count": 30, "attention_head_count": 32, "embedding_length": 2048,
             "context_length": 32768, "attention_head_count_kv": heads}
    p = autoconfig_core.prepare({"n_sessions": 1, "arch": "llama", "model": model,
                                 "file_size": 1_600_000_000, "backends": BACKENDS})
    assert p["refuse"] is None and p["shape"]["kv_heads"] == max(heads) >= old


def test_kv_list_nulls_and_short_lists():
    assert autoconfig_core._kv_first_int([None, 2] * 5) == 2
    assert autoconfig_core._kv_first_int([None] * 9, default=5) == 5
    assert autoconfig_core._kv_first_int([1] + [8] * 7) == 1  # eight entries: unchanged


def test_whole_arrays_share_a_per_header_budget():
    n = gguf_meta.MAX_PER_LAYER_KEPT
    whole = gguf_meta.MAX_PER_LAYER_VALUES // n
    data = _lfm2(*[_arr(f"lfm2.a{i:02d}", U8, [1] * n) for i in range(whole + 2)],
                 _arr("lfm2.attention.head_count_kv", U8, [8] * n), block_count=n)
    raw = gguf_meta.read_raw_bytes(data)
    assert all(raw[f"lfm2.a{i:02d}"] == [1] * n for i in range(whole))
    for k in (f"lfm2.a{whole:02d}", "lfm2.attention.head_count_kv"):
        assert raw[k] == {"_array": True, "count": n, "sample": raw[k]["sample"]} and len(raw[k]["sample"]) == 8
    assert "_error" not in raw


def test_a_hostile_header_of_many_per_layer_arrays_stays_cheap():
    import time
    n = gguf_meta.MAX_PER_LAYER_KEPT
    data = _lfm2(*[_arr(f"k{i}", U8, [1] * n) for i in range(2000)], block_count=n)
    t = time.monotonic()
    raw = gguf_meta.read_raw_bytes(data)
    assert time.monotonic() - t < 5
    assert sum(len(v) for v in raw.values() if isinstance(v, list)) == gguf_meta.MAX_PER_LAYER_VALUES


def _plan_bytes(model: dict, ctx: int = 32768) -> int:
    p = autoconfig_core.prepare({"n_sessions": 1, "arch": "llama", "model": model,
                                 "file_size": 1_600_000_000, "backends": BACKENDS})
    assert p["refuse"] is None
    return autoconfig_core.kv_shape_bytes(p["shape"], ctx, 1.0625)


def test_summary_carries_the_old_sample_reading_and_kv_takes_the_larger():
    """Lead decision on #1186: KV from a whole list is never below the old 8-sample reading."""
    raw = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.attention.head_count", U32, [8] * 8 + [32] * 22),
                                         _kv("lfm2.attention.head_count_kv", U32, 8)))
    m = gguf_meta.summarize(raw)["model"]
    # mode of the whole list is 32 (head_dim 64); the old sample's mode was 8 (head_dim 256)
    assert m["attention_head_count"] == 32 and m["per_layer_sample"] == {"attention_head_count": 8}
    old = dict({k: v for k, v in m.items() if not k.startswith("per_layer")}, attention_head_count=8)
    new = {k: v for k, v in m.items() if not k.startswith("per_layer")}
    assert _plan_bytes(m) == max(_plan_bytes(old), _plan_bytes(new)) > _plan_bytes(new)


def test_a_zero_kv_dim_entry_never_goes_below_head_dim():
    raw = gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.attention.key_length", U32, [0] * 20 + [8] * 10)))
    m = gguf_meta.summarize(raw)["model"]
    assert m["key_length"] == 8 and m["per_layer_zero_dims"] == ["key_length"]
    p = autoconfig_core.prepare({"n_sessions": 1, "arch": "lfm2", "model": m,
                                 "file_size": 1_600_000_000, "backends": BACKENDS})
    assert p["shape"]["k_dim"] == 2048 // 32  # head_dim, not 8


def test_summaries_without_whole_lists_are_unchanged():
    m = gguf_meta.summarize(gguf_meta.read_raw_bytes(_lfm2(_arr("lfm2.x", U32, [1] * 29))))["model"]
    assert not any(k.startswith("per_layer") for k in m)
