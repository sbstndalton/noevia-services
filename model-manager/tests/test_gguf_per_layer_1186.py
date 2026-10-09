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
