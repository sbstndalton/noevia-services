"""#1197: a GGUF without general.parameter_count still has a parameter count: the sum of its
tensor-info element counts. Synthetic headers only (no model files, no weights)."""
from __future__ import annotations

import struct

from app import gguf_meta

U32, STRING, U64 = 4, 8, 10


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", STRING) + _s(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<II", U32, value)


def _kv_u64(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<IQ", U64, value)


def _kv_bool(key: str, value: bool) -> bytes:
    return _s(key) + struct.pack("<I?", 7, value)


def _info(name: str, dims: list[int], ggml_type: int = 0) -> bytes:
    """One tensor-info entry: name, n_dimensions, dimensions, ggml type, data offset."""
    return _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) \
        + struct.pack("<IQ", ggml_type, 0)


def _gguf(kvs: list[bytes], infos: list[bytes] = (), tensor_count: int | None = None) -> bytes:
    count = len(infos) if tensor_count is None else tensor_count
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", count)
            + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"".join(infos))


BASE = [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 1)]
TENSORS = [_info("token_embd.weight", [4096, 32000]), _info("blk.0.attn_q.weight", [4096, 4096]),
           _info("output_norm.weight", [4096])]
TENSOR_TOTAL = 4096 * 32000 + 4096 * 4096 + 4096  # 147,853,312


def _summary(buf: bytes) -> dict:
    return gguf_meta.summarize(gguf_meta.read_raw_bytes(buf))


def test_params_are_the_sum_of_tensor_element_counts_when_the_header_has_no_count():
    summary = _summary(_gguf(BASE, TENSORS))
    assert summary["general"]["params"] == gguf_meta._fmt_params(TENSOR_TOTAL) == "147.9 M"
    assert summary["general"]["params_raw"] == TENSOR_TOTAL


def test_a_declared_parameter_count_still_wins():
    summary = _summary(_gguf(BASE + [_kv_u64("general.parameter_count", 7_000_000_000)], TENSORS))
    assert summary["general"]["params"] == "7.0 B"
    assert summary["general"]["params_raw"] == 7_000_000_000


def test_a_zero_dimension_tensor_is_one_element_and_no_tensors_means_no_count():
    assert _summary(_gguf(BASE, [_info("scalar", [])]))["general"]["params"] == "1"
    assert _summary(_gguf(BASE, []))["general"]["params"] is None


def test_a_truncated_tensor_block_gives_no_count_and_no_crash():
    # Declares three tensors, carries only the first: a range-fetched header cut short.
    summary = _summary(_gguf(BASE, TENSORS[:1], tensor_count=3))
    assert summary["general"]["params"] is None
    assert summary["general"]["params_raw"] is None


def test_hostile_tensor_blocks_give_no_count():
    too_many = _gguf(BASE, [], tensor_count=2 ** 40)
    assert _summary(too_many)["general"]["params"] is None
    too_many_dims = _gguf(BASE, [_info("odd", [1] * 9)])
    assert _summary(too_many_dims)["general"]["params"] is None
    overflow = _gguf(BASE, [_info("huge", [2 ** 40, 2 ** 40, 2 ** 40])])
    assert _summary(overflow)["general"]["params"] is None


def test_the_file_path_and_the_byte_path_agree(tmp_path):
    buf = _gguf(BASE, TENSORS)
    path = tmp_path / "synthetic.gguf"
    path.write_bytes(buf)
    assert gguf_meta._read_raw(path) and gguf_meta.summarize(gguf_meta._read_raw(path))["general"]["params"] == "147.9 M"
    assert _summary(buf)["general"]["params"] == "147.9 M"


def test_a_dimension_past_2_to_the_53_is_refused_like_the_js_and_rust_readers():
    assert _summary(_gguf(BASE, [_info("wide", [2 ** 53])]))["general"]["params"] is None
    assert _summary(_gguf(BASE, [_info("edge", [2 ** 53 - 1])]))["general"]["params"] is not None


def test_a_boolean_declared_count_is_not_a_count():
    summary = _summary(_gguf(BASE + [_kv_bool("general.parameter_count", True)], TENSORS))
    assert summary["general"]["params"] == "147.9 M"
