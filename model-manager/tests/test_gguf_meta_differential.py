"""Differential test (#909): the real gguf-meta binary and the Python parser must agree on
synthetic GGUF headers. Runs only when GGUF_META_BIN points at a built gguf-meta (CI builds it
at release/versions.lock's NOEVIA_RS_REF); skipped otherwise. GGUF_META_FIXTURES may add a
directory of extra *.gguf files (noevia-rs's crates/gguf/tests/fixtures). Synthetic only."""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import pytest

from app import config, gguf_meta

BIN = os.environ.get("GGUF_META_BIN", "")
pytestmark = pytest.mark.skipif(not BIN, reason="GGUF_META_BIN not set: no gguf-meta binary to compare")

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f",
        BOOL: "<B", U64: "<Q", I64: "<q", F64: "<d"}


def _s(x: str | bytes) -> bytes:
    b = x.encode() if isinstance(x, str) else x
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, t: int, v) -> bytes:
    if t == STRING:
        return _s(key) + struct.pack("<I", STRING) + _s(v)
    return _s(key) + struct.pack("<I", t) + struct.pack(_FMT[t], v)


def _arr(key: str, sub: int, items: list, count: int | None = None) -> bytes:
    out = _s(key) + struct.pack("<I", ARRAY) + struct.pack("<IQ", sub, len(items) if count is None else count)
    for it in items:
        out += _s(it) if sub == STRING else struct.pack(_FMT[sub], it)
    return out


def _gguf(*pairs: bytes, kv_count: int | None = None, version: int = 3, tensors: int = 0) -> bytes:
    n = len(pairs) if kv_count is None else kv_count
    return b"GGUF" + struct.pack("<IQQ", version, tensors, n) + b"".join(pairs)


TEMPLATE = "{% if enable_thinking %}<think>{% endif %}{{ reasoning_effort }}<|channel|>thought"


def _cases() -> dict[str, bytes]:
    c = {
        "llama_full": _gguf(
            _kv("general.architecture", STRING, "llama"), _kv("general.name", STRING, "Synthetic 8B"),
            _kv("general.file_type", U32, 15), _kv("general.quantization_version", U32, 2),
            _kv("general.parameter_count", U64, 8_030_261_248),
            _kv("llama.context_length", U32, 131072), _kv("llama.embedding_length", U32, 4096),
            _kv("llama.block_count", U32, 32), _kv("llama.attention.head_count", U32, 32),
            _kv("llama.attention.head_count_kv", U32, 8), _kv("llama.rope.freq_base", F32, 500000.0),
            _kv("llama.rope.scaling.type", STRING, "yarn"), _kv("llama.rope.scaling.factor", F32, 8.0),
            _arr("tokenizer.ggml.tokens", STRING, [f"t{i}" for i in range(40)]),
            _kv("tokenizer.ggml.bos_token_id", U32, 1), _kv("tokenizer.ggml.unknown_token_id", I32, -1),
            _kv("tokenizer.ggml.add_bos_token", BOOL, 1), _kv("tokenizer.chat_template", STRING, TEMPLATE)),
        "gemma_swa_arrays": _gguf(
            _kv("general.architecture", STRING, "gemma3"), _kv("general.file_type", U32, 32),
            _arr("gemma3.attention.head_count_kv", U32, [4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 8]),
            _arr("gemma3.attention.sliding_window_pattern", BOOL, [1, 1, 1, 1, 1, 0]),
            _kv("gemma3.attention.sliding_window", U32, 1024),
            _arr("gemma3.feed_forward_length", U32, [7, 7, 9, 9, 9, 7, 3, 3, 3, 3])),
        "moe_mixed_types": _gguf(
            _kv("general.architecture", STRING, "qwen3moe"), _kv("qwen3moe.expert_count", U8, 128),
            _kv("qwen3moe.expert_used_count", U16, 8), _kv("qwen3moe.full_attention_interval", I8, -4),
            _kv("qwen3moe.ssm.inner_size", F64, 3072.9), _arr("qwen3moe.rope.freq_base", F32, [1.5, 2.5]),
            _kv("general.parameter_count", F64, 12345.678), _kv("general.file_type", F32, 15.0)),
        "zero_kv": _gguf(),
        "version_2": _gguf(_kv("general.architecture", STRING, "llama"), version=2, tensors=291),
        "truncated_value": _gguf(_kv("general.architecture", STRING, "llama"))[:-3],
        "range_cut_tokens": _gguf(
            _kv("general.architecture", STRING, "llama"),
            _arr("tokenizer.ggml.tokens", STRING, [f"tok{i}" for i in range(20)], count=150_000),
            _kv("llama.context_length", U32, 8192)),
        "implausible_kv_count": _gguf(kv_count=10 ** 9),
        "invalid_utf8": _gguf(_kv("general.architecture", STRING, "llama"),
                              _s("general.name") + struct.pack("<I", STRING) + _s(b"bad \xff\xfe name")),
        "nested_one_level": _gguf(_s("x.nested") + struct.pack("<IIQ", ARRAY, ARRAY, 2)
                                  + struct.pack("<IQ", U32, 1) + struct.pack("<I", 7)
                                  + struct.pack("<IQ", U32, 0)),
        "long_template": _gguf(_kv("tokenizer.chat_template", STRING, "x" * (gguf_meta.MAX_STRING_LEN + 10))),
        "bad_magic": b"NOPE" + b"\0" * 32,
        "short_header": b"GGUF\x03\x00",
        "nonfinite_floats": _gguf(
            _kv("general.architecture", STRING, "llama"), _kv("llama.rope.freq_base", F32, math.nan),
            _kv("llama.rope.scaling.factor", F64, math.inf)),
        # #1186: whole per-layer arrays share a per-header budget; past it head_count_kv keeps
        # only the sample (in both parsers), and the summary shows which.
        "per_layer_budget": _gguf(
            _kv("general.architecture", STRING, "lfm2"), _kv("lfm2.block_count", U32, 4096),
            *[_arr(f"lfm2.a{i:02d}", U8, [i % 7] * 4096) for i in range(66)],
            _arr("lfm2.attention.head_count_kv", U8, [8] * 4096),
            _arr("lfm2.feed_forward_length", U16, [3] * 4096)),
        "per_layer_within_budget": _gguf(
            _kv("general.architecture", STRING, "lfm2"), _kv("lfm2.block_count", U32, 4096),
            *[_arr(f"lfm2.a{i:02d}", U8, [1] * 4096) for i in range(63)],
            _arr("lfm2.attention.head_count_kv", U8, [1] + [8] * 4095)),
    }
    return c


def _fixture_files(tmp_path: Path) -> list[Path]:
    files = []
    for name, data in _cases().items():
        p = tmp_path / f"{name}.gguf"
        p.write_bytes(data)
        files.append(p)
    extra = os.environ.get("GGUF_META_FIXTURES")
    if extra:
        found = sorted(Path(extra).glob("*.gguf"))
        assert found, f"GGUF_META_FIXTURES={extra} has no .gguf files"
        files += found
    return files


def _python(path: Path):
    try:
        return "ok", gguf_meta.summarize(gguf_meta.read_raw(path))
    except Exception as e:  # noqa: BLE001 - the error class is what is compared
        return "error", type(e)


@pytest.fixture(autouse=True)
def _rust(monkeypatch):
    gguf_meta._LOGGED.clear()
    gguf_meta._RUST_CACHE.clear()
    monkeypatch.setattr(config.settings, "gguf_parser", "rust")
    monkeypatch.setattr(config.settings, "gguf_meta_bin", BIN)
    yield
    gguf_meta._RUST_CACHE.clear()


def test_binary_is_runnable():
    assert gguf_meta._rust_binary(), f"GGUF_META_BIN={BIN} is not an executable file"


def test_rust_and_python_summaries_agree(tmp_path):
    """Full parity: where both succeed the summaries are equal, and Rust fails exactly where
    the Python parser raises (noevia-rs#2 / #913 closed the last non-finite gap)."""
    mismatches, agreed = [], 0
    for path in _fixture_files(tmp_path):
        kind, py = _python(path)
        rust = gguf_meta._run_rust(path)
        if rust is not None and kind == "ok":
            if rust != py:
                mismatches.append(f"{path.name}:\n  rust:   {rust}\n  python: {py}")
            agreed += 1
        elif rust is not None:
            mismatches.append(f"{path.name}: rust ok but python raised {py.__name__}")
        elif kind == "ok":
            mismatches.append(f"{path.name}: rust failed but python succeeded")
    assert not mismatches, "\n".join(mismatches)
    assert agreed >= 8


def test_switch_end_to_end_matches_python_everywhere(tmp_path):
    """Through summarize_path (Rust, falling back to Python) every fixture gives exactly the
    Python result, errors included."""
    for path in _fixture_files(tmp_path):
        kind, py = _python(path)
        if kind == "ok":
            assert gguf_meta.summarize_path(path) == py, path.name
            assert gguf_meta.summarize_bytes(path.read_bytes()) == py, path.name
        else:
            with pytest.raises(py):
                gguf_meta.summarize_path(path)
            with pytest.raises(py):
                gguf_meta.summarize_bytes(path.read_bytes())
