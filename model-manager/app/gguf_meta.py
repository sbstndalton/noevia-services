"""Minimal GGUF v3 metadata reader. Zero deps beyond stdlib."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

# GGUFValueType from gguf spec
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY = 6, 7, 8, 9
_UINT64, _INT64, _FLOAT64 = 10, 11, 12

_SCALAR_FMT: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8),
    _FLOAT32: ("<f", 4), _FLOAT64: ("<d", 8),
    _BOOL: ("<?", 1),
}

MAX_ARRAY_ELEMENTS_KEPT = 8
MAX_STRING_LEN = 200_000
# Keys are short identifiers ("llama.attention.head_count"); the GGUF spec itself caps them at 65535.
MAX_KEY_LEN = 65_535
# Real files carry tens to low thousands of KV pairs; this stops a multi-GB file of empty pairs
# building a multi-GB dict.
MAX_KV_COUNT = 100_000
# Total string characters kept across all values (chat templates run to tens of KB).
MAX_RETAINED_CHARS = 16_000_000
# One array nested directly inside another is accepted; deeper is rejected.
MAX_ARRAY_DEPTH = 1
# A top-level numeric array whose length equals the header's own `<arch>.block_count` is a
# per-layer value (hybrid models list attention.head_count_kv per layer, 0 on recurrent
# layers) and is kept whole instead of as an 8-element sample, so autoconfig can see every
# layer (sbstndalton/noevia#1186). Only when block_count is at most this; worst case is this
# many numbers per such key.
MAX_PER_LAYER_KEPT = 4096
_NUMERIC_TYPES = frozenset(
    (_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _UINT64, _INT64, _FLOAT64)
)

# subset of llama.cpp LlamaFileType — enough to name every real-world GGUF quant
FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M",
    32: "BF16", 33: "Q4_0_4_4", 34: "Q4_0_4_8", 35: "Q4_0_8_8",
    36: "TQ1_0", 37: "TQ2_0",
}


class GgufMetaError(Exception):
    pass


_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


class _Source:
    """A seekable stream plus how many bytes are left in it.

    Every length, count and skip in a GGUF header is attacker-controlled: it is read from a
    file the user dropped in the models directory or from a remote repo's first megabyte. The
    parser therefore never trusts one without comparing it with what the stream can actually
    still supply, and never hands one straight to read() or seek() (a multi-GB length makes a
    buffered reader reserve that much memory; a value >= 2**63 raises OverflowError).
    """

    def __init__(self, f) -> None:
        self.f = f
        start = f.tell()
        self.size = f.seek(0, 2)
        f.seek(start)
        self.retained = 0     # characters of string data kept so far, across the whole header
        self.ran_out = False  # an array declared more elements than the stream holds

    def run_out(self) -> None:
        """An array's elements outrun the stream. Normal for a range-fetched header, whose big
        tokenizer arrays are cut off at the first megabyte, so the array keeps its count and
        sample; the parse just ends here (and says so) rather than guessing at what follows."""
        self.ran_out = True
        self.f.seek(0, 2)

    def remaining(self) -> int:
        return max(0, self.size - self.f.tell())

    def read(self, n: int) -> bytes:
        if n < 0 or n > self.remaining():
            raise GgufMetaError("header is truncated or declares a length past the end of the data")
        return self.f.read(n)

    def skip(self, n: int) -> None:
        if n < 0 or n > self.remaining():
            raise GgufMetaError("header is truncated or declares a length past the end of the data")
        self.f.seek(n, 1)

    def unpack(self, fmt: str, size: int):
        return struct.unpack(fmt, self.read(size))[0]


def _read_string(src: _Source, *, max_len: int = MAX_STRING_LEN) -> str:
    n = src.unpack("<Q", 8)
    if n > src.remaining():
        raise GgufMetaError(f"string length {n} runs past the end of the data")
    if n > max_len:
        # Keep the head, step over the rest: reading all n bytes would pull an arbitrarily
        # large run of the file into memory just to throw nearly all of it away.
        chunk = src.read(max_len)
        src.skip(n - max_len)
        text = chunk.decode("utf-8", errors="replace") + "…[truncated]"
    else:
        text = src.read(n).decode("utf-8", errors="replace")
    src.retained += len(text)
    if src.retained > MAX_RETAINED_CHARS:
        raise GgufMetaError("header holds implausibly much string data")
    return text


def _skip_string(src: _Source) -> None:
    n = src.unpack("<Q", 8)
    src.skip(n)


def _per_layer_len(out: dict[str, Any]) -> int | None:
    """The header's block_count when it was already read and is a plausible layer count:
    an int (not a bool) under `<general.architecture>.block_count`, 1..MAX_PER_LAYER_KEPT."""
    arch = out.get("general.architecture")
    if not isinstance(arch, str):
        return None
    n = out.get(f"{arch}.block_count")
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_PER_LAYER_KEPT:
        return None
    return n


def _read_value(src: _Source, vtype: int, depth: int = 0, per_layer: int | None = None):
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        return src.unpack(fmt, size)
    if vtype == _STRING:
        return _read_string(src)
    if vtype == _ARRAY:
        subtype = src.unpack("<I", 4)
        count = src.unpack("<Q", 8)
        if subtype == _ARRAY and depth >= MAX_ARRAY_DEPTH:
            # Arrays of arrays of arrays: nothing real emits them, and each level multiplies
            # what has to be walked - an attacker's header would recurse without bound.
            raise GgufMetaError("arrays nested more than one level deep are not supported")
        if subtype not in _SCALAR_FMT and subtype not in (_STRING, _ARRAY):
            raise GgufMetaError(f"unknown array element type {subtype}")
        keep_whole = (
            depth == 0 and per_layer is not None and count == per_layer
            and subtype in _NUMERIC_TYPES
            # A cut-off header keeps the old count + sample (and run_out) rather than failing.
            and _SCALAR_FMT[subtype][1] * count <= src.remaining()
        )
        if count > MAX_ARRAY_ELEMENTS_KEPT and not keep_whole:
            if subtype == _ARRAY:
                raise GgufMetaError(f"unsupported nested array subtype {subtype}")
            sample = [_read_value(src, subtype, depth + 1) for _ in range(MAX_ARRAY_ELEMENTS_KEPT)]
            remaining = count - MAX_ARRAY_ELEMENTS_KEPT
            if subtype == _STRING:
                try:
                    for _ in range(remaining):
                        _skip_string(src)
                except GgufMetaError:
                    src.run_out()
            else:
                _, size = _SCALAR_FMT[subtype]
                if size * remaining > src.remaining():
                    src.run_out()
                else:
                    src.skip(size * remaining)
            return {"_array": True, "count": count, "sample": sample}
        return [_read_value(src, subtype, depth + 1) for _ in range(count)]
    raise GgufMetaError(f"unknown value type {vtype}")


def _read_raw_stream(f) -> dict[str, Any]:
    """Parse GGUF metadata from any seekable binary stream.

    Split out from _read_raw so the same parser can run over a range-fetched header
    (io.BytesIO) as well as a local file — metadata lives at the very start of a GGUF,
    so the first ~1 MB is enough to read every KV pair without pulling the weights.

    Hostile or corrupt input never raises out of here past the magic check: a bad length,
    count or nesting depth is recorded as `_error` and parsing stops with whatever was read.
    """
    src = _Source(f)
    magic = src.f.read(4)
    if magic != b"GGUF":
        raise GgufMetaError(f"not a GGUF file (magic={magic!r})")
    version = src.unpack("<I", 4)
    tensor_count = src.unpack("<Q", 8)
    kv_count = src.unpack("<Q", 8)
    out: dict[str, Any] = {
        "_gguf_version": version,
        "_tensor_count": tensor_count,
        "_kv_count": kv_count,
    }
    if kv_count > MAX_KV_COUNT:
        out["_error"] = f"stopped at KV read: implausible kv_count {kv_count}"
        return out
    try:
        for _ in range(kv_count):
            key = _read_string(src, max_len=MAX_KEY_LEN)
            vtype = src.unpack("<I", 4)
            out[key] = _read_value(src, vtype, per_layer=_per_layer_len(out))
    except (GgufMetaError, struct.error, OSError, OverflowError, MemoryError, RecursionError) as e:
        out["_error"] = f"stopped at KV read: {e or type(e).__name__}"
    if src.ran_out and "_error" not in out:
        out["_error"] = "stopped at KV read: array runs past the end of the data"
    return out


def _read_raw(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return _read_raw_stream(f)


def read_raw_bytes(buf: bytes) -> dict[str, Any]:
    """Parse GGUF metadata out of an in-memory buffer (e.g. a range-fetched header)."""
    import io
    return _read_raw_stream(io.BytesIO(buf))


def read_raw(path: Path) -> dict[str, Any]:
    st = path.stat()
    key = f"{path}|{st.st_mtime_ns}|{st.st_size}"
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached:
            return cached
    raw = _read_raw(path)
    with _CACHE_LOCK:
        _CACHE[key] = raw
    return raw


def _scalar_int(v: Any) -> int | None:
    """Unwrap array-summary dicts / lists to a representative int, or None if not resolvable.
    Some archs (Gemma, MoE variants) encode per-layer values as arrays; the config-panel and
    autoconfig want a scalar. Pick the most-common value from the sample."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, dict) and v.get("_array"):
        sample = v.get("sample") or []
        if sample:
            counts: dict[int, int] = {}
            for x in sample:
                if isinstance(x, (int, float)):
                    counts[int(x)] = counts.get(int(x), 0) + 1
            if counts:
                return max(counts, key=lambda k: counts[k])
    if isinstance(v, list) and v:
        first = v[0]
        if isinstance(first, (int, float)):
            return int(first)
    return None


def _scalar_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    n = _scalar_int(v)
    return float(n) if n is not None else None


def scan_chat_template_features(template: str) -> dict:
    """Detect reasoning/thinking-related capabilities of a Jinja chat template.
    Everything is checked against the raw template source — no execution."""
    if not template or not isinstance(template, str):
        return {}
    t = template.lower()
    return {
        # kwargs the template accepts (found as variable references)
        "accepts_enable_thinking": "enable_thinking" in t,
        "accepts_reasoning_effort": "reasoning_effort" in t,
        "accepts_preserve_thinking": "preserve_thinking" in t,
        # output markers — knowing these tells us reasoning-format should be 'deepseek' so
        # OpenAI-compatible clients (OpenWebUI etc.) can hide the thinking trace nicely
        "uses_think_tags": "<think>" in t,
        "uses_channel_thought": "channel>thought" in t or "channel|>thought" in t or "<|channel|>thought" in t,
    }


def _fmt_params(n: Any) -> str | None:
    if not isinstance(n, (int, float)) or n <= 0:
        return None
    n = float(n)
    if n >= 1e12:
        return f"{n / 1e12:.1f} T"
    if n >= 1e9:
        return f"{n / 1e9:.1f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} K"
    return str(int(n))


# Architectures whose rope scaling llama.cpp applies per layer from the GGUF itself. A global
# --rope-scale would also scale the layers that must stay unscaled (Gemma 3's sliding-window
# local layers), so a preset must never carry rope keys for them, declared metadata or not.
# rope_owned_by_gguf lives in autoconfig_core (pure, stdlib-only, so MODEL_AUTOCONFIG=rust's
# values assembly can be checked against it); re-exported here under its old name.
from .autoconfig_core import _ROPE_FROM_GGUF_ARCHS, rope_owned_by_gguf  # noqa: E402,F401


def _finite(v: Any) -> Any:
    """Copy of a parsed value with every NaN/+-Infinity float replaced by None (#901).

    GGUF float metadata is attacker-controlled bytes, so it can be non-finite. Python's json
    would emit NaN/Infinity, which Starlette's JSONResponse (allow_nan=False) refuses with a
    500, and int(nan)/int(inf) in the scalar helpers raise. null is what the Rust port emits."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_finite(x) for x in v]
    return v


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    raw = _finite(raw)
    arch = raw.get("general.architecture", "") or ""

    def a(key: str, default: Any = None) -> Any:
        return raw.get(f"{arch}.{key}", default) if arch else default

    file_type = raw.get("general.file_type")
    quant_name = FILE_TYPE_NAMES.get(file_type, str(file_type) if file_type is not None else None)

    vocab = raw.get("tokenizer.ggml.tokens")
    if isinstance(vocab, dict) and vocab.get("_array"):
        vocab_size = vocab.get("count")
    else:
        vocab_size = a("vocab_size")

    return {
        "arch": arch,
        "general": {
            "name": raw.get("general.name"),
            "description": raw.get("general.description"),
            "author": raw.get("general.author"),
            "license": raw.get("general.license"),
            "url": raw.get("general.url"),
            "quant": quant_name,
            "quant_version": raw.get("general.quantization_version"),
            "params": _fmt_params(raw.get("general.parameter_count")),
            "params_raw": raw.get("general.parameter_count"),
            "gguf_version": raw.get("_gguf_version"),
            "tensor_count": raw.get("_tensor_count"),
            "kv_count": raw.get("_kv_count"),
            # Set when the header was cut short or malformed: the rest of the summary is then
            # whatever parsed before the fault, not the whole story.
            "header_error": raw.get("_error"),
        },
        "model": {
            "arch": arch,
            "context_length": _scalar_int(a("context_length")),
            "embedding_length": _scalar_int(a("embedding_length")),
            "block_count": _scalar_int(a("block_count")),
            "feed_forward_length": _scalar_int(a("feed_forward_length")),
            "attention_head_count": _scalar_int(a("attention.head_count")),
            # kept raw — autoconfig has _kv_first_int() that samples the array intelligently
            "attention_head_count_kv": a("attention.head_count_kv"),
            "rope_freq_base": _scalar_float(a("rope.freq_base")),
            "rope_scaling_type": a("rope.scaling.type"),
            "rope_scaling_factor": _scalar_float(a("rope.scaling.factor")),
            "rope_scaling_original_context": _scalar_int(a("rope.scaling.original_context_length")),
            "vocab_size": _scalar_int(vocab_size),
            "expert_count": _scalar_int(a("expert_count")),
            # Built-in multi-token-prediction layers (llama.cpp "nextn"): draft-mtp needs no head file.
            "nextn_predict_layers": _scalar_int(a("nextn_predict_layers")),
            "expert_used_count": _scalar_int(a("expert_used_count")),
            "key_length": _scalar_int(a("attention.key_length")),
            "value_length": _scalar_int(a("attention.value_length")),
            "full_attention_interval": _scalar_int(a("full_attention_interval")),
            # Sliding-window attention, declared properly rather than guessed. Gemma-4 sets
            # all four: most layers attend over a short window and use a NARROWER head dim
            # than the global layers, and some layers share another layer's KV entirely.
            # Sizing without these over-estimates the cache by an order of magnitude.
            "sliding_window": _scalar_int(a("attention.sliding_window")),
            "key_length_swa": _scalar_int(a("attention.key_length_swa")),
            "value_length_swa": _scalar_int(a("attention.value_length_swa")),
            "shared_kv_layers": _scalar_int(a("attention.shared_kv_layers")),
            # kept raw: a per-layer array of which layers are local vs global
            "sliding_window_pattern": a("attention.sliding_window_pattern"),
            "ssm_state_size": _scalar_int(a("ssm.state_size")),
            "ssm_inner_size": _scalar_int(a("ssm.inner_size")),
            # Mamba conv window and B/C group count: with state and inner size they give the
            # per-sequence recurrent state llama.cpp allocates for an SSM layer (#1159).
            "ssm_conv_kernel": _scalar_int(a("ssm.conv_kernel")),
            "ssm_group_count": _scalar_int(a("ssm.group_count")),
        },
        "tokenizer": {
            "model": raw.get("tokenizer.ggml.model"),
            "pre": raw.get("tokenizer.ggml.pre"),
            "bos_token_id": raw.get("tokenizer.ggml.bos_token_id"),
            "eos_token_id": raw.get("tokenizer.ggml.eos_token_id"),
            "unknown_token_id": raw.get("tokenizer.ggml.unknown_token_id"),
            "padding_token_id": raw.get("tokenizer.ggml.padding_token_id"),
            "add_bos_token": raw.get("tokenizer.ggml.add_bos_token"),
            "add_eos_token": raw.get("tokenizer.ggml.add_eos_token"),
        },
        "chat_template": raw.get("tokenizer.chat_template"),
        "chat_template_features": scan_chat_template_features(raw.get("tokenizer.chat_template") or ""),
    }



# --- Parser switch (#909) ---------------------------------------------------------------------
# GGUF_PARSER=rust runs the bounded Rust reader from sbstndalton/noevia-rs (`gguf-meta <path>`,
# baked into the image at release/versions.lock's NOEVIA_RS_REF) instead of the Python parser
# above. It ships dark: the default is python, and every Rust failure (missing binary, nonzero
# exit, timeout, oversized or malformed output) falls back to the Python parser, so callers see
# exactly the Python behaviour, errors included. Only summaries go through Rust; read_raw()
# (raw key/value access, e.g. services.model_shape) stays Python.

PARSERS = ("python", "rust")
GGUF_META_TIMEOUT_S = 10.0
GGUF_META_STDOUT_CAP = 4 * 1024 * 1024
# Always objects in a summary ("arch" is copied from the file, so it may be any JSON type).
_SUMMARY_KEYS = ("general", "model", "tokenizer")

_log = logging.getLogger(__name__)
_LOGGED: set[str] = set()
_LOGGED_LOCK = threading.Lock()
# (path|mtime|size) -> Rust summary, or None when Rust failed on that version of the file (so a
# file Rust cannot read is not re-spawned on every page render; Python answers instead).
_RUST_CACHE: dict[str, dict[str, Any] | None] = {}


def _log_once(reason: str, message: str) -> None:
    with _LOGGED_LOCK:
        if reason in _LOGGED:
            return
        _LOGGED.add(reason)
    _log.warning(message)


def _setting(name: str, env: str, default: str) -> str:
    # gguf_meta.py is also loaded straight from its file by noevia-rs's fixture generator, where
    # the package (and so .config) is absent: fall back to the environment there.
    try:
        from .config import settings
        return str(getattr(settings, name, default) or "")
    except ImportError:
        return os.environ.get(env, default)


def parser_choice() -> str:
    """The configured GGUF parser: "python" (default) or "rust". Anything else is python."""
    value = _setting("gguf_parser", "GGUF_PARSER", "python").strip().lower() or "python"
    if value not in PARSERS:
        _log_once("invalid_setting",
                  f"GGUF_PARSER={value!r} is not one of {', '.join(PARSERS)}; using python")
        return "python"
    return value


def _rust_binary() -> str | None:
    configured = _setting("gguf_meta_bin", "GGUF_META_BIN", "gguf-meta").strip() or "gguf-meta"
    if os.sep in configured:
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    return shutil.which(configured)


def _rust_fail(reason: str, detail: str) -> None:
    _log_once(f"rust:{reason}", f"gguf-meta failed ({reason}: {detail}); using the Python GGUF parser")


def _run_rust(path: Path) -> dict[str, Any] | None:
    """Summarise `path` with the gguf-meta binary, or None (after logging once per reason) when
    it cannot. No shell: the path is one argv element. Output is capped and time-limited."""
    binary = _rust_binary()
    if binary is None:
        _rust_fail("missing_binary", "gguf-meta not found or not executable")
        return None
    try:
        proc = subprocess.Popen(
            [binary, os.fspath(Path(path).absolute())],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            close_fds=True, shell=False,
        )
    except OSError as e:
        _rust_fail("spawn_error", str(e))
        return None
    buf = bytearray()

    def pump() -> None:
        try:
            while len(buf) <= GGUF_META_STDOUT_CAP:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
        except (OSError, ValueError):
            pass

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    reader.join(GGUF_META_TIMEOUT_S)
    try:
        if reader.is_alive():
            _rust_fail("timeout", f"no result within {GGUF_META_TIMEOUT_S:g} s")
            return None
        if len(buf) > GGUF_META_STDOUT_CAP:
            _rust_fail("output_too_large", f"more than {GGUF_META_STDOUT_CAP} bytes on stdout")
            return None
        try:
            code = proc.wait(timeout=GGUF_META_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            _rust_fail("timeout", "process did not exit after closing stdout")
            return None
        if code != 0:
            _rust_fail("nonzero_exit", f"exit status {code}")
            return None
        try:
            summary = json.loads(bytes(buf).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as e:
            _rust_fail("bad_json", type(e).__name__)
            return None
        if not isinstance(summary, dict) or "arch" not in summary \
                or not all(isinstance(summary.get(k), dict) for k in _SUMMARY_KEYS):
            _rust_fail("bad_json", "not a summary object")
            return None
        # Same sanitiser as the Python path (#901): Rust already emits null, this is belt and braces.
        return _finite(summary)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        reader.join(1)
        if proc.stdout:
            proc.stdout.close()


def summarize_path(path: Path) -> dict[str, Any]:
    """summarize(read_raw(path)) through the configured parser. The single entry point every
    file summary goes through; raises what the Python parser raises."""
    path = Path(path)
    if parser_choice() == "rust":
        st = path.stat()
        key = f"{path}|{st.st_mtime_ns}|{st.st_size}"
        with _CACHE_LOCK:
            hit = key in _RUST_CACHE
            cached = _RUST_CACHE.get(key)
        if not hit:
            cached = _run_rust(path)
            with _CACHE_LOCK:
                _RUST_CACHE[key] = cached
        if cached is not None:
            return copy.deepcopy(cached)
    return summarize(read_raw(path))


def summarize_bytes(buf: bytes) -> dict[str, Any]:
    """summarize(read_raw_bytes(buf)) through the configured parser (e.g. a range-fetched
    remote header). With rust the bytes go through a private temp file."""
    if parser_choice() == "rust":
        summary = None
        try:
            fd, tmp = tempfile.mkstemp(prefix="gguf-head-", suffix=".gguf")
        except OSError as e:
            _rust_fail("tempfile", str(e))
        else:
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(buf)
                summary = _run_rust(Path(tmp))
            except OSError as e:
                _rust_fail("tempfile", str(e))
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        if summary is not None:
            return summary
    return summarize(read_raw_bytes(buf))
