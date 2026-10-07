"""Hugging Face repo tree listing -> model file entries (#964).

Every field of a repository's tree listing comes from the network: file paths (which become
shard groups, quant labels and, later, download destinations) and sizes (which feed fit
estimates and the size check after a download). `files_from_tree` turns that listing into the
file list the rest of the service uses.

MODEL_FILES_IMPL=rust runs the same function in the bounded Rust leaf from sbstndalton/noevia-rs
(`model-files tree`, baked into the image at release/versions.lock's NOEVIA_RS_REF). It ships
dark: the default is python. Unlike the GGUF parser switch, the Rust path FAILS CLOSED: a missing
binary, nonzero exit, timeout, oversized input or malformed output raises ModelFilesError (the
API answers 502) instead of quietly using Python, so an operator who turned Rust on is never
served an unchecked listing.

The pure Python reference (`files_from_tree_py`) is the behaviour before #964, unchanged; the
differential fixtures in both repos are generated from it. This module imports only the stdlib
and .utils at load time, so noevia-rs's fixture generator can import it without the service's
dependencies.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
from typing import Any

from .utils import shard_key

IMPLS = ("python", "rust")
MODEL_FILES_TIMEOUT_S = 10.0
# The Rust CLI refuses more than this on stdin; refusing here first gives a clearer error.
MODEL_FILES_STDIN_CAP = 16 * 1024 * 1024
MODEL_FILES_STDOUT_CAP = 32 * 1024 * 1024

_log = logging.getLogger(__name__)
_LOGGED: set[str] = set()
_LOGGED_LOCK = threading.Lock()

_QUANT_RE = re.compile(r"\b(I?Q\d+(?:_[A-Z0-9]+)*|F16|F32|BF16|FP8|FP4)\b", re.IGNORECASE)
_FILE_KEYS = ("path", "size", "quant", "shard_base", "shard_index", "shard_total")


class ModelFilesError(RuntimeError):
    """MODEL_FILES_IMPL=rust could not produce a checked file list (fail closed)."""


def infer_quant(filename: str) -> str | None:
    m = _QUANT_RE.search(filename)
    return m.group(0).upper() if m else None


def is_support_file(path: str) -> bool:
    low = path.lower()
    # things llama.cpp sometimes needs alongside a GGUF
    return low.endswith((".mmproj", "mmproj.gguf", "chat_template.jinja", "tokenizer.model"))


def files_from_tree_py(entries: list[Any]) -> list[dict[str, Any]]:
    """The reference: GGUF and support files of a tree listing, in listing order."""
    files: list[dict[str, Any]] = []
    for e in entries:
        if e.get("type") != "file":
            continue
        path = e.get("path", "")
        if not path.lower().endswith(".gguf") and not is_support_file(path):
            continue
        # HF tree entries: size is under "size" for direct files; LFS files have "lfs": {"size": ...}
        size = int((e.get("lfs") or {}).get("size") or e.get("size") or 0)
        base, idx, tot = shard_key(path)
        files.append({"path": path, "size": size, "quant": infer_quant(path),
                      "shard_base": base, "shard_index": idx, "shard_total": tot})
    return files


def _log_once(reason: str, message: str) -> None:
    with _LOGGED_LOCK:
        if reason in _LOGGED:
            return
        _LOGGED.add(reason)
    _log.warning(message)


def _setting(name: str, env: str, default: str) -> str:
    try:
        from .config import settings
        return str(getattr(settings, name, default) or "")
    except ImportError:
        return os.environ.get(env, default)


def impl_choice() -> str:
    """The configured implementation: "python" (default) or "rust". Anything else is python."""
    value = _setting("model_files_impl", "MODEL_FILES_IMPL", "python").strip().lower() or "python"
    if value not in IMPLS:
        _log_once("invalid_setting",
                  f"MODEL_FILES_IMPL={value!r} is not one of {', '.join(IMPLS)}; using python")
        return "python"
    return value


def _rust_binary() -> str | None:
    configured = _setting("model_files_bin", "MODEL_FILES_BIN", "model-files").strip() or "model-files"
    if os.sep in configured:
        return configured if os.path.isfile(configured) and os.access(configured, os.X_OK) else None
    return shutil.which(configured)


def _fail(reason: str, detail: str) -> ModelFilesError:
    _log_once(f"rust:{reason}", f"model-files failed ({reason}: {detail}); refusing the listing")
    return ModelFilesError(f"model-files {reason}")


def _project(e: Any) -> Any:
    """Drop the fields the function never reads (commit messages, oids, security scans) so the
    Rust side's string and size caps apply only to what matters. Faithful: a non-object stays
    as it is (both sides reject it), and so does a non-object lfs."""
    if not isinstance(e, dict):
        return e
    out = {k: e[k] for k in ("type", "path", "size") if k in e}
    if "lfs" in e:
        lfs = e["lfs"]
        out["lfs"] = {"size": lfs["size"]} if isinstance(lfs, dict) and "size" in lfs else (
            {} if isinstance(lfs, dict) else lfs)
    return out


def _valid_file(f: Any) -> bool:
    if not isinstance(f, dict) or tuple(sorted(f)) != tuple(sorted(_FILE_KEYS)):
        return False
    ints_or_none = all(f[k] is None or (isinstance(f[k], int) and not isinstance(f[k], bool))
                       for k in ("shard_index", "shard_total"))
    return (isinstance(f["path"], str) and isinstance(f["shard_base"], str)
            and isinstance(f["size"], int) and not isinstance(f["size"], bool)
            and (f["quant"] is None or isinstance(f["quant"], str)) and ints_or_none)


def files_from_tree_rust(entries: list[Any]) -> list[dict[str, Any]]:
    binary = _rust_binary()
    if binary is None:
        raise _fail("missing_binary", "model-files not found or not executable")
    try:
        payload = json.dumps([_project(e) for e in entries], ensure_ascii=True).encode()
    except (TypeError, ValueError) as e:
        raise _fail("unencodable_input", type(e).__name__) from None
    if len(payload) > MODEL_FILES_STDIN_CAP:
        raise _fail("input_too_large", f"{len(payload)} bytes")
    try:
        proc = subprocess.run([binary, "tree"], input=payload, capture_output=True,
                              timeout=MODEL_FILES_TIMEOUT_S, check=False,
                              # A minimal environment: the child needs nothing of ours (tokens,
                              # settings), only a PATH.
                              env={"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")})
    except subprocess.TimeoutExpired:
        raise _fail("timeout", f"no result within {MODEL_FILES_TIMEOUT_S:g} s") from None
    except OSError as e:
        raise _fail("spawn", type(e).__name__) from None
    if proc.returncode != 0:
        detail = proc.stderr[:300].decode("utf-8", "replace").strip()
        raise _fail("rejected", f"exit {proc.returncode}: {detail}")
    if len(proc.stdout) > MODEL_FILES_STDOUT_CAP:
        raise _fail("output_too_large", f"more than {MODEL_FILES_STDOUT_CAP} bytes")
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        raise _fail("malformed_output", "not JSON") from None
    files = out.get("files") if isinstance(out, dict) else None
    if not isinstance(files, list) or len(files) > len(entries) or not all(map(_valid_file, files)):
        raise _fail("malformed_output", "unexpected shape")
    return files


def files_from_tree(entries: list[Any]) -> list[dict[str, Any]]:
    """The file list of a tree listing, by the configured implementation."""
    if impl_choice() == "rust":
        return files_from_tree_rust(entries)
    return files_from_tree_py(entries)
