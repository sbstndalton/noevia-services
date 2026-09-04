"""Shared helpers: HTTP client factory, token estimation, atomic local writes."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import httpx


def make_client(
    base_url: Optional[str] = None,
    timeout_s: float = 300.0,
    auth: Optional[tuple] = None,
    headers: Optional[dict] = None,
) -> httpx.Client:
    """Shared httpx.Client factory — connection pooling, explicit timeouts.

    trust_env=False so proxy env vars can't silently divert LAN traffic.
    Requests uncompressed responses (Accept-Encoding: identity): Apache-side gzip
    makes Nextcloud return compression-variant ETags ("...-gzip") whose If-Match
    comparisons then fail with spurious 412s on every conditional write.
    """
    merged = {"Accept-Encoding": "identity"}
    if headers:
        merged.update(headers)
    return httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(timeout_s),
        auth=auth,
        headers=merged,
        trust_env=False,
        follow_redirects=True,
    )


def estimate_tokens(text: str) -> int:
    """Fast, dependency-free token estimate. ~3.6 chars/token for English prose.

    Over-estimates a little; used only for context budgeting, never for truncation
    of user-visible content.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))


def atomic_write(path: Path, data: bytes) -> None:
    """Atomic local-file write: temp file in same dir + fsync + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def read_text_file(path: Path) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        return None


_JSON_DEFAULTS: dict = {}


def to_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)
