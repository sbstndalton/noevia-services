"""Shared helpers: HTTP client factory and token estimation."""
from __future__ import annotations

from typing import Optional

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
