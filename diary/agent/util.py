"""Shared helpers: HTTP client factory and token estimation."""
from __future__ import annotations

from typing import Optional

import httpx


def make_client(
    base_url: Optional[str] = None,
    timeout_s: float = 300.0,
    auth: Optional[tuple] = None,
    headers: Optional[dict] = None,
    follow_redirects: bool = False,
) -> httpx.Client:
    """Shared httpx.Client factory — connection pooling, explicit timeouts.

    trust_env=False so proxy env vars can't silently divert LAN traffic.
    Requests uncompressed responses (Accept-Encoding: identity): Apache-side gzip
    makes Nextcloud return compression-variant ETags ("...-gzip") whose If-Match
    comparisons then fail with spurious 412s on every conditional write.

    follow_redirects defaults to False: a server the user configures must not be
    able to bounce a request inward (RFC1918, cloud metadata) and have us follow.
    Configure such a server by its final URL instead. Callers that talk to an
    operator-controlled endpoint (not user-supplied) may opt back in.
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
        follow_redirects=follow_redirects,
    )


def ensure_not_redirect(resp) -> None:
    """Refuse 3xx responses from user-configured storage endpoints.

    Clients are built with follow_redirects=False (see make_client), and httpx
    then *returns* the 3xx response instead of raising — while raise_for_status()
    treats 3xx as success. Without this check a redirecting server would feed
    empty content (or an empty listing) into the corpus as if it were real.
    """
    if 300 <= resp.status_code < 400:
        raise RuntimeError(
            f"storage endpoint redirected (HTTP {resp.status_code}) — configure the "
            "server by its final URL instead; redirects are refused for security"
        )


def estimate_tokens(text: str) -> int:
    """Fast, dependency-free token estimate. ~3.6 chars/token for English prose.

    Over-estimates a little; used only for context budgeting, never for truncation
    of user-visible content.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.6))
