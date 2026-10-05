"""gguf_header must never buffer more than the header window, even when the server ignores
Range and answers 200 with the whole file (#874 item 3). Mocked HTTP only."""
from __future__ import annotations

import struct

import httpx
import pytest

from app import hf


def _gguf(block_count: int = 32) -> bytes:
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 2)
            + s("general.architecture") + struct.pack("<I", 8) + s("llama")
            + s("llama.block_count") + struct.pack("<II", 4, block_count))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(hf, "get_token", lambda: "")
    hf._HEADER_CACHE.clear()
    hf._HEADER_GATED.clear()


REAL_CLIENT = httpx.AsyncClient


def _install(monkeypatch, handler):
    real = REAL_CLIENT

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)
    monkeypatch.setattr(hf.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_a_server_that_ignores_range_is_read_only_up_to_the_cap(monkeypatch):
    served = {"bytes": 0}
    chunk = 64 * 1024

    async def body():
        yield _gguf()
        served["bytes"] += len(_gguf())
        # Endless "weights": the client must stop reading long before this ends.
        for _ in range(100_000):                       # ~6.5 GB if fully consumed
            served["bytes"] += chunk
            yield b"\0" * chunk

    _install(monkeypatch, lambda r: httpx.Response(200, content=body()))
    summary = await hf.gguf_header("o/r", "m.gguf")
    assert summary["model"]["block_count"] == 32
    assert served["bytes"] <= hf._HEADER_BYTES + 4 * chunk, served


@pytest.mark.asyncio
async def test_partial_content_and_gated_responses_still_behave(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(206, content=_gguf(12)))
    assert (await hf.gguf_header("o/ok", "m.gguf"))["model"]["block_count"] == 12

    _install(monkeypatch, lambda r: httpx.Response(401))
    assert await hf.gguf_header("o/gated", "m.gguf") is None
    assert "token" in hf.gated_reason("o/gated")

    _install(monkeypatch, lambda r: httpx.Response(404))
    assert await hf.gguf_header("o/missing", "m.gguf") is None

    _install(monkeypatch, lambda r: httpx.Response(200, content=b"not a gguf"))
    assert await hf.gguf_header("o/junk", "m.gguf") is None
