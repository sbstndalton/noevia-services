"""Hugging Face search: the query the hub actually receives, and the fallbacks.

These pin the two behaviours that made the Download tab look broken — a browse with no
query returning nothing, and a named model returning nothing because its repo is not
tagged `gguf` — without touching the network.
"""
import httpx
import pytest

from app import hf


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    """These tests are about the query, not about credential storage."""
    monkeypatch.setattr(hf, "get_token", lambda: "")


def _install(monkeypatch, handler):
    """Point search_models at an in-process hub and record the requests it makes."""
    seen: list[httpx.Request] = []

    def _record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_record)
        return real(*args, **kwargs)

    monkeypatch.setattr(hf.httpx, "AsyncClient", factory)
    return seen


def _model(repo_id: str, **extra) -> dict:
    return {"id": repo_id, "downloads": 10, "likes": 1, "lastModified": "2026-01-01T00:00:00.000Z",
            "pipeline_tag": "text-generation", "tags": ["gguf"], **extra}


def _ok(payload) -> httpx.Response:
    return httpx.Response(200, json=payload)


@pytest.mark.asyncio
async def test_browse_with_no_query_asks_for_gguf_and_returns_results(monkeypatch):
    seen = _install(monkeypatch, lambda r: _ok([_model("owner/a-GGUF")]))
    out = await hf.search_models("", sort="downloads", limit=5)
    assert [m.id for m in out] == ["owner/a-GGUF"]
    params = seen[0].url.params
    assert params["filter"] == "gguf" and params["limit"] == "5"
    assert params["sort"] == "downloads" and params["direction"] == "-1"
    assert "search" not in params


@pytest.mark.asyncio
async def test_search_never_asks_for_full(monkeypatch):
    """`full=true` made the hub serialise every sibling of every hit and was the reason an
    ordinary search timed out. gguf_count going unknown is the accepted trade."""
    seen = _install(monkeypatch, lambda r: _ok([_model("owner/a-GGUF")]))
    out = await hf.search_models("qwen")
    assert "full" not in seen[0].url.params
    assert out[0].gguf_count is None


@pytest.mark.asyncio
async def test_untagged_repo_is_found_by_a_second_pass(monkeypatch):
    """The hub's `gguf` tag is not applied to every repo that holds GGUF files. A named
    search that the tag filter answers with nothing is retried without it."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("filter") == "gguf":
            return _ok([])
        return _ok([_model("someone/Private-Mix-GGUF", tags=["text-generation"]),
                    _model("someone/unrelated-safetensors", tags=["text-generation"])])

    seen = _install(monkeypatch, handler)
    out = await hf.search_models("private mix")
    assert [m.id for m in out] == ["someone/Private-Mix-GGUF"]
    assert len(seen) == 2 and "filter" not in seen[1].url.params


@pytest.mark.asyncio
async def test_second_pass_keeps_everything_when_nothing_looks_gguf(monkeypatch):
    """Better to show what the hub matched than to claim there are no results."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("filter") == "gguf":
            return _ok([])
        return _ok([_model("someone/plain", tags=["text-generation"])])

    _install(monkeypatch, handler)
    assert [m.id for m in await hf.search_models("plain")] == ["someone/plain"]


@pytest.mark.asyncio
async def test_browse_does_not_fall_back_to_untagged(monkeypatch):
    """With no query there is nothing to disambiguate, so an untagged top-30 would just be
    the hub's most-downloaded models — not a GGUF browse."""
    seen = _install(monkeypatch, lambda r: _ok([]))
    assert await hf.search_models("") == []
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_unknown_sort_is_normalised(monkeypatch):
    seen = _install(monkeypatch, lambda r: _ok([]))
    await hf.search_models("", sort="; drop table")
    assert seen[0].url.params["sort"] == "downloads"


@pytest.mark.asyncio
async def test_rate_limit_explains_the_token(monkeypatch):
    _install(monkeypatch, lambda r: httpx.Response(429, text="slow down"))
    with pytest.raises(hf.HfSearchError) as e:
        await hf.search_models("qwen")
    assert "429" in str(e.value) and "token" in str(e.value)


@pytest.mark.asyncio
async def test_network_failure_is_reported_as_one(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    _install(monkeypatch, handler)
    with pytest.raises(hf.HfSearchError) as e:
        await hf.search_models("qwen")
    assert "Could not reach Hugging Face" in str(e.value)
