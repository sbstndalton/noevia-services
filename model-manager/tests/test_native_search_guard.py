"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_SEARCH_CONFIRMED=1 the private Python child refuses
GET /api/v1/search and GET /api/v1/search/repo (the native parent answers them) and serves two private
calls that hand the parent only live state: `GET /api/v1/internal/search-runtime` (the GPU budget and
the downloaded files, as `api.search` computes them) and `POST /api/v1/internal/search-repo-runtime`
(the fit chips and context estimates of the groups the parent listed, as `api.search_repo` computes
them). Without the confirmation neither call exists (404) and Python keeps both routes.

Synthetic repositories, backends, chips and tokens only. FastAPI startup never runs; the Hub, Docker and
the GPU sampler are replaced by stand-ins, so no request leaves the process.
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import api, db, discover, hf, helpers, services
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_SEARCH_CONFIRMED"
TOKEN = "synthetic-search-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
SEARCH = "/api/v1/search"
REPO = "/api/v1/search/repo"
RUNTIME = "/api/v1/internal/search-runtime"
REPO_RUNTIME = "/api/v1/internal/search-repo-runtime"
GIB = 1024 ** 3


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(db, "_DB_PATH", None)
    db.init()
    monkeypatch.delenv(ENV, raising=False)

    async def no_hub(*_a, **_k):
        raise hf.HfSearchError("Could not reach Hugging Face. Check this server's connection and try again.")

    monkeypatch.setattr(hf, "search_models", no_hub)
    monkeypatch.setattr(api, "_trusted_quantisers", lambda: set())
    return TestClient(app, raise_server_exceptions=False)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_only_the_two_reads_after_the_token_gate(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for path in (SEARCH, REPO + "?repo=example-org/m", SEARCH + "?q=x&limit=3"):
        assert client.get(path).status_code == 401, path
        refused = client.get(path, headers=auth())
        assert refused.status_code == 503 and refused.content == REFUSED, path
    # Other methods keep FastAPI's 405 and the slash variants their redirect.
    for path in (SEARCH, REPO):
        assert client.post(path, headers=auth()).status_code == 405
        assert client.put(path, headers=auth()).status_code == 405
    for path in (SEARCH + "/", REPO + "/"):
        assert client.get(path, headers=auth(), follow_redirects=False).status_code == 307
    # The neighbouring routes are not touched.
    assert client.get("/api/v1/settings", headers=auth()).status_code == 200
    assert client.get("/api/v1/downloads", headers=auth()).status_code == 200
    # Percent-encoded spellings reach the same route in Starlette, so they are refused as well.
    assert client.get("/api/v1/%73earch", headers=auth()).content == REFUSED


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner_and_hides_the_runtime_calls(monkeypatch, client, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    answer = client.get(SEARCH + "?q=x", headers=auth())
    assert answer.status_code == 200 and answer.json()["results"] == []
    assert "Could not reach" in answer.json()["error"]
    assert client.get(RUNTIME, headers=auth()).status_code == 404
    assert client.post(REPO_RUNTIME, headers=auth(), json={}).status_code == 404


def test_the_search_guard_alone_leaves_the_other_native_guards_out(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for other in ("NOEVIA_NATIVE_MODEL_MANAGER_DRAFT_HEADS_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOADS_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DETAIL_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_CONFIRMED"):
        monkeypatch.delenv(other, raising=False)
    assert client.get("/api/v1/sections/x/draft-heads", headers=auth()).content != REFUSED
    assert client.get("/api/v1/models/detail?key=x", headers=auth()).content != REFUSED


def test_runtime_calls_are_behind_the_token_gate(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.get(RUNTIME).status_code == 401
    assert client.post(REPO_RUNTIME, json={}).status_code == 401


def test_search_runtime_is_the_budget_and_the_downloaded_map_api_search_computes(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    backends = [{"name": "a", "vram_gb": 12.0}, {"name": "b", "vram_gb": 24.5}, {"name": "cpu", "vram_gb": 0},
                {"name": "unset", "vram_gb": None}]
    downloaded = {"example-org/m-GGUF": ["m/m-Q4_K_M.gguf", "m/mmproj-m.gguf"], "example-org/ü": ["x.gguf"]}
    monkeypatch.setattr(helpers, "_backend_list", lambda: backends)
    monkeypatch.setattr(helpers, "_downloaded_and_still_present", lambda: downloaded)
    answer = client.get(RUNTIME, headers=auth())
    assert answer.status_code == 200 and answer.headers["content-type"] == "application/json"
    assert answer.json() == {"budget": "24.5", "downloaded": downloaded,
                             "quantisers": sorted(discover.TRUSTED_QUANTISERS), "labs": sorted(discover.TRUSTED_LABS)}
    # The sets are the module's, read at call time (a deployment that extends them is followed).
    monkeypatch.setattr(discover, "TRUSTED_QUANTISERS", {"b-quant", "a-quant"})
    monkeypatch.setattr(discover, "TRUSTED_LABS", {"z-lab"})
    changed = client.get(RUNTIME, headers=auth()).json()
    assert changed["quantisers"] == ["a-quant", "b-quant"] and changed["labs"] == ["z-lab"]
    assert client.post(RUNTIME, headers=auth()).status_code == 405
    # No backend: the budget is 0.0 (the search then judges against 1e6 GB). A non-finite figure
    # survives as Python's own repr.
    monkeypatch.setattr(helpers, "_backend_list", lambda: [])
    assert client.get(RUNTIME, headers=auth()).json()["budget"] == "0.0"
    monkeypatch.setattr(helpers, "_backend_list", lambda: [{"vram_gb": float("inf")}])
    assert client.get(RUNTIME, headers=auth()).json()["budget"] == "inf"
    # A failing collaborator is this app's own 500, as for the public route.
    monkeypatch.setattr(helpers, "_downloaded_and_still_present", lambda: 1 / 0)
    assert client.get(RUNTIME, headers=auth()).status_code == 500


def _estimates(summary, size, mm_gb=0.0):
    return ([{"key": "fast", "ctx": 4096, "size": size, "mm": mm_gb}], "" if size else "why")


@pytest.fixture()
def runtime_stubs(monkeypatch, client):
    chips = []

    def fit(size):
        chips.append(size)
        return [{"name": "synthetic-gpu", "vram_gb": 0.1 + 0.2, "verdict": "fits", "ratio_pct": size % 7}]

    calls = []

    def estimates(summary, size, mm_gb=0.0):
        calls.append((summary, size, mm_gb))
        return _estimates(summary, size, mm_gb)

    monkeypatch.setattr(services, "vram_fit_chips", fit)
    monkeypatch.setattr(helpers, "_preset_estimates_with_reason", estimates)
    monkeypatch.setenv(ENV, "1")
    client.chips, client.estimate_calls = chips, calls
    return client


def test_repo_runtime_answers_chips_estimates_and_native_context(runtime_stubs):
    client = runtime_stubs
    summary = {"model": {"context_length": 8192}, "architecture": "llama"}
    body = {"groups": [{"bytes": 5 * GIB, "estimate": True}, {"bytes": 7, "estimate": False},
                       {"bytes": 3, "estimate": True}], "mmprojBytes": GIB // 2, "summary": summary}
    answer = client.post(REPO_RUNTIME, headers=auth(), json=body)
    assert answer.status_code == 200 and answer.headers["content-type"] == "application/json"
    groups = answer.json()["groups"]
    assert [g["fit"][0]["ratio_pct"] for g in groups] == [5 * GIB % 7, 0, 3]
    assert "estimates" in groups[0] and "estimates" not in groups[1]
    assert groups[0]["nativeCtx"] == 8192 and groups[0]["reason"] == ""
    assert groups[2]["reason"] == "" and groups[2]["estimates"][0]["mm"] == 0.5
    # Exactly the calls api.search_repo makes: the header summary, the group's bytes and the
    # smallest projector in GiB (true division of the integers).
    assert client.estimate_calls == [(summary, 5 * GIB, 0.5), (summary, 3, 0.5)]
    # The text the parent splices is the text a FastAPI route would render.
    assert answer.content == json.dumps({"groups": groups}, ensure_ascii=False, separators=(",", ":")).encode()


def test_repo_runtime_without_summary_projector_or_native_context(runtime_stubs):
    client = runtime_stubs
    answer = client.post(REPO_RUNTIME, headers=auth(), json={"groups": [{"bytes": 9, "estimate": True}],
                                                              "mmprojBytes": None, "summary": None})
    assert answer.json() == {"groups": [{"fit": [{"name": "synthetic-gpu", "vram_gb": 0.30000000000000004,
                                                  "verdict": "fits", "ratio_pct": 2}]}]}
    assert client.estimate_calls == []
    for model in (None, {}, {"context_length": None}, {"context_length": 0}):
        summary = {"model": model}
        got = client.post(REPO_RUNTIME, headers=auth(), json={"groups": [{"bytes": 1, "estimate": True}],
                                                              "mmprojBytes": None, "summary": summary})
        assert got.json()["groups"][0]["nativeCtx"] == 0
    # A zero-byte projector is "a projector with no size", not "no projector".
    client.post(REPO_RUNTIME, headers=auth(), json={"groups": [{"bytes": 1, "estimate": True}],
                                                    "mmprojBytes": 0, "summary": {"a": 1}})
    assert client.estimate_calls[-1] == ({"a": 1}, 1, 0.0)


@pytest.mark.parametrize("body", [
    None, [], {}, {"groups": []}, {"groups": [], "mmprojBytes": None},
    {"groups": [], "mmprojBytes": None, "summary": None, "extra": 1},
    {"groups": {}, "mmprojBytes": None, "summary": None},
    {"groups": [1], "mmprojBytes": None, "summary": None},
    {"groups": [{"bytes": -1, "estimate": True}], "mmprojBytes": None, "summary": None},
    {"groups": [{"bytes": True, "estimate": True}], "mmprojBytes": None, "summary": None},
    {"groups": [{"bytes": 1.5, "estimate": True}], "mmprojBytes": None, "summary": None},
    {"groups": [{"bytes": 1, "estimate": 1}], "mmprojBytes": None, "summary": None},
    {"groups": [{"bytes": 1}], "mmprojBytes": None, "summary": None},
    {"groups": [], "mmprojBytes": -1, "summary": None},
    {"groups": [], "mmprojBytes": 1.0, "summary": None},
    {"groups": [], "mmprojBytes": None, "summary": []},
    {"groups": [{"bytes": 1, "estimate": True}] * 4097, "mmprojBytes": None, "summary": None},
])
def test_repo_runtime_validates_its_request(runtime_stubs, body):
    client = runtime_stubs
    answer = client.post(REPO_RUNTIME, headers=auth(), json=body)
    assert answer.status_code == 422 and answer.json() == {"detail": "invalid runtime request"}
    assert client.chips == [] and client.estimate_calls == []


def test_repo_runtime_rejects_other_methods_and_bodies(runtime_stubs):
    client = runtime_stubs
    assert client.get(REPO_RUNTIME, headers=auth()).status_code == 405
    for raw in (b"", b"{", b"\xff", b"NaN"):
        assert client.post(REPO_RUNTIME, headers=auth(), content=raw).status_code == 422
    full = {"groups": [{"bytes": 1, "estimate": True}] * 4096, "mmprojBytes": None, "summary": None}
    assert client.post(REPO_RUNTIME, headers=auth(), json=full).status_code == 200
    assert len(client.chips) == 4096


def test_repo_runtime_failure_is_this_apps_500(runtime_stubs, monkeypatch):
    client = runtime_stubs

    def boom(*_a):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(services, "vram_fit_chips", boom)
    answer = client.post(REPO_RUNTIME, headers=auth(), json={"groups": [{"bytes": 1, "estimate": False}],
                                                              "mmprojBytes": None, "summary": None})
    assert answer.status_code == 500
