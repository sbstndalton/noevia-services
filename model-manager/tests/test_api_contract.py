"""The /api/v1 routes the app really serves must match docs/spec-model-loader-api-v1.md (#269).

Adding or removing a route without updating the contract doc fails here. The web side of the
same contract is checked by apps/web/server/model-loader-contract.test.cjs.
"""
import re
from collections import deque
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app import api, hw, main, services
from app.config import settings
from app.main import app

DOC = Path(__file__).resolve().parents[3] / "docs" / "spec-model-loader-api-v1.md"


def _norm(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path)


def _documented() -> set[str]:
    block = re.search(r"<!-- route-table:start -->(.*?)<!-- route-table:end -->", DOC.read_text(), re.S)
    assert block, "route table markers missing from the contract doc"
    rows = set()
    for line in block.group(1).splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 5 and cells[1] in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
            rows.add(f"{cells[1]} {_norm(cells[2])}")
    return rows


def _served() -> set[str]:
    out = set()
    for r in app.routes:
        if isinstance(r, APIRoute) and r.path.startswith("/api/v1"):
            out.update(f"{m} {_norm(r.path)}" for m in r.methods if m not in {"HEAD", "OPTIONS"})
    return out


def test_doc_lists_every_served_route_and_nothing_else():
    served, doc = _served(), _documented()
    assert len(served) > 30
    assert not served - doc, f"served but undocumented: {sorted(served - doc)}"
    assert not doc - served, f"documented but not served: {sorted(doc - served)}"


def test_only_health_is_unauthenticated(monkeypatch):
    monkeypatch.setattr(settings, "model_loader_token", "fixture-token")
    with TestClient(app) as c:
        assert c.get("/api/v1/health").status_code == 200
        for path in ("/api/v1/models", "/api/v1/sections"):
            r = c.get(path)
            assert r.status_code == 401 and r.json() == {"error": "unauthorized"}
        assert c.put("/api/v1/models-ini", json={"baseRevision": "x", "text": "a=1"}).status_code == 401
        assert c.get("/api/v1/sections", headers={"X-Model-Loader-Token": "fixture-token"}).status_code == 200


def test_native_owned_paths_refuse_without_python_fallback(monkeypatch):
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED", "1")
    monkeypatch.setattr(settings, "model_loader_token", "fixture-token")
    with TestClient(app) as c:
        assert c.get("/api/v1/health").status_code == 503
        assert c.get("/api/v1/backends/synthetic/logs").status_code == 401
        for method in (c.get, c.post):
            assert method("/api/v1/backends/synthetic/logs", headers={"X-Model-Loader-Token": "fixture-token"}).status_code == 503
        assert c.get("/api/v1/sections", headers={"X-Model-Loader-Token": "fixture-token"}).status_code == 200


# Synthetic host ownership checks: no proc, Docker, startup or engine actions.
@pytest.mark.parametrize("confirmed,expected", [(None, 1), ("0", 1), ("1", 0)])
def test_backend_sampler_skips_only_confirmed_native_host(monkeypatch, confirmed, expected):
    if confirmed is None:
        monkeypatch.delenv("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED", raising=False)
    else:
        monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED", confirmed)
    calls = []
    monkeypatch.setattr(hw, "_read_host", lambda: calls.append("host"))
    monkeypatch.setattr(services, "_effective_container_names", lambda: ["synthetic-engine"])
    monkeypatch.setattr(hw, "_collect", lambda name: calls.append(name) or hw.BackendStats(ok=False))
    monkeypatch.setattr(hw, "_record_history", lambda *args: None)
    monkeypatch.setattr(hw, "_CACHE", {})
    class EndTick(Exception):
        pass
    monkeypatch.setattr(hw.time, "sleep", lambda _: (_ for _ in ()).throw(EndTick()))
    with pytest.raises(EndTick):
        hw._sample_loop()
    assert calls.count("host") == expected
    assert calls.count("synthetic-engine") == 1


def test_confirmed_host_refuses_direct_python_after_token_gate(monkeypatch):
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED", "1")
    monkeypatch.setattr(main.settings, "model_loader_token", "synthetic-host-token")
    monkeypatch.setattr(hw, "host_history", lambda: (_ for _ in ()).throw(AssertionError("Python host response must not run")))
    # No context manager: FastAPI startup and hardware/model tasks are never run.
    client = TestClient(main.app)
    assert client.get("/api/v1/host").status_code == 401
    assert client.get("/api/v1/host", headers={"x-model-loader-token": "synthetic-host-token"}).status_code == 503


def test_default_python_host_contract_remains_available(monkeypatch):
    monkeypatch.delenv("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED", raising=False)
    monkeypatch.setattr(main.settings, "model_loader_token", "")
    monkeypatch.setattr(hw, "_HOST_HISTORY", deque([hw.HostPoint(float(i), 0.0, 1.0, 2.0, 1.0) for i in range(450)], maxlen=450))
    answer = TestClient(main.app).get("/api/v1/host")
    assert answer.status_code == 200
    assert answer.json() == api.host()
    assert len(answer.json()["history"]) == 180
    assert answer.json()["current"]["ts"] == 449.0
def test_native_sections_guard_is_authenticated_and_method_aware(monkeypatch):
    import asyncio
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from app.main import _require_token

    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED", "1")
    monkeypatch.setattr(settings, "model_loader_token", "fixture-token")
    calls = []

    async def downstream(request):
        calls.append((request.method, request.url.path))
        return JSONResponse({"fixture": "python-owner"})

    def invoke(method, path, token="fixture-token"):
        request = Request({"type": "http", "method": method, "path": path,
                           "query_string": b"", "headers": [(b"x-model-loader-token", token.encode())],
                           "scheme": "http", "server": ("fixture", 80)})
        return asyncio.run(_require_token(request, downstream))

    for path in ("/api/v1/sections", "/api/v1/sections/tiny"):
        assert invoke("GET", path, "wrong").status_code == 401
        response = invoke("GET", path)
        assert response.status_code == 503
        assert response.body == b'{"detail":"Native model manager owns this endpoint"}'
        assert not calls
    for method, path in (("PUT", "/api/v1/sections/tiny"), ("DELETE", "/api/v1/sections/tiny"),
                         ("HEAD", "/api/v1/sections/tiny"), ("POST", "/api/v1/sections/tiny/rename"),
                         ("GET", "/api/v1/sections/tiny/autoconfig"), ("GET", "/api/v1/sections/")):
        assert invoke(method, path).status_code == 200
    assert len(calls) == 6
    monkeypatch.delenv("NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED")
    assert invoke("GET", "/api/v1/sections").status_code == 200
