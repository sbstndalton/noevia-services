"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_BACKEND_CONTROL_CONFIRMED=1 the private Python child
refuses the plain-name backend restart/test the native parent answers, keeps every other shape, and
serves the parent's private call that records `last_restart_error` for GET /backends.
No Docker, container or model is touched.
"""
import pytest
from fastapi.testclient import TestClient

from app import services
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_BACKEND_CONTROL_CONFIRMED"
TOKEN = "synthetic-backend-control-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
CALL = "/api/v1/internal/backend-restart-error"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(services, "_effective_container_names", lambda: ["llama-a"])
    calls = []
    monkeypatch.setattr(services, "restart_llama_backend", lambda n: calls.append(n) or (True, "restart queued"))
    services._restart_errors.clear()
    c = TestClient(app, raise_server_exceptions=False)
    c.calls = calls
    return c


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_native_shapes_only(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.post("/api/v1/backends/llama-a/restart").status_code == 401
    for path in ("/api/v1/backends/llama-a/restart", "/api/v1/backends/llama-a/test"):
        answer = client.post(path, headers=auth(), json={"prompt": "x"})
        assert answer.status_code == 503 and answer.content == REFUSED
    assert client.calls == []
    # An escaped name stays here (the parent forwards it).
    assert client.post("/api/v1/backends/llama%2Da/restart", headers=auth()).json() == {"ok": True, "message": "restart queued"}


def test_restart_error_call_feeds_get_backends_state(monkeypatch, client):
    assert client.put(CALL, headers=auth(), json={"name": "llama-a", "error": "x"}).status_code == 404
    monkeypatch.setenv(ENV, "1")
    assert client.put(CALL, json={"name": "a", "error": None}).status_code == 401
    assert client.put(CALL, headers=auth(), json={"name": "llama-a", "error": "APIError: synthetic"}).json() == {"ok": True}
    assert services._restart_errors == {"llama-a": "APIError: synthetic"}
    assert client.put(CALL, headers=auth(), json={"name": "llama-a", "error": None}).status_code == 200
    assert services._restart_errors == {}
    for bad in ({}, {"name": 5}, {"name": "a", "error": 3}):
        assert client.put(CALL, headers=auth(), json=bad).status_code == 422
    assert client.get(CALL, headers=auth()).status_code == 405


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert client.post("/api/v1/backends/llama-a/restart", headers=auth()).json() == {"ok": True, "message": "restart queued"}
