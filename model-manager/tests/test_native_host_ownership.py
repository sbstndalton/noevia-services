"""Synthetic host ownership checks: no proc, Docker, startup or engine actions."""
from collections import deque

import pytest
from fastapi.testclient import TestClient
from app import api, hw, main, services


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
