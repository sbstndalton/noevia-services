"""#1940: _free_gpu must wait for the backend restart to complete before it polls the router,
otherwise the old backend (still up) answers 200 and the sweep starts while it is restarting.
A refused or failed restart is reported and the GPUs are not claimed clear. Fake docker client."""
from __future__ import annotations

import pytest

from app import bench, services


class _Resp:
    status_code = 200


class FakeBackend:
    def __init__(self, fail=None):
        self.restarted = False
        self.fail = fail

    def restart(self, timeout=None):
        if self.fail:
            raise self.fail
        self.restarted = True


class FakeContainers:
    def __init__(self, c):
        self.c = c

    def get(self, name):
        return self.c


class FakeClient:
    def __init__(self, c):
        self.containers = FakeContainers(c)


@pytest.fixture(autouse=True)
def _idle(monkeypatch):
    bench._STATE = bench.JobState()
    bench._CANCEL.clear()
    monkeypatch.setattr(bench.time, "sleep", lambda _s: None)
    monkeypatch.setattr(bench, "loaded_aliases", lambda _u: {"alias"})
    monkeypatch.setattr(services, "_effective_container_names", lambda: ["backend-1"])
    services._restart_errors.pop("backend-1", None)
    yield
    services._restart_errors.pop("backend-1", None)
    bench._STATE = bench.JobState()


def _lines():
    return list(bench.state().lines)


def test_a_refused_restart_is_logged_and_the_router_is_never_polled(monkeypatch):
    monkeypatch.setattr(services, "_effective_container_names", lambda: [])  # not configured
    polled = []
    monkeypatch.setattr(bench.httpx, "get", lambda *a, **k: polled.append(a) or _Resp())
    bench._free_gpu("backend-1", "http://backend-1:8080")
    assert any("could not restart backend-1: container not in configured list" in ln
               for ln in _lines())
    assert polled == []
    assert not any("GPUs clear" in ln for ln in _lines())


def test_a_failed_restart_is_logged_recorded_and_the_router_is_never_polled(monkeypatch):
    backend = FakeBackend(fail=RuntimeError("daemon went away"))
    monkeypatch.setattr(services, "_docker_client", lambda: FakeClient(backend))
    polled = []
    monkeypatch.setattr(bench.httpx, "get", lambda *a, **k: polled.append(a) or _Resp())
    bench._free_gpu("backend-1", "http://backend-1:8080")
    assert any("could not restart backend-1: RuntimeError: daemon went away" in ln
               for ln in _lines())
    assert services._restart_errors["backend-1"] == "RuntimeError: daemon went away"
    assert polled == []


def test_the_restart_has_happened_before_the_router_is_first_polled(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(services, "_docker_client", lambda: FakeClient(backend))
    seen = []

    def fake_get(url, timeout=None):
        seen.append(backend.restarted)
        return _Resp()

    monkeypatch.setattr(bench.httpx, "get", fake_get)
    bench._free_gpu("backend-1", "http://backend-1:8080")
    assert seen == [True]
    assert any("GPUs clear" in ln for ln in _lines())


def test_the_default_restart_stays_fire_and_forget(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(services, "_docker_client", lambda: FakeClient(backend))
    started = []

    class _T:
        def __init__(self, target=None, **kw):
            self.target = target

        def start(self):
            started.append(self.target)

    monkeypatch.setattr(services.threading, "Thread", _T)
    assert services.restart_llama_backend("backend-1") == (True, "restart queued")
    assert len(started) == 1 and not backend.restarted   # queued, not run inline
    started[0]()
    assert backend.restarted
