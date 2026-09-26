"""Issue #341: an internal-network-only llama.cpp container publishes no port
(NetworkSettings.Ports = {}, Config.ExposedPorts = null), so the probe used to skip and
report (None, None) with no error -- Hardware then showed "no model loaded" even while the
engine was running fine. The loader must find the real port (from Docker's port map, then
--port/env, then a configured default) and, when it truly can't, say so via probe_error
instead of pretending an unknown state is a known one.
"""
import asyncio

import httpx
import pytest

from app import services
from app.config import settings


def _attrs(cmd=None, env=None, ports=None, exposed=None):
    return {
        "NetworkSettings": {"Ports": ports or {}},
        "Config": {"Cmd": cmd or [], "Env": env or [], "ExposedPorts": exposed},
    }


# ---------- port resolution ----------

def test_extract_ports_still_reads_the_docker_port_map():
    attrs = _attrs(ports={"8080/tcp": [{"HostPort": "8082"}]})
    host_ports, internal = services._extract_ports(attrs)
    assert internal == 8080 and host_ports == ["8082->8080/tcp"]


def test_no_port_map_falls_back_to_the_containers_own_dash_dash_port_flag():
    attrs = _attrs(cmd=["/server", "--host", "0.0.0.0", "--port", "8099", "-m", "model.gguf"])
    host_ports, internal = services._resolve_internal_port(attrs)
    assert internal == 8099 and host_ports == []


def test_no_port_map_falls_back_to_llama_arg_port_env():
    attrs = _attrs(env=["PATH=/usr/bin", "LLAMA_ARG_PORT=9001"])
    _, internal = services._resolve_internal_port(attrs)
    assert internal == 9001


def test_no_port_map_and_no_cmd_or_env_falls_back_to_configured_default(monkeypatch):
    monkeypatch.setattr(settings, "llama_default_port", 8080)
    attrs = _attrs()
    _, internal = services._resolve_internal_port(attrs)
    assert internal == 8080  # llama.cpp server's own default


def test_default_port_fallback_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "llama_default_port", 0)
    _, internal = services._resolve_internal_port(_attrs())
    assert internal is None


# ---------- _probe_loaded_model ----------

class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Stands in for httpx.AsyncClient(...) as an async context manager."""
    def __init__(self, *a, response=None, raises=None, **kw):
        self._response = response
        self._raises = raises

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, *a, **kw):
        if self._raises is not None:
            raise self._raises
        return self._response


def _mock_client(monkeypatch, *, response=None, raises=None):
    monkeypatch.setattr(
        services.httpx, "AsyncClient",
        lambda *a, **kw: _FakeClient(*a, response=response, raises=raises, **kw),
    )


def test_probe_with_no_resolvable_port_sets_port_unknown_and_never_returns_none_none():
    loaded, err = asyncio.run(services._probe_loaded_model("cowork-llama-1", None))
    assert loaded is None
    assert err == "port unknown"
    assert (loaded, err) != (None, None)


def test_probe_refused_sets_a_sanitized_probe_error(monkeypatch):
    _mock_client(monkeypatch, raises=httpx.ConnectError("secret-internal-detail 10.0.0.9:8080"))
    loaded, err = asyncio.run(services._probe_loaded_model("cowork-llama-1", 8080))
    assert loaded is None
    assert err == "connection refused"
    assert "10.0.0.9" not in err and "secret-internal-detail" not in err


def test_probe_timeout_sets_a_sanitized_probe_error(monkeypatch):
    _mock_client(monkeypatch, raises=httpx.ReadTimeout("timed out talking to http://llama:8080/v1/models"))
    loaded, err = asyncio.run(services._probe_loaded_model("cowork-llama-1", 8080))
    assert loaded is None
    assert err == "timeout"
    assert "llama:8080" not in err


def test_probe_http_error_status_sets_probe_error(monkeypatch):
    _mock_client(monkeypatch, response=_FakeResponse(503, {}))
    loaded, err = asyncio.run(services._probe_loaded_model("cowork-llama-1", 8080))
    assert loaded is None and err == "HTTP 503"


def test_probe_200_with_none_loaded_returns_null_and_a_count_note(monkeypatch):
    payload = {"data": [
        {"id": "gemma-e2b", "status": {"value": "not_loaded"}},
        {"id": "qwen-8b", "status": {"value": "not_loaded"}},
    ]}
    _mock_client(monkeypatch, response=_FakeResponse(200, payload))
    loaded, err = asyncio.run(services._probe_loaded_model("cowork-llama-1", 8080))
    assert loaded is None
    assert err == "2 configured, none loaded"


def test_probe_200_with_a_loaded_model_returns_its_id(monkeypatch):
    payload = {"data": [
        {"id": "gemma-e2b", "status": {"value": "loaded"}},
        {"id": "qwen-8b", "status": {"value": "not_loaded"}},
    ]}
    _mock_client(monkeypatch, response=_FakeResponse(200, payload))
    loaded, err = asyncio.run(services._probe_loaded_model("cowork-llama-1", 8080))
    assert loaded == "gemma-e2b" and err is None


# ---------- snapshot_llama_backends: what actually reaches the API/Hardware tab ----------

class _FakeImage:
    def __init__(self, tag):
        self.tags = [tag]
        self.short_id = "sha256:deadbeef"


class _FakeContainer:
    def __init__(self, name, attrs, status="running"):
        self.name = name
        self.status = status
        self.image = _FakeImage("ghcr.io/ggml-org/llama.cpp:server-cuda")
        self.short_id = "abc123"
        self.attrs = attrs


class _FakeContainers:
    def __init__(self, by_name):
        self._by_name = by_name

    def get(self, name):
        from docker.errors import NotFound
        if name not in self._by_name:
            raise NotFound(name)
        return self._by_name[name]

    def list(self, all=False):
        return list(self._by_name.values())


class _FakeDockerClient:
    def __init__(self, by_name):
        self.containers = _FakeContainers(by_name)


def _snapshot_for(monkeypatch, container, *, names=("cowork-llama-1",)):
    monkeypatch.setattr(services, "_docker_client", lambda: _FakeDockerClient({container.name: container}))
    monkeypatch.setattr(services, "_effective_container_names", lambda: list(names))
    return asyncio.run(services.snapshot_llama_backends())


def test_snapshot_sets_probe_error_when_no_port_can_be_found(monkeypatch):
    monkeypatch.setattr(settings, "llama_default_port", 0)
    container = _FakeContainer("cowork-llama-1", _attrs())  # no ports, no cmd, no env, no default
    [backend] = _snapshot_for(monkeypatch, container)
    assert backend.internal_port is None
    assert backend.loaded_model is None
    assert backend.probe_error == "port unknown"


def test_snapshot_uses_the_default_port_when_docker_publishes_none(monkeypatch):
    monkeypatch.setattr(settings, "llama_default_port", 8080)
    attrs = _attrs()  # NetworkSettings.Ports = {}, Config.ExposedPorts = null -- the live #341 case
    container = _FakeContainer("cowork-llama-1", attrs)
    payload = {"data": [{"id": "gemma-e2b", "status": {"value": "loaded"}}]}
    _mock_client(monkeypatch, response=_FakeResponse(200, payload))
    [backend] = _snapshot_for(monkeypatch, container)
    assert backend.internal_port == 8080
    assert backend.loaded_model == "gemma-e2b"
    assert backend.probe_error is None


def test_snapshot_never_shows_no_model_loaded_for_a_skipped_or_failed_probe(monkeypatch):
    """Issue #341's core contract: probe_error must be set whenever the probe was skipped or
    failed, so Hardware can tell "unavailable" apart from a genuinely empty engine -- and
    loaded_model must not be reported as a confident null in that case."""
    monkeypatch.setattr(settings, "llama_default_port", 8080)
    container = _FakeContainer("cowork-llama-1", _attrs())
    _mock_client(monkeypatch, raises=httpx.ConnectError("refused"))
    [backend] = _snapshot_for(monkeypatch, container)
    assert backend.loaded_model is None
    assert backend.probe_error == "connection refused"


def test_snapshot_successful_empty_probe_clears_probe_error(monkeypatch):
    """A probe that completed and genuinely found nothing loaded is not the same failure as
    #341 (unreachable port): probe_error must be None so Hardware shows "no model loaded",
    not "unavailable"."""
    monkeypatch.setattr(settings, "llama_default_port", 8080)
    container = _FakeContainer("cowork-llama-1", _attrs())
    payload = {"data": [{"id": "gemma-e2b", "status": {"value": "not_loaded"}}]}
    _mock_client(monkeypatch, response=_FakeResponse(200, payload))
    [backend] = _snapshot_for(monkeypatch, container)
    assert backend.loaded_model is None
    assert backend.probe_error is None
