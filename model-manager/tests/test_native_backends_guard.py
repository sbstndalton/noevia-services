"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_BACKENDS_CONFIRMED=1 the native parent samples the
backends and answers GET /api/v1/backends. The private Python child then refuses that GET, stops
sampling the backends itself, takes the parent's samples into the cache its other readers use, and
runs the GPU reads that need `docker exec` when the parent asks. Every container is a synthetic
stand-in; no Docker daemon, container, GPU or model is touched.
"""
import threading

import pytest
from fastapi.testclient import TestClient

from app import hw, services
from app.api import _plain
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_BACKENDS_CONFIRMED"
HOST_ENV = "NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED"
TOKEN = "synthetic-backends-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PUSH = "/api/v1/internal/backend-stats"
PROBE = "/api/v1/internal/gpu-exec-probe"

CARD = {"index": 0, "name": "Synthetic GPU", "util_pct": 12.5, "vram_used_gb": 1.25, "vram_total_gb": 8.0,
        "temp_c": 40.0, "power_w": 20.5, "shared_used_gb": 0.5, "shared_total_gb": 16.0, "clock_mhz": 2100,
        "device": "card7"}
GPU = {"vendor": "vulkan", "name": "Synthetic GPU", "util_pct": 12.5, "vram_used_gb": 1.25, "vram_total_gb": 8.0,
       "temp_c": 40.0, "power_w": 20.5, "gpu_count": 1, "cards": [CARD], "memory_kind": "unified",
       "shared_used_gb": 0.5, "shared_total_gb": 16.0, "clock_mhz": 2100, "source": "sysfs", "measured": True}
STATS = {
    "llama-a": {"ok": True, "error": None, "gpu": GPU,
                "container": {"cpu_pct": 3.5, "mem_used_gb": 1.5, "mem_limit_gb": 0.0, "mem_anon_gb": None}},
    "llama-b": {"ok": False, "error": "container is exited", "gpu": None, "container": None},
}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(HOST_ENV, raising=False)

    async def snapshot():
        return [services.LlamaBackend(name="llama-a", found=False, status="not_found")]

    monkeypatch.setattr(services, "snapshot_llama_backends", snapshot)
    saved = dict(hw._CACHE)
    hw._CACHE.clear()
    yield TestClient(app, raise_server_exceptions=False)
    hw._CACHE.clear()
    hw._CACHE.update(saved)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_unconfirmed_child_keeps_the_route_and_hides_the_private_calls(client):
    answer = client.get("/api/v1/backends", headers=auth())
    assert answer.status_code == 200 and answer.json()["backends"][0]["name"] == "llama-a"
    assert client.put(PUSH, headers=auth(), json={"backends": STATS}).status_code == 404
    assert client.get(PROBE, headers=auth(), params={"name": "llama-a", "probe": "nvidia"}).status_code == 404
    assert hw._CACHE == {}


def test_confirmed_child_refuses_get_backends_only(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.get("/api/v1/backends").status_code == 401
    answer = client.get("/api/v1/backends", headers=auth())
    assert answer.status_code == 503 and answer.content == REFUSED
    # Other methods and the slash variant keep FastAPI's own answers here.
    assert client.post("/api/v1/backends", headers=auth()).status_code == 405
    # The slash variant is FastAPI's redirect to the exact path (which the parent answers).
    assert client.get("/api/v1/backends/", headers=auth(), follow_redirects=False).status_code == 307


def test_pushed_samples_replace_the_cache_its_readers_use(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    hw._CACHE["gone"] = hw.BackendStats(ok=False, error="old")
    assert client.put(PUSH, json={"backends": STATS}).status_code == 401
    answer = client.put(PUSH, headers=auth(), json={"backends": STATS})
    assert answer.status_code == 200 and answer.json() == {"ok": True}
    assert sorted(hw._CACHE) == ["llama-a", "llama-b"]
    assert {n: _plain(hw.stats_for(n)) for n in STATS} == STATS
    # The readers see the pushed figures exactly as their own sampler's.
    assert hw.vram_gb_for("llama-a") == 8.0 and hw.gpu_count_for("llama-a") == 1
    assert hw.card_vram_gb_for("llama-a") == [8.0]
    assert isinstance(hw.stats_for("llama-a").gpu.cards[0], hw.GpuCard)
    assert hw.stats_for("unknown").error == "warming up…"
    # Payloads of the wrong shape change nothing.
    bad_card = dict(CARD, extra=1)
    for bad in ({}, {"backends": []}, {"backends": STATS, "x": 1},
                {"backends": {"a": {"ok": True}}},
                {"backends": {"a": dict(STATS["llama-a"], gpu=dict(GPU, cards=[bad_card]))}},
                {"backends": {"a": dict(STATS["llama-a"], gpu=dict(GPU, cards={}))}}):
        assert client.put(PUSH, headers=auth(), json=bad).status_code == 422, bad
    assert client.put(PUSH, headers=auth(), content=b"not json").status_code == 422
    assert sorted(hw._CACHE) == ["llama-a", "llama-b"]
    assert client.get(PUSH, headers=auth()).status_code == 405
    # An empty push (no effective backend) empties the cache.
    assert client.put(PUSH, headers=auth(), json={"backends": {}}).status_code == 200
    assert hw._CACHE == {}


def test_exec_probe_call_runs_the_python_read(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    seen = []

    def probe(name, kind):
        seen.append((name, kind))
        return None if kind == "amd" else hw._gpu_from(GPU)

    monkeypatch.setattr(hw, "exec_probe", probe)
    assert client.get(PROBE, params={"name": "llama-a", "probe": "nvidia"}).status_code == 401
    answer = client.get(PROBE, headers=auth(), params={"name": "llama-a", "probe": "nvidia"})
    assert answer.status_code == 200 and answer.json() == {"gpu": GPU}
    assert client.get(PROBE, headers=auth(), params={"name": "llama.b_2", "probe": "amd"}).json() == {"gpu": None}
    assert client.get(PROBE, headers=auth(), params={"name": "x", "probe": "amd-nvidia"}).status_code == 200
    for params in ({"name": "llama-a"}, {"name": "llama-a", "probe": "rocm"}, {"name": "", "probe": "amd"},
                   {"name": "-x", "probe": "amd"}, {"name": "a/b", "probe": "amd"}, {"name": "a" * 257, "probe": "amd"}):
        assert client.get(PROBE, headers=auth(), params=params).status_code == 422, params
    assert client.put(PROBE, headers=auth()).status_code == 405
    assert seen == [("llama-a", "nvidia"), ("llama.b_2", "amd"), ("x", "amd-nvidia")]


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 "])
def test_any_other_confirmation_keeps_python_owner(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert client.get("/api/v1/backends", headers=auth()).status_code == 200
    assert client.put(PUSH, headers=auth(), json={"backends": STATS}).status_code == 404


# ---------- the sampler hand-off ----------

class _Stop(Exception):
    pass


def _one_tick(monkeypatch):
    def stop(_seconds):
        raise _Stop

    monkeypatch.setattr(hw.time, "sleep", stop)
    with pytest.raises(_Stop):
        hw._sample_loop()


def test_native_backends_stop_the_python_backend_sampler(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    monkeypatch.delenv(HOST_ENV, raising=False)
    monkeypatch.setattr(services, "_effective_container_names", lambda: pytest.fail("discovery must not run"))
    monkeypatch.setattr(hw, "_collect", lambda name: pytest.fail("no backend is sampled here"))
    monkeypatch.setattr(hw, "_read_host", lambda: hw.HostPoint(ts=1.0, cpu_pct=1.0, mem_used_gb=1.0,
                                                               mem_total_gb=2.0, mem_available_gb=1.0))
    before = len(hw._HOST_HISTORY)
    _one_tick(monkeypatch)
    assert len(hw._HOST_HISTORY) == min(before + 1, hw._HISTORY_MAXLEN)
    # With the host sampler native too, nothing is left: the host is not read either.
    monkeypatch.setenv(HOST_ENV, "1")
    monkeypatch.setattr(hw, "_read_host", lambda: pytest.fail("the host is native"))
    _one_tick(monkeypatch)


def test_start_sampler_starts_nothing_when_both_samplers_are_native(monkeypatch):
    monkeypatch.setattr(hw, "_sampler_started", False)
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self.name))
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv(HOST_ENV, "1")
    hw.start_sampler()
    assert started == []
    monkeypatch.delenv(HOST_ENV)
    hw.start_sampler()
    assert started == ["hw-sampler"]


def test_python_sampler_unchanged_without_confirmation(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv(HOST_ENV, "1")
    monkeypatch.setattr(services, "_effective_container_names", lambda: ["llama-z"])
    monkeypatch.setattr(hw, "_collect", lambda name: hw.BackendStats(ok=False, error=f"synthetic {name}"))
    saved = dict(hw._CACHE)
    try:
        _one_tick(monkeypatch)
        assert hw.stats_for("llama-z").error == "synthetic llama-z"
    finally:
        hw._CACHE.clear()
        hw._CACHE.update(saved)


# ---------- exec_probe itself ----------

class _Exec:
    def __init__(self, exit_code, output):
        self.exit_code, self.output = exit_code, output


class _Container:
    def __init__(self, answers):
        self.answers, self.commands = answers, []

    def exec_run(self, cmd, demux=False):
        self.commands.append(cmd.split()[0])
        return self.answers.get(cmd.split()[0], _Exec(1, b""))


class _Client:
    def __init__(self, container):
        self.container = container

    class _Containers:
        def __init__(self, container):
            self.container = container

        def get(self, name):
            if name != "llama-a":
                raise hw.NotFound("synthetic")
            return self.container

    @property
    def containers(self):
        return self._Containers(self.container)


def test_exec_probe_runs_the_same_reads_as_collect(monkeypatch):
    nvidia = _Exec(0, b"50, 2048, 8192, 61, 120.5, Synthetic Card\n")
    rocm = _Exec(0, b'{"card0": {"GPU use (%)": "7", "GPU Memory Allocated (VRAM%)": "25", '
                    b'"Temperature (Sensor edge) (C)": "45.0", "Average Graphics Package Power (W)": "30.0"}}')
    monkeypatch.setattr(settings, "gpu_vram", "llama-a:16")
    for probe, answers, commands, vendor in (
        ("nvidia", {"nvidia-smi": nvidia}, ["nvidia-smi"], "nvidia"),
        ("amd", {"rocm-smi": rocm}, ["rocm-smi"], "amd"),
        ("amd-nvidia", {"nvidia-smi": nvidia}, ["rocm-smi", "nvidia-smi"], "nvidia"),
        ("amd-nvidia", {}, ["rocm-smi", "nvidia-smi"], None),
    ):
        c = _Container(answers)
        monkeypatch.setattr(hw, "_client", lambda c=c: _Client(c))
        got = hw.exec_probe("llama-a", probe)
        assert c.commands == commands
        assert (got.vendor if got else None) == vendor
    monkeypatch.setattr(hw, "_client", lambda: _Client(_Container({"nvidia-smi": nvidia})))
    assert hw.exec_probe("missing", "nvidia") is None
    monkeypatch.setattr(hw, "_client", lambda: None)
    assert hw.exec_probe("llama-a", "nvidia") is None
    with pytest.raises(ValueError):
        hw.exec_probe("llama-a", "rocm")
