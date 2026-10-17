"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_RUNNER_CONFIRMED=1 the private Python child
refuses the four benchmark job routes (the native parent owns the one job slot), does not clean up
leftover sweep containers at startup (the parent does), and serves the parent's private call for a
request's observations. No Docker, container, GPU or model is touched.
"""
import pytest
from fastapi.testclient import TestClient

from app import bench
from app import main as main_module
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_RUNNER_CONFIRMED"
TOKEN = "synthetic-benchmark-runner-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
CALL = "/api/v1/internal/bench-observe"
ROUTES = [("post", "/api/v1/benchmark/start"), ("post", "/api/v1/benchmark/sweep"),
          ("get", "/api/v1/benchmark/progress"), ("post", "/api/v1/benchmark/cancel")]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(bench, "_STATE", bench.JobState(run_id=3, status="done", backend="b"))
    seen = []
    monkeypatch.setattr(bench, "_peak_vram", lambda b: seen.append(("peak", b)) or [1.5, 2.25])
    monkeypatch.setattr(bench, "_other_traffic",
                        lambda b, aliases, since: seen.append(("other", b, sorted(aliases), since)) or True)
    monkeypatch.setattr(bench, "_read_load_ms", lambda b, a: seen.append(("load", b, a)) or 1234.5)
    c = TestClient(app, raise_server_exceptions=False)
    c.seen = seen
    return c


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_the_four_job_routes(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for method, path in ROUTES:
        assert getattr(client, method)(path).status_code == 401
        kwargs = {"json": {}} if method == "post" else {}
        answer = getattr(client, method)(path, headers=auth(), **kwargs)
        assert answer.status_code == 503 and answer.content == REFUSED, path
    # The run read and the overview follow their own switches.
    assert client.get("/api/v1/benchmark/runs/1", headers=auth()).status_code != 503
    assert client.get("/api/v1/benchmark", headers=auth()).status_code != 503


@pytest.mark.parametrize("value", [None, "0", "true", " 1"])
def test_other_spellings_keep_python_ownership(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert client.get(CALL + "?backend=b&since=1.0", headers=auth()).status_code == 404
    answer = client.get("/api/v1/benchmark/progress", headers=auth())
    assert answer.status_code == 200 and answer.json()["job"]["run_id"] == 3


def test_observe_call_answers_the_three_observations(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    query = "?backend=llama-a&alias=m1&aliases=m1&aliases=m2&aliases=m1&since=1700000004.5&cold=1"
    assert client.get(CALL + query).status_code == 401
    answer = client.get(CALL + query, headers=auth())
    assert answer.status_code == 200
    assert answer.content == b'{"peak_vram_json":"[1.5, 2.25]","contended":true,"load_ms":1234.5}'
    assert client.seen == [("peak", "llama-a"), ("other", "llama-a", ["m1", "m2"], 1700000004.5),
                           ("load", "llama-a", "m1")]
    client.seen.clear()
    warm = client.get(CALL + "?backend=llama-a&alias=m1&aliases=m1&since=2.0&cold=0", headers=auth())
    assert warm.content == b'{"peak_vram_json":"[1.5, 2.25]","contended":true,"load_ms":null}'
    assert [s[0] for s in client.seen] == ["peak", "other"]
    assert client.get(CALL + "?backend=llama-a&since=x", headers=auth()).status_code == 422
    assert client.get(CALL + "?since=1", headers=auth()).status_code == 422
    assert client.post(CALL, headers=auth()).status_code == 405


@pytest.mark.parametrize("value,reaped", [("1", False), (None, True), ("0", True)])
def test_startup_cleanup_belongs_to_one_owner(monkeypatch, value, reaped):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    calls = []
    monkeypatch.setattr(bench, "reap_orphans", lambda: calls.append(1) or {"reaped": 0, "stuck": 0})
    monkeypatch.setattr(main_module.db, "init", lambda: None)
    monkeypatch.setattr(main_module.db, "seed_bench_prompts", lambda: None)
    monkeypatch.setattr(main_module.hw, "start_sampler", lambda: None)
    monkeypatch.setattr(main_module.settings, "migrate_cache_ram_on_start", False)
    main_module._startup()
    assert bool(calls) is reaped
