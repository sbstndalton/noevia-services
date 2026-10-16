"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_BENCH_RUNS_CONFIRMED=1 the private Python child
refuses canonical GET /api/v1/benchmark/runs/{id} (the native parent reads them) and serves the
parent's private bench-run-runtime call with the charts' two live values. Synthetic store rows,
backends and predictions only; no model file, container or GPU is touched.
"""
import pytest
from fastapi.testclient import TestClient

from app import db, helpers
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_BENCH_RUNS_CONFIRMED"
TOKEN = "synthetic-bench-runs-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
RUNTIME = "/api/v1/internal/bench-run-runtime"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(helpers, "_backend_list", lambda: [{"vram_gb": 23.987}, {"vram_gb": 11}])
    monkeypatch.setattr(helpers, "_predicted_vram_gb", lambda a: {"m": 7.125}.get(a))
    db.init()
    c = db._conn()
    c.execute("INSERT INTO bench_run(id, backend, status, started_at) VALUES(1, 'b', 'done', 1.0)")
    c.execute("INSERT INTO bench_variant(id, run_id, alias) VALUES(1, 1, 'm')")
    c.execute("INSERT INTO bench_result(variant_id, prompt_name, rep, gen_tps, peak_vram_json) VALUES(1, 'p', 0, 2.0, '[1.5]')")
    c.commit()
    return TestClient(app, raise_server_exceptions=False)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_canonical_ids_only(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.get("/api/v1/benchmark/runs/1").status_code == 401
    for segment in ("1", "007", "-1", "9223372036854775807"):
        refused = client.get(f"/api/v1/benchmark/runs/{segment}", headers=auth())
        assert refused.status_code == 503, segment
        assert refused.content == REFUSED
    # Spellings the parent forwards are still answered here.
    assert client.get("/api/v1/benchmark/runs/%31", headers=auth()).status_code == 200
    assert client.get("/api/v1/benchmark/runs/abc", headers=auth()).status_code == 422
    assert client.get("/api/v1/benchmark/runs/+1", headers=auth()).status_code == 200
    assert client.get("/api/v1/benchmark/runs/9223372036854775808", headers=auth()).status_code == 500
    assert client.get("/api/v1/benchmark", headers=auth()).status_code == 200


def test_runtime_call_matches_the_charts(monkeypatch, client):
    charts = client.get("/api/v1/benchmark/runs/1", headers=auth()).json()["charts"]
    assert client.get(RUNTIME, headers=auth(), params={"alias": "m"}).status_code == 404
    monkeypatch.setenv(ENV, "1")
    assert client.get(RUNTIME, params={"alias": "m"}).status_code == 401
    answer = client.get(RUNTIME, headers=auth(), params=[("alias", a) for a in charts["vram_labels"]])
    assert answer.status_code == 200
    assert answer.json() == {"capacity_gb": charts["capacity_gb"],
                             "predicted": dict(zip(charts["vram_labels"], charts["vram_predicted"]))}
    assert answer.content == b'{"capacity_gb":23.99,"predicted":{"m":7.125}}'
    assert client.post(RUNTIME, headers=auth()).status_code == 405
    assert client.get(RUNTIME, headers=auth(), params=[("alias", "x")] * 1025).status_code == 422


def test_runtime_capacity_is_zero_when_the_sampler_raises(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    def boom():
        raise RuntimeError("synthetic")
    monkeypatch.setattr(helpers, "_backend_list", boom)
    answer = client.get(RUNTIME, headers=auth())
    assert answer.content == b'{"capacity_gb":0.0,"predicted":{}}'


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert client.get("/api/v1/benchmark/runs/1", headers=auth()).status_code == 200
    assert client.get(RUNTIME, headers=auth()).status_code == 404
