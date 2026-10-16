"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_OVERVIEW_CONFIRMED=1 the private Python child
refuses GET /api/v1/benchmark (the native parent answers it), keeps every job control here, and
serves the parent's private call for the runner's live job state.
No Docker, container or model is touched.
"""
import pytest
from fastapi.testclient import TestClient

from app import bench
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_OVERVIEW_CONFIRMED"
TOKEN = "synthetic-benchmark-overview-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
CALL = "/api/v1/internal/bench-job"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(bench, "_STATE", bench.JobState(run_id=3, status="done", backend="b", total=4, done=4,
                                                        started_at=10.0, finished_at=12.5, lines=["x"]))
    return TestClient(app, raise_server_exceptions=False)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_the_overview_only(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.get("/api/v1/benchmark").status_code == 401
    answer = client.get("/api/v1/benchmark", headers=auth())
    assert answer.status_code == 503 and answer.content == REFUSED
    # The job controls and the run read stay here.
    assert client.get("/api/v1/benchmark/progress", headers=auth()).json()["job"]["run_id"] == 3
    assert client.post("/api/v1/benchmark/cancel", headers=auth()).json()["job"]["status"] == "done"


@pytest.mark.parametrize("value", [None, "0", "true", " 1"])
def test_other_spellings_keep_python_ownership(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert client.get(CALL, headers=auth()).status_code == 404
    answer = client.get("/api/v1/benchmark", headers=auth())
    assert answer.status_code != 503


def test_job_call_is_the_overview_job(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.get(CALL).status_code == 401
    answer = client.get(CALL, headers=auth())
    assert answer.status_code == 200
    assert answer.content == (b'{"run_id":3,"status":"done","backend":"b","total":4,"done":4,"current":"",'
                              b'"started_at":10.0,"finished_at":12.5,"error":"","unit":"requests","lines":["x"],'
                              b'"pct":100,"elapsed":2.5,"eta":0.0,"active":false}')
    assert client.post(CALL, headers=auth()).status_code == 405
