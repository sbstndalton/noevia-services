"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOAD_TARGETS_CONFIRMED=1 the private Python child
refuses GET /api/v1/download-targets (the native parent answers it) and keeps every other route,
including POST /api/v1/downloads, which validates its target against the same list.

Synthetic folders and tokens only; FastAPI startup never runs and nothing is downloaded.
"""
import pytest
from fastapi.testclient import TestClient

from app import api, services
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOAD_TARGETS_CONFIRMED"
TOKEN = "synthetic-download-targets-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/download-targets"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    models = tmp_path / "models"
    (models / "extra").mkdir(parents=True)
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "model_download_targets", "extra")
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(services, "MOUNTINFO_PATH", str(tmp_path / "no-mountinfo"))
    return TestClient(app)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_only_the_get_after_token_gate(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    scans = []
    real = api.download_targets
    monkeypatch.setattr(api, "download_targets", lambda: scans.append(1) or real())
    assert client.get(PATH).status_code == 401
    refused = client.get(PATH, headers=auth())
    assert refused.status_code == 503
    assert refused.content == REFUSED
    assert scans == []
    assert client.post(PATH, headers=auth()).status_code == 405
    assert client.get(PATH + "/", headers=auth(), follow_redirects=False).status_code == 307
    # POST /downloads still validates its target here (a bad target never reaches the network).
    bad = client.post("/api/v1/downloads", headers=auth(), json={"target": "nope", "repo": "a/b"})
    assert bad.status_code == 400
    assert scans == [1]


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner(monkeypatch, client, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    answer = client.get(PATH, headers=auth())
    assert answer.status_code == 200
    assert [t["id"] for t in answer.json()["targets"]] == ["", "extra"]
