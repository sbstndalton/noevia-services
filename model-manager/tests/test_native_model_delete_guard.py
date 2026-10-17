"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DELETE_CONFIRMED=1 the private Python child
refuses POST /api/v1/models/delete (the native parent checks the containers and deletes) and
deletes nothing; other methods keep their 405 and every other route stays here. The models folder
is a temporary one with synthetic zero-filled files; Docker is not consulted.
"""
import pytest
from fastapi.testclient import TestClient

from app import services
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DELETE_CONFIRMED"
TOKEN = "synthetic-model-delete-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/models/delete"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    models = tmp_path / "models"
    (models / "sub").mkdir(parents=True)
    with open(models / "synthetic-a.gguf", "wb") as f:
        f.truncate(16)
    with open(models / "sub" / "synthetic-b.gguf", "wb") as f:
        f.truncate(8)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "model_download_targets", "")
    monkeypatch.setattr(services, "_docker_client", lambda: None)
    monkeypatch.delenv(ENV, raising=False)
    c = TestClient(app, raise_server_exceptions=False)
    c.models = models
    return c


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_the_delete_and_deletes_nothing(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    body = {"models": ["synthetic-a.gguf", "sub/synthetic-b.gguf"]}
    assert client.post(PATH, json=body).status_code == 401
    answer = client.post(PATH, json=body, headers=auth())
    assert answer.status_code == 503 and answer.content == REFUSED
    assert (client.models / "synthetic-a.gguf").exists()
    assert (client.models / "sub" / "synthetic-b.gguf").exists()
    assert client.get(PATH, headers=auth()).status_code == 405
    assert client.get("/api/v1/models", headers=auth()).content != REFUSED


@pytest.mark.parametrize("value", [None, "0", "true", " 1"])
def test_other_spellings_keep_python_ownership(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    answer = client.post(PATH, json={"models": ["sub/synthetic-b.gguf"]}, headers=auth())
    assert answer.status_code == 200, answer.content
    assert answer.json()["results"][0]["ok"] is True
    assert not (client.models / "sub").exists()
    assert (client.models / "synthetic-a.gguf").exists()
