"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_DRAFT_HEADS_CONFIRMED=1 the private Python child
refuses GET /api/v1/sections/{name}/draft-heads (the native parent reads it); other methods keep
their 405, the draft-head download and every other route stay here. The models folder is a
temporary one with a synthetic file; no Hugging Face request is made (no download history)."""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_DRAFT_HEADS_CONFIRMED"
TOKEN = "synthetic-draft-heads-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/sections/synthetic-m/draft-heads"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    with open(models / "synthetic-m.gguf", "wb") as f:
        f.truncate(16)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(db, "_DB_PATH", None)
    db.init()
    monkeypatch.delenv(ENV, raising=False)
    return TestClient(app, raise_server_exceptions=False)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_the_read(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.get(PATH).status_code == 401
    answer = client.get(PATH, headers=auth())
    assert answer.status_code == 503 and answer.content == REFUSED
    for path in ("/api/v1/sections/a%20b/draft-heads", "/api/v1/sections/%C3%BC/draft-heads"):
        assert client.get(path, headers=auth()).content == REFUSED
    assert client.post(PATH, headers=auth()).status_code == 405
    # A slash in the name, the bare section read and the draft-head download are not this route.
    assert client.get("/api/v1/sections/a%2Fb/draft-heads", headers=auth()).status_code == 404
    assert client.get("/api/v1/sections//draft-heads", headers=auth()).content != REFUSED
    assert client.get("/api/v1/sections/synthetic-m", headers=auth()).content != REFUSED
    download = client.post(PATH + "/download", json={}, headers=auth())
    assert download.content != REFUSED


@pytest.mark.parametrize("value", [None, "0", "true", " 1"])
def test_other_spellings_keep_python_ownership(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    answer = client.get(PATH, headers=auth())
    assert answer.status_code == 200, answer.content
    assert answer.json()["section"] == "synthetic-m" and answer.json()["repo"] is None
