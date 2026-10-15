"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_MODEL_UPDATES_CONFIRMED=1 the private Python child
refuses GET /api/v1/models/updates (the native parent answers it) and keeps every other route,
including the update_check writers.

Synthetic model files, store rows and tokens only; FastAPI startup never runs.
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_MODEL_UPDATES_CONFIRMED"
TOKEN = "synthetic-model-updates-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/models/updates"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "synthetic-Q4.gguf").write_bytes(b"")
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    db.init()
    db.set_update_check("synthetic-Q4.gguf", "synthetic/repo", "2999-01-01T00:00:00.000Z", 1700000000.5)
    return TestClient(app)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_only_the_get_after_token_gate(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    reads = []
    real = db.all_update_checks
    monkeypatch.setattr(db, "all_update_checks", lambda: reads.append(1) or real())
    assert client.get(PATH).status_code == 401
    refused = client.get(PATH, headers=auth())
    assert refused.status_code == 503
    assert refused.content == REFUSED
    assert reads == []
    # Other methods, the slash variant and the neighbouring model routes keep their answers.
    assert client.post(PATH, headers=auth()).status_code == 405
    assert client.get(PATH + "/", headers=auth(), follow_redirects=False).status_code == 307
    listed = client.get("/api/v1/models", headers=auth())
    assert listed.status_code == 200
    assert [m["name"] for m in listed.json()["models"]] == ["synthetic-Q4.gguf"]


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner(monkeypatch, client, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    answer = client.get(PATH, headers=auth())
    assert answer.status_code == 200
    status = answer.json()["status"]["synthetic-Q4.gguf"]
    assert status["status"] == "stale" and status["remote"] == "2999-01-01" and status["checked_at"] == 1700000000.5


def test_model_updates_guard_does_not_touch_other_native_guards(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for other in ("NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_DIAGNOSE_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_CONFIRMED"):
        monkeypatch.delenv(other, raising=False)
    assert client.get("/api/v1/health").json() == {"ok": True}
    assert client.get("/api/v1/settings", headers=auth()).status_code == 200
    assert client.get("/api/v1/prompts", headers=auth()).status_code == 200


def test_settings_guard_alone_leaves_model_updates_with_python(monkeypatch, client):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_CONFIRMED", "1")
    assert client.get(PATH, headers=auth()).status_code == 200
