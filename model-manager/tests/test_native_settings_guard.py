"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_CONFIRMED=1 the private Python child
refuses GET /api/v1/settings (the native parent answers it) and keeps PUT, the single writer.

Synthetic tokens and a temporary store only; FastAPI startup never runs.
"""
import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_CONFIRMED"
TOKEN = "synthetic-settings-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'


@pytest.fixture()
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    db.init()
    return TestClient(app)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_only_get_after_token_gate(monkeypatch, store):
    monkeypatch.setenv(ENV, "1")
    reads = []
    real = db.get_setting
    monkeypatch.setattr(db, "get_setting", lambda *a: reads.append(a) or real(*a))
    assert store.get("/api/v1/settings").status_code == 401
    refused = store.get("/api/v1/settings", headers=auth())
    assert refused.status_code == 503
    assert refused.content == REFUSED
    assert reads == []
    # The writer stays here, and its echo is still Python's own get_settings.
    written = store.put("/api/v1/settings", json={"hfToken": "  hf_SYNTHETIC0000wxyz  "}, headers=auth())
    assert written.status_code == 200
    assert written.json() == {"hasToken": True, "tokenHint": "…wxyz"}
    assert db.get_setting("hf_token") == "hf_SYNTHETIC0000wxyz"
    # Other methods and the slash variant keep their existing answers.
    assert store.post("/api/v1/settings", headers=auth()).status_code == 405
    assert store.get("/api/v1/settings/", headers=auth(), follow_redirects=False).status_code == 307


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner(monkeypatch, store, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert store.get("/api/v1/settings", headers=auth()).json() == {"hasToken": False, "tokenHint": ""}
    db.set_setting("hf_token", "äöüßéèàç\U0001F600")
    assert store.get("/api/v1/settings", headers=auth()).json() == {"hasToken": True, "tokenHint": "…èàç\U0001F600"}


def test_settings_guard_does_not_touch_other_native_guards(monkeypatch, store):
    monkeypatch.setenv(ENV, "1")
    for other in ("NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_DIAGNOSE_CONFIRMED"):
        monkeypatch.delenv(other, raising=False)
    assert store.get("/api/v1/health").status_code == 503  # always native (noevia#1071)
    assert store.get("/api/v1/prompts", headers=auth()).status_code == 200
