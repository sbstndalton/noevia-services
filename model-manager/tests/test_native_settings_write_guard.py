"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_WRITE_CONFIRMED=1 the private Python child
refuses PUT /api/v1/settings (the native parent writes the token) and serves the parent's private
token test, the same `{ok, message}` put_settings computes. Synthetic tokens; no network.
"""
import pytest
from fastapi.testclient import TestClient

from app import db, hf
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_WRITE_CONFIRMED"
TOKEN = "synthetic-settings-write-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
TEST = "/api/v1/internal/hf-token-test"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)

    async def fake(token):
        if token == "boom":
            raise RuntimeError("synthetic failure")
        return (token.startswith("hf_good"), f"checked {len(token)} chars")
    monkeypatch.setattr(hf, "validate_token", fake)
    db.init()
    return TestClient(app, raise_server_exceptions=False)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_put_and_writes_nothing(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.put("/api/v1/settings", json={"hfToken": "x"}).status_code == 401
    refused = client.put("/api/v1/settings", headers=auth(), json={"hfToken": "hf_new"})
    assert refused.status_code == 503 and refused.content == REFUSED
    assert db.get_setting("hf_token", "") == ""
    assert client.get("/api/v1/settings", headers=auth()).status_code == 200


def test_token_test_matches_put_settings(monkeypatch, client):
    client.put("/api/v1/settings", headers=auth(), json={"hfToken": "hf_good_123456789"})
    expected = client.put("/api/v1/settings", headers=auth(), json={"test": True}).json()["test"]
    assert client.get(TEST, headers=auth()).status_code == 404
    monkeypatch.setenv(ENV, "1")
    assert client.get(TEST).status_code == 401
    assert client.get(TEST, headers=auth()).json() == expected
    assert client.post(TEST, headers=auth()).status_code == 405
    db.set_setting("hf_token", "boom")
    assert client.get(TEST, headers=auth()).json() == {"ok": False, "message": "Test failed: synthetic failure"}


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_writing(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert client.put("/api/v1/settings", headers=auth(), json={"hfToken": "hf_x_12345678"}).json()["tokenHint"] == "…5678"
