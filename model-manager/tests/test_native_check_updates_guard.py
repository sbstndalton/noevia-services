"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_CHECK_UPDATES_CONFIRMED=1 the private Python child
refuses POST /api/v1/models/check-updates (the native parent runs the check) and keeps the update
status read and every other route. No Hub request is made.
"""
import pytest
from fastapi.testclient import TestClient

from app import hf
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_CHECK_UPDATES_CONFIRMED"
TOKEN = "synthetic-check-updates-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/models/check-updates"


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    calls = []

    async def fake_check(records):
        calls.append(records)
        return 0
    monkeypatch.setattr(hf, "check_updates_for", fake_check)
    c = TestClient(app, raise_server_exceptions=False)
    c.calls = calls
    return c


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_the_check_only(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.post(PATH).status_code == 401
    answer = client.post(PATH, headers=auth())
    assert answer.status_code == 503 and answer.content == REFUSED
    assert client.calls == []
    assert client.get(PATH, headers=auth()).status_code == 405


@pytest.mark.parametrize("value", [None, "0", "true", " 1"])
def test_other_spellings_keep_python_ownership(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    answer = client.post(PATH, headers=auth())
    assert answer.content != REFUSED
