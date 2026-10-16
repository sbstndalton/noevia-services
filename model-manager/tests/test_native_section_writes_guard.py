"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_SECTION_WRITES_CONFIRMED=1 the private Python child
refuses the five models.ini writers (the native parent is the only writer) and serves the parent's
private safe-defaults-plan call, read-only. Also #1921: a rename that cannot edit the header in
place no longer copies [DEFAULT] keys into the section. Synthetic files only.
"""
import struct

import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_SECTION_WRITES_CONFIRMED"
TOKEN = "synthetic-section-writes-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PLAN = "/api/v1/internal/safe-defaults-plan"


def gguf(ctx=4096) -> bytes:
    kv = [("general.architecture", 8, "llama"), ("llama.context_length", 4, ctx)]
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, kind, value in kv:
        k = key.encode()
        out += struct.pack("<Q", len(k)) + k + struct.pack("<I", kind)
        out += (struct.pack("<Q", len(value)) + value.encode()) if kind == 8 else struct.pack("<I", value)
    return out


@pytest.fixture()
def client(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "small.gguf").write_bytes(gguf())
    (models / "models.ini").write_text("[a]\nk = v\n")
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.delenv(ENV, raising=False)
    return TestClient(app, raise_server_exceptions=False)


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_every_writer_and_writes_nothing(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    before = settings.models_ini_path.read_bytes()
    assert client.put("/api/v1/models-ini", json={}).status_code == 401
    for method, path in [("PUT", "/api/v1/models-ini"), ("PUT", "/api/v1/sections/a"), ("DELETE", "/api/v1/sections/a"),
                         ("POST", "/api/v1/sections/a/rename"), ("POST", "/api/v1/sections/small/safe-defaults"),
                         ("PUT", "/api/v1/sections/a%20b")]:
        answer = client.request(method, path, headers=auth(), json={})
        assert answer.status_code == 503, (method, path)
        assert answer.content == REFUSED
    assert settings.models_ini_path.read_bytes() == before
    # Reads and other shapes stay here.
    assert client.get("/api/v1/sections/a", headers=auth()).status_code == 200
    assert client.put("/api/v1/sections/a/", headers=auth(), json={}, follow_redirects=False).status_code in (307, 404, 405)
    assert client.get("/api/v1/sections/a/rename", headers=auth()).status_code == 405


def test_plan_is_read_only_and_matches_the_route(monkeypatch, client):
    assert client.get(PLAN, headers=auth(), params={"name": "small"}).status_code == 404
    monkeypatch.setenv(ENV, "1")
    before = settings.models_ini_path.read_bytes()
    plan = client.get(PLAN, headers=auth(), params={"name": "small"})
    assert plan.status_code == 200
    body = plan.json()
    assert body["status"] == 200 and body["mtp"] is False and body["values"]["model"] == "/models/small.gguf"
    assert settings.models_ini_path.read_bytes() == before
    assert client.get(PLAN, headers=auth(), params={"name": "a"}).json() == {"status": 409, "detail": "settings already exist for this model"}
    assert client.get(PLAN, headers=auth(), params={"name": "nope"}).json()["status"] == 404
    assert client.post(PLAN, headers=auth()).status_code == 405
    monkeypatch.delenv(ENV)
    written = client.post("/api/v1/sections/small/safe-defaults", headers=auth()).json()
    assert written["mtp"] is False
    assert {k: v for k, v in written["section"].items() if k != "cache-ram"} == body["values"]


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_writing(monkeypatch, client, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    rev = client.get("/api/v1/sections", headers=auth()).json()["revision"]
    assert client.put("/api/v1/sections/a", headers=auth(), json={"baseRevision": rev, "values": {}}).status_code == 200
    assert client.get(PLAN, headers=auth(), params={"name": "small"}).status_code == 404


def test_1921_rename_fallback_keeps_default_keys_inherited(client):
    settings.models_ini_path.write_text("[DEFAULT]\nd = 1\n[b] junk\nk = v\n")
    rev = client.get("/api/v1/sections", headers=auth()).json()["revision"]
    assert client.post("/api/v1/sections/b/rename", headers=auth(), json={"baseRevision": rev, "newName": "b2"}).status_code == 200
    assert settings.models_ini_path.read_text() == "[DEFAULT]\nd = 1\n\n[b2]\nk = v\n\n"


def test_1934_section_name_with_a_trailing_newline_is_refused_and_models_ini_is_unchanged(client):
    before = settings.models_ini_path.read_bytes()
    rev = client.get("/api/v1/sections", headers=auth()).json()["revision"]
    assert client.put("/api/v1/sections/foo%0A", headers=auth(), json={"baseRevision": rev, "values": {"k": "v"}}).status_code == 400
    assert settings.models_ini_path.read_bytes() == before
    assert not list(settings.models_ini_path.parent.glob("models.ini.*"))
