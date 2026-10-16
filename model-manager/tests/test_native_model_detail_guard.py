"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DETAIL_CONFIRMED=1 the private Python child
refuses GET /api/v1/models/detail (the native parent reads the files) and serves the private
GET /api/v1/internal/model-detail-runtime call, which hands the parent exactly the live fields
`_entry` would place: `loadedOn` from `_loaded_map()` and `fit` from `vram_fit_chips(bytes)`.
Without the confirmation the runtime call does not exist (404) and Python keeps the detail read.

Synthetic model files, backends, chips and tokens only; FastAPI startup never runs, and Docker
and the GPU sampler are replaced by stand-ins.
"""
import json
import struct

import pytest
from fastapi.testclient import TestClient

from app import db, services
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DETAIL_CONFIRMED"
TOKEN = "synthetic-model-detail-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/models/detail"
RUNTIME = "/api/v1/internal/model-detail-runtime"


def _gguf(arch: str) -> bytes:
    key, value = b"general.architecture", arch.encode()
    return (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", len(key)) + key
            + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    models = tmp_path / "models"
    (models / "sub").mkdir(parents=True)
    (models / "synthetic-Q4.gguf").write_bytes(_gguf("llama"))
    (models / "sub" / "main.gguf").write_bytes(_gguf("llama"))
    (models / "sub" / "mmproj-main.gguf").write_bytes(_gguf("clip"))
    (models / "solo").mkdir()
    (models / "solo" / "mmproj-orphan.gguf").write_bytes(_gguf("clip"))
    (models / "models.ini").write_text("[short-id]\nmodel = synthetic-Q4.gguf\n")
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    db.init()
    monkeypatch.delenv(ENV, raising=False)

    async def backends():
        return [services.LlamaBackend(name="synthetic-a", found=True, status="running",
                                      loaded_model="short-id, main"),
                services.LlamaBackend(name="synthetic-b", found=True, status="running", loaded_model="short-id"),
                services.LlamaBackend(name="idle", found=True, status="exited")]

    calls = []

    def chips(size):
        calls.append(size)
        return [{"name": "synthetic-gpu", "vram_gb": 0.1 + 0.2, "verdict": "fits", "ratio_pct": size % 7},
                {"name": "edge", "vram_gb": 1e16, "verdict": "slow", "ratio_pct": 0, "is_cpu": True}]

    monkeypatch.setattr(services, "snapshot_llama_backends", backends)
    monkeypatch.setattr(services, "vram_fit_chips", chips)
    test_client = TestClient(app)
    test_client.chip_calls = calls
    return test_client


def auth():
    return {"x-model-loader-token": TOKEN}


def test_confirmed_child_refuses_only_the_detail_get_after_token_gate(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    assert client.get(PATH + "?key=synthetic-Q4.gguf").status_code == 401
    refused = client.get(PATH + "?key=synthetic-Q4.gguf", headers=auth())
    assert refused.status_code == 503
    assert refused.content == REFUSED
    assert client.chip_calls == []
    # Other methods, the slash variant and the neighbouring model routes keep their answers.
    assert client.post(PATH, headers=auth()).status_code == 405
    assert client.get(PATH + "/", headers=auth(), follow_redirects=False).status_code == 307
    listed = client.get("/api/v1/models", headers=auth())
    assert listed.status_code == 200
    assert client.get("/api/v1/models/updates", headers=auth()).status_code == 200
    # The runtime call is behind the same token gate.
    assert client.get(RUNTIME + "?bytes=1&fit=1").status_code == 401


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner_and_hides_the_runtime_call(monkeypatch, client, value):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    answer = client.get(PATH + "?key=synthetic-Q4.gguf", headers=auth())
    assert answer.status_code == 200
    assert answer.json()["loadedOn"] == ["synthetic-a", "synthetic-b"]
    hidden = client.get(RUNTIME + "?bytes=1&fit=1", headers=auth())
    assert hidden.status_code == 404
    assert hidden.json() == {"detail": "Not Found"}


@pytest.mark.parametrize("key", ["synthetic-Q4.gguf", "sub/main.gguf", "solo/mmproj-orphan.gguf"])
def test_runtime_call_is_exactly_what_the_python_detail_places(monkeypatch, client, key):
    """The native parent splices `loaded[modelId]` and the chips text verbatim; that must
    reproduce Python's own detail answer byte for byte."""
    python = client.get(PATH + f"?key={key}", headers=auth())
    assert python.status_code == 200
    entry = python.json()
    monkeypatch.setenv(ENV, "1")
    runtime = client.get(RUNTIME + f"?bytes={entry['bytes']}&fit={0 if entry['companion'] else 1}",
                         headers=auth())
    assert runtime.status_code == 200
    assert runtime.headers["content-type"] == "application/json"
    body = runtime.json()
    assert set(body) == {"loaded", "fit"}
    assert body["loaded"].get(entry["modelId"], []) == entry["loadedOn"]
    # Byte-level: the chips text the parent splices is the text Python renders inside detail.
    fit_text = json.dumps(body["fit"], ensure_ascii=False, separators=(",", ":"))
    assert f'"fit":{fit_text},"badges"'.encode() in python.content
    assert f'"fit":{fit_text}}}'.encode() == runtime.content[runtime.content.index(b'"fit":'):]
    if entry["companion"]:
        assert body["fit"] == [] and entry["fit"] == []


def test_runtime_call_validates_its_query(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for query in ("", "?bytes=1", "?fit=1", "?bytes=-1&fit=1", "?bytes=x&fit=1", "?bytes=1&fit=maybe",
                  "?bytes=%s&fit=1" % ("9" * 40), "?bytes=%s&fit=1" % ("1" * 5000), "?bytes=%D9%A1&fit=1"):
        assert client.get(RUNTIME + query, headers=auth()).status_code == 422, query
    assert client.post(RUNTIME + "?bytes=1&fit=1", headers=auth()).status_code == 405
    assert client.get(RUNTIME + "?bytes=0&fit=0", headers=auth()).json()["fit"] == []
    assert client.chip_calls == []
    # #1889: 39 digits (any 128-bit total) is still accepted.
    assert client.get(RUNTIME + "?bytes=%s&fit=1" % ("9" * 39), headers=auth()).status_code == 200
    assert client.chip_calls == [int("9" * 39)]


def test_model_detail_guard_does_not_touch_other_native_guards(monkeypatch, client):
    monkeypatch.setenv(ENV, "1")
    for other in ("NOEVIA_NATIVE_MODEL_MANAGER_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_DIAGNOSE_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_SETTINGS_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_MODEL_UPDATES_CONFIRMED"):
        monkeypatch.delenv(other, raising=False)
    assert client.get("/api/v1/health").json() == {"ok": True}
    assert client.get("/api/v1/sections", headers=auth()).status_code == 200


def test_updates_guard_alone_leaves_model_detail_with_python(monkeypatch, client):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_MODEL_UPDATES_CONFIRMED", "1")
    assert client.get(PATH + "?key=synthetic-Q4.gguf", headers=auth()).status_code == 200
