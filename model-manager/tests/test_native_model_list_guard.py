"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_MODELS_CONFIRMED=1 the private Python child refuses
GET /api/v1/models (the native parent reads the files, ini and badges) and serves the private
POST /api/v1/internal/models-runtime call with exactly the live fields `_entry` places: `loadedOn`
from `_loaded_map()` and `fit` from `vram_fit_chips(bytes)`. Under
NOEVIA_NATIVE_MODEL_MANAGER_OVERVIEW_CONFIRMED=1 it refuses GET /api/v1/overview and serves the
private GET /api/v1/internal/overview-runtime call with the overview's `backends` and
`activeDownloads`. Without its confirmation each call does not exist (404) and Python keeps the read.

Synthetic model files, backends, chips, jobs and tokens only; FastAPI startup never runs, and Docker,
the GPU sampler and the download table are replaced by stand-ins.
"""
import json
import struct
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db, downloader, services
from app.config import settings
from app.main import app

MODELS_ENV = "NOEVIA_NATIVE_MODEL_MANAGER_MODELS_CONFIRMED"
OVERVIEW_ENV = "NOEVIA_NATIVE_MODEL_MANAGER_OVERVIEW_CONFIRMED"
TOKEN = "synthetic-model-list-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
MODELS = "/api/v1/models"
OVERVIEW = "/api/v1/overview"
MODELS_RUNTIME = "/api/v1/internal/models-runtime"
OVERVIEW_RUNTIME = "/api/v1/internal/overview-runtime"


def _gguf(arch: str) -> bytes:
    key, value = b"general.architecture", arch.encode()
    return (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", len(key)) + key
            + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    models = tmp_path / "models"
    (models / "sub").mkdir(parents=True)
    (models / "synthetic-Q4.gguf").write_bytes(_gguf("synth"))
    (models / "sub" / "main.gguf").write_bytes(_gguf("synth") + b"\0" * 37)
    (models / "sub" / "mmproj-main.gguf").write_bytes(_gguf("vision"))
    (models / "models.ini").write_text("[short-id]\nmodel = synthetic-Q4.gguf\n")
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "models_host_path", "/srv/synthetic-models")
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    db.init()
    for env in (MODELS_ENV, OVERVIEW_ENV):
        monkeypatch.delenv(env, raising=False)

    async def backends():
        return [services.LlamaBackend(name="synthetic-a", found=True, status="running",
                                      loaded_model="short-id, main"),
                services.LlamaBackend(name="synthetic-b", found=True, status="running", loaded_model="short-id"),
                services.LlamaBackend(name="idle", found=True, status="exited")]

    calls = []

    def chips(size):
        calls.append(size)
        return [{"name": "synthetic-gpu", "vram_gb": 0.1 + 0.2, "verdict": "fits", "ratio_pct": size % 7},
                {"name": "edge", "vram_gb": float("nan"), "verdict": "slow", "ratio_pct": 1e16, "is_cpu": True}]

    jobs = [SimpleNamespace(status=s) for s in ("queued", "downloading", "done", "failed", "downloading")]
    monkeypatch.setattr(services, "snapshot_llama_backends", backends)
    monkeypatch.setattr(services, "vram_fit_chips", chips)
    monkeypatch.setattr(downloader.manager, "snapshot", lambda: list(jobs))
    test_client = TestClient(app)
    test_client.chip_calls = calls
    return test_client


def auth():
    return {"x-model-loader-token": TOKEN}


def post_sizes(client, sizes):
    return client.post(MODELS_RUNTIME, content=json.dumps({"bytes": sizes}), headers={
        **auth(), "content-type": "application/json"})


def test_confirmed_child_refuses_only_the_models_get_after_token_gate(monkeypatch, client):
    monkeypatch.setenv(MODELS_ENV, "1")
    assert client.get(MODELS).status_code == 401
    refused = client.get(MODELS, headers=auth())
    assert refused.status_code == 503
    assert refused.content == REFUSED
    assert client.chip_calls == []
    # Other methods, the slash variant, the overview and the neighbouring model routes keep theirs.
    assert client.post(MODELS, headers=auth()).status_code == 405
    assert client.get(MODELS + "/", headers=auth(), follow_redirects=False).status_code == 307
    assert client.get(OVERVIEW, headers=auth()).status_code == 200
    assert client.get(MODELS + "/updates", headers=auth()).status_code == 200
    assert client.get(MODELS + "/detail?key=synthetic-Q4.gguf", headers=auth()).status_code == 200
    # The runtime calls are behind the same token gate, and only the confirmed one exists.
    assert client.post(MODELS_RUNTIME, content=b'{"bytes":[]}').status_code == 401
    assert client.get(OVERVIEW_RUNTIME, headers=auth()).status_code == 404


def test_confirmed_child_refuses_only_the_overview_get_after_token_gate(monkeypatch, client):
    monkeypatch.setenv(OVERVIEW_ENV, "1")
    assert client.get(OVERVIEW).status_code == 401
    refused = client.get(OVERVIEW, headers=auth())
    assert refused.status_code == 503
    assert refused.content == REFUSED
    assert client.post(OVERVIEW, headers=auth()).status_code == 405
    assert client.get(OVERVIEW + "/", headers=auth(), follow_redirects=False).status_code == 307
    assert client.get(MODELS, headers=auth()).status_code == 200
    assert client.get(OVERVIEW_RUNTIME).status_code == 401
    assert post_sizes(client, []).status_code == 404


@pytest.mark.parametrize("value", [None, "0", "", "true", " 1", "1 ", "yes"])
def test_any_other_confirmation_keeps_python_owner_and_hides_the_runtime_calls(monkeypatch, client, value):
    for env in (MODELS_ENV, OVERVIEW_ENV):
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, value)
    listed = client.get(MODELS, headers=auth())
    assert listed.status_code == 200
    assert {m["modelId"]: m["loadedOn"] for m in listed.json()["models"]} == {
        "short-id": ["synthetic-a", "synthetic-b"], "main": ["synthetic-a"]}
    assert client.get(OVERVIEW, headers=auth()).json()["activeDownloads"] == 3
    for answer in (post_sizes(client, [1]), client.get(OVERVIEW_RUNTIME, headers=auth())):
        assert answer.status_code == 404
        assert answer.json() == {"detail": "Not Found"}


def test_models_runtime_is_exactly_what_the_python_list_places(monkeypatch, client):
    """The native parent splices `loaded[modelId]` and each chips text verbatim; that must
    reproduce Python's own list answer byte for byte, a non-finite float included (null)."""
    python = client.get(MODELS, headers=auth())
    assert python.status_code == 200
    entries = python.json()["models"]
    sizes = [e["bytes"] for e in entries]
    assert len(set(sizes)) == len(sizes) == 2
    client.chip_calls.clear()
    monkeypatch.setenv(MODELS_ENV, "1")
    runtime = post_sizes(client, sizes)
    assert runtime.status_code == 200
    assert runtime.headers["content-type"] == "application/json"
    assert client.chip_calls == sizes
    body = runtime.json()
    assert set(body) == {"loaded", "fit"}
    assert len(body["fit"]) == len(entries)
    assert all(fit[1]["vram_gb"] is None for fit in body["fit"])
    fit_texts = []
    for entry, fit in zip(entries, body["fit"]):
        assert body["loaded"].get(entry["modelId"], []) == entry["loadedOn"]
        fit_texts.append(json.dumps(fit, ensure_ascii=False, separators=(",", ":")))
        # The text the parent places is the text Python renders inside the list.
        assert f'"fit":{fit_texts[-1]},'.encode() in python.content
    loaded_text = json.dumps(body["loaded"], ensure_ascii=False, separators=(",", ":"))
    assert runtime.content == f'{{"loaded":{loaded_text},"fit":[{",".join(fit_texts)}]}}'.encode()


def test_overview_runtime_is_exactly_what_the_python_overview_places(monkeypatch, client):
    python = client.get(OVERVIEW, headers=auth())
    assert python.status_code == 200
    monkeypatch.setenv(OVERVIEW_ENV, "1")
    runtime = client.get(OVERVIEW_RUNTIME, headers=auth())
    assert runtime.status_code == 200
    assert runtime.headers["content-type"] == "application/json"
    body = runtime.json()
    assert set(body) == {"backends", "activeDownloads"}
    assert body["activeDownloads"] == 3
    backends = json.dumps(body["backends"], ensure_ascii=False, separators=(",", ":"))
    assert f'"backends":{backends},"activeDownloads":3,'.encode() in python.content
    assert runtime.content == f'{{"backends":{backends},"activeDownloads":3}}'.encode()


def test_models_runtime_validates_its_body(monkeypatch, client):
    monkeypatch.setenv(MODELS_ENV, "1")
    bad = [b"", b"[]", b"null", b'{"bytes":1}', b'{"bytes":[1],"x":1}', b'{"size":[1]}',
           b'{"bytes":[-1]}', b'{"bytes":[1.0]}', b'{"bytes":[true]}', b'{"bytes":["1"]}',
           b'{"bytes":[null]}', b'{"bytes":[%s]}' % (b"9" * 40), b'{"bytes":[%s]}' % (b"1" * 5000),
           json.dumps({"bytes": [1] * 4097}).encode(), b'{"bytes":[1]' + b" " * (256 * 1024) + b"}",
           b'{"bytes":[1]']
    for body in bad:
        answer = client.post(MODELS_RUNTIME, content=body, headers=auth())
        assert answer.status_code == 422, body[:40]
        assert answer.content == b'{"detail":"invalid runtime body"}'
    assert client.get(MODELS_RUNTIME, headers=auth()).status_code == 405
    assert client.chip_calls == []
    assert post_sizes(client, []).json() == {"loaded": {"short-id": ["synthetic-a", "synthetic-b"],
                                                         "main": ["synthetic-a"]}, "fit": []}
    # Any 128-bit total (39 digits) and the parent's model limit are still accepted.
    assert post_sizes(client, [int("9" * 39)]).status_code == 200
    assert post_sizes(client, [0] * 4096).status_code == 200
    assert client.chip_calls == [int("9" * 39)] + [0] * 4096


def test_overview_runtime_answers_get_only(monkeypatch, client):
    monkeypatch.setenv(OVERVIEW_ENV, "1")
    for method in ("post", "put", "delete"):
        answer = getattr(client, method)(OVERVIEW_RUNTIME, headers=auth())
        assert answer.status_code == 405
        assert answer.content == b'{"detail":"Method Not Allowed"}'


def test_model_list_guards_do_not_touch_other_native_guards(monkeypatch, client):
    monkeypatch.setenv(MODELS_ENV, "1")
    monkeypatch.setenv(OVERVIEW_ENV, "1")
    for other in ("NOEVIA_NATIVE_MODEL_MANAGER_HOST_CONFIRMED", "NOEVIA_NATIVE_MODEL_MANAGER_SECTIONS_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_MODEL_DETAIL_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_MODEL_UPDATES_CONFIRMED",
                  "NOEVIA_NATIVE_MODEL_MANAGER_DOWNLOADS_CONFIRMED"):
        monkeypatch.delenv(other, raising=False)
    assert client.get("/api/v1/health").status_code == 503  # always native (noevia#1071)
    assert client.get("/api/v1/sections", headers=auth()).status_code == 200
    assert client.get("/api/v1/models/updates", headers=auth()).status_code == 200
    assert client.get("/api/v1/internal/model-detail-runtime?bytes=1&fit=1", headers=auth()).status_code == 404
