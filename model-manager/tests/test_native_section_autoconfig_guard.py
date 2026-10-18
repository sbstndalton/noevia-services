"""noevia#1318: under NOEVIA_NATIVE_MODEL_MANAGER_SECTION_AUTOCONFIG_CONFIRMED=1 the private Python
child refuses GET /api/v1/sections/{name}/autoconfig (the native parent answers it) and serves the
private POST /api/v1/internal/section-autoconfig-runtime call, which hands the parent exactly the live
inputs the route reads: the telemetry ingest and the section's measurements (empty ones when that
fails), then the backend inventory (helpers._backend_list). Without the confirmation the runtime
call does not exist (404) and Python keeps the route.

Synthetic model files, backends, measurements and tokens only; FastAPI startup never runs, and
Docker, the telemetry store and the GPU sampler are replaced by stand-ins.
"""
import json
import math
import struct

import pytest
from fastapi.testclient import TestClient

from app import db, helpers, services, telemetry
from app.config import settings
from app.main import app

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_SECTION_AUTOCONFIG_CONFIRMED"
TOKEN = "synthetic-section-autoconfig-guard-token"
REFUSED = b'{"detail":"Native model manager owns this endpoint"}'
PATH = "/api/v1/sections/synthetic-m/autoconfig"
RUNTIME = "/api/v1/internal/section-autoconfig-runtime"


def _gguf(arch: str) -> bytes:
    kv = [(b"general.architecture", 8, arch.encode()),
          (f"{arch}.block_count".encode(), 4, 8), (f"{arch}.context_length".encode(), 4, 4096),
          (f"{arch}.embedding_length".encode(), 4, 512), (f"{arch}.attention.head_count".encode(), 4, 8),
          (f"{arch}.attention.head_count_kv".encode(), 4, 8)]
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for key, kind, value in kv:
        out += struct.pack("<Q", len(key)) + key + struct.pack("<I", kind)
        out += struct.pack("<Q", len(value)) + value if kind == 8 else struct.pack("<I", value)
    return out


MEASURED = telemetry.Stats(n=7, gen_p50=141.5, gen_p25=120.25, gen_p75=150.0, prompt_p50=6695.1,
                           draft_acc_p50=0.62963, draft_len_p50=2.13, draft_n=3, last_ts=1700000000.5,
                           spec_types=["draft-mtp (max=4)"], single_config=False, config_count=2)
HISTORY = [telemetry.ConfigRun(instance="8080@1700000000", n=5, gen_p50=141.5, prompt_p50=6695.1,
                               last_ts=1700000000.5, argv={"--model": "/models/s/m.gguf", "--ctx-size": "8192"},
                               is_current=True, diff={"--ctx-size": "8192"}, rel_pct=100),
           telemetry.ConfigRun(instance="8080@1690000000", n=3, gen_p50=99.0, argv={"--model": "/models/s/m.gguf"},
                               diff={"--ctx-size": "unset"}, rel_pct=70)]
BACKENDS = [{"name": "synthetic-gpu", "vendor": "unknown", "vram_gb": 16.0, "gpu_count": 1,
             "card_vram_gb": [16.0], "host_ram_gb": 62.5, "baseline": {"ctx-size": "4096"},
             "baseline_args": ["--ctx-size", "4096"]}]


@pytest.fixture()
def world(monkeypatch, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "synthetic-m.gguf").write_bytes(_gguf("llama"))
    monkeypatch.setattr(settings, "model_loader_token", TOKEN)
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_ini_path", models / "models.ini")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(db, "_DB_PATH", None)
    db.init()
    monkeypatch.delenv(ENV, raising=False)
    calls = []

    def ingest(names, **_):
        calls.append(("ingest", list(names)))
        return 0

    def stats_for(*, model_path="", alias="", **_):
        calls.append(("stats", model_path, alias))
        return MEASURED

    def history(*, model_path="", alias="", **_):
        calls.append(("history", model_path, alias))
        return HISTORY

    def backend_list():
        calls.append(("backends",))
        return [dict(b) for b in BACKENDS]

    monkeypatch.setattr(services, "_effective_container_names", lambda: ["synthetic-gpu"])
    monkeypatch.setattr(telemetry, "ingest", ingest)
    monkeypatch.setattr(telemetry, "stats_for", stats_for)
    monkeypatch.setattr(telemetry, "config_history", history)
    monkeypatch.setattr(helpers, "_backend_list", backend_list)
    return TestClient(app, raise_server_exceptions=False), calls


def auth():
    return {"x-model-loader-token": TOKEN}


def runtime(client, body):
    return client.post(RUNTIME, content=json.dumps(body), headers={**auth(), "content-type": "application/json"})


def test_confirmed_child_refuses_the_read_and_serves_the_runtime(monkeypatch, world):
    client, calls = world
    monkeypatch.setenv(ENV, "1")
    assert client.get(PATH).status_code == 401
    answer = client.get(PATH, headers=auth())
    assert answer.status_code == 503 and answer.content == REFUSED
    for path in ("/api/v1/sections/a%20b/autoconfig", "/api/v1/sections/%C3%BC/autoconfig",
                 PATH + "?sessions=x"):
        assert client.get(path, headers=auth()).content == REFUSED
    assert client.post(PATH, headers=auth()).status_code == 405
    # A slash in the name and the section read itself are not this route.
    assert client.get("/api/v1/sections/a%2Fb/autoconfig", headers=auth()).status_code == 404
    assert client.get("/api/v1/sections/synthetic-m", headers=auth()).content != REFUSED
    assert calls == []

    got = runtime(client, {"section": "synthetic-m", "model": "/models/s/m.gguf"})
    assert got.status_code == 200 and got.headers["content-type"] == "application/json"
    # The telemetry is read before the backends, for this section and model path.
    assert calls == [("ingest", ["synthetic-gpu"]), ("stats", "/models/s/m.gguf", "synthetic-m"),
                     ("history", "/models/s/m.gguf", "synthetic-m"), ("backends",)]
    body = got.json()
    assert list(body) == ["measured", "history", "promptTps", "backends"]
    assert json.loads(body["backends"]) == BACKENDS and body["promptTps"] == "6695.1"
    assert body["measured"]["prompt_p50"] == 6695.1 and body["history"][1]["diff"] == {"--ctx-size": "unset"}
    assert client.get(RUNTIME, headers=auth()).status_code == 405
    assert client.post(RUNTIME).status_code == 401


BAD_BODIES = [b"", b"[]", b"not json", b'{"section": "m"}', b'{"section": 1, "model": ""}',
              b'{"section": "m", "model": "", "x": 1}', b'{"section": "m", "model": null}',
              b'{"section": "' + b"m" * (256 * 1024) + b'", "model": ""}']


@pytest.mark.parametrize("raw", BAD_BODIES)
def test_runtime_refuses_other_bodies(monkeypatch, world, raw):
    client, calls = world
    monkeypatch.setenv(ENV, "1")
    got = client.post(RUNTIME, content=raw, headers=auth())
    assert got.status_code == 422 and got.content == b'{"detail":"invalid runtime body"}'
    assert calls == []


def test_runtime_measurement_failure_gives_empty_ones(monkeypatch, world):
    client, _ = world
    monkeypatch.setenv(ENV, "1")

    def broken(*_, **__):
        raise RuntimeError("synthetic store failure")

    monkeypatch.setattr(telemetry, "stats_for", broken)
    body = runtime(client, {"section": "synthetic-m", "model": ""}).json()
    assert body["measured"] == json.loads(json.dumps(telemetry.Stats().__dict__)) and body["history"] == []
    assert json.loads(body["backends"]) == BACKENDS and body["promptTps"] == "0.0"


def test_runtime_backend_failure_is_a_500(monkeypatch, world):
    client, _ = world
    monkeypatch.setenv(ENV, "1")

    def broken():
        raise RuntimeError("synthetic discovery failure")

    monkeypatch.setattr(helpers, "_backend_list", broken)
    got = runtime(client, {"section": "synthetic-m", "model": ""})
    assert got.status_code == 500 and got.content == b"Internal Server Error"


def test_runtime_keeps_non_finite_backend_floats(monkeypatch, world):
    client, _ = world
    monkeypatch.setenv(ENV, "1")
    odd = [{"name": "g", "vram_gb": math.inf, "card_vram_gb": [math.nan]}]
    monkeypatch.setattr(helpers, "_backend_list", lambda: odd)
    got = runtime(client, {"section": "synthetic-m", "model": ""})
    assert got.status_code == 200
    assert got.json()["backends"] == '[{"name":"g","vram_gb":Infinity,"card_vram_gb":[NaN]}]'


def test_runtime_keeps_a_non_finite_prompt_rate(monkeypatch, world):
    client, _ = world
    monkeypatch.setenv(ENV, "1")
    huge = telemetry.Stats(n=5, prompt_p50=float("9" * 400))
    monkeypatch.setattr(telemetry, "stats_for", lambda **_: huge)
    body = runtime(client, {"section": "synthetic-m", "model": ""}).json()
    # The measurements show it as null (pydantic's JSON mode); the analysis input keeps it.
    assert body["measured"]["prompt_p50"] is None and body["promptTps"] == "inf"


def _raw(text: str, key: str) -> str:
    start = text.index('"' + key + '":') + len(key) + 3
    _, end = json.JSONDecoder().raw_decode(text, start)
    return text[start:end]


def test_runtime_measurements_render_as_the_route_does(monkeypatch, world):
    client, _ = world
    # The route itself (Python-owned) renders measured and history; the runtime gives the same text.
    route = client.get(PATH, headers=auth())
    assert route.status_code == 200, route.content
    monkeypatch.setenv(ENV, "1")
    got = runtime(client, {"section": "synthetic-m", "model": ""}).content.decode()
    for key in ("measured", "history"):
        assert _raw(got, key) == _raw(route.content.decode(), key)


@pytest.mark.parametrize("value", [None, "0", "true", " 1"])
def test_other_spellings_keep_python_ownership(monkeypatch, world, value):
    client, _ = world
    if value is not None:
        monkeypatch.setenv(ENV, value)
    answer = client.get(PATH, headers=auth())
    assert answer.status_code == 200, answer.content
    assert answer.json()["section"] == "synthetic-m"
    assert runtime(client, {"section": "synthetic-m", "model": ""}).status_code == 404
