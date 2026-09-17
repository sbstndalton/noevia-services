from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import INI, ROOT
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_ini():
    (ROOT / "models" / "models.ini").write_text(INI)


def test_health_models_and_sections(client):
    assert client.get("/api/v1/health").json() == {"ok": True}
    models = client.get("/api/v1/models").json()
    [m] = models["models"]
    assert m["name"] == "tiny-Q4_K_M.gguf" and m["sections"] == ["tiny"] and m["shape"]["label"] == "dense"
    sections = client.get("/api/v1/sections").json()
    assert [s["name"] for s in sections["sections"]] == ["tiny"]
    assert sections["schema"][0]["tier"] == "Common" and any(f["key"] == "ctx-size" for f in sections["schema"][0]["fields"])


def test_save_is_pinned_to_the_revision_and_keeps_the_preamble(client):
    current = client.get("/api/v1/sections/tiny").json()
    assert current["values"]["ctx-size"] == "4096"
    stale = current["revision"]
    ok = client.put("/api/v1/sections/tiny", json={"baseRevision": stale, "values": {**current["values"], "ctx-size": "8192"}, "extras": ""})
    assert ok.status_code == 200
    text = (ROOT / "models" / "models.ini").read_text()
    assert text.startswith("version = 1\n") and "ctx-size = 8192" in text
    again = client.put("/api/v1/sections/tiny", json={"baseRevision": stale, "values": {"ctx-size": "2048"}})
    assert again.status_code == 409
    assert client.put("/api/v1/sections/tiny", json={"values": {}}).status_code == 400
    bad = client.put("/api/v1/sections/tiny", json={"baseRevision": ok.json()["revision"], "values": {"ctx-size": "1\n[evil]"}})
    assert bad.status_code == 400
    extras = client.put("/api/v1/sections/tiny", json={"baseRevision": ok.json()["revision"], "values": {}, "extras": "[evil]\nx = 1"})
    assert extras.status_code == 400


def test_rename_and_delete(client):
    rev = client.get("/api/v1/sections").json()["revision"]
    assert client.post("/api/v1/sections/tiny/rename", json={"newName": "tiny-chat", "baseRevision": rev}).status_code == 200
    rev = client.get("/api/v1/sections").json()["revision"]
    assert [s["name"] for s in client.get("/api/v1/sections").json()["sections"]] == ["tiny-chat"]
    assert client.delete(f"/api/v1/sections/tiny-chat?baseRevision={rev}").status_code == 200
    assert client.get("/api/v1/sections").json()["sections"] == []
    assert (ROOT / "models" / "models.ini").read_text().startswith("version = 1")


def test_autoconfig_without_a_gpu_backend_explains_itself(client):
    r = client.get("/api/v1/sections/tiny/autoconfig?sessions=2&vision=false").json()
    assert r["arch"] == "llama"
    assert "No GPU backend" in r["recommendation"]["error"]
    assert client.get("/api/v1/sections/missing/autoconfig").json()["error"].startswith("No model file")


def test_prompts_badges_downloads_and_host(client):
    added = client.post("/api/v1/prompts", json={"name": "Synthetic", "body": "Say OK."}).json()
    pid = added["id"]
    assert any(p["name"] == "Synthetic" for p in added["prompts"])
    assert client.post("/api/v1/prompts", json={"name": "", "body": ""}).status_code == 400
    assert all(p["id"] != pid for p in client.delete(f"/api/v1/prompts/{pid}").json()["prompts"])
    badges = client.put("/api/v1/badges", json={"alias": "tiny", "category": "coding", "rating": 4, "note": "synthetic"}).json()["badges"]
    assert badges[0]["rating"] == 4
    assert client.put("/api/v1/badges", json={"alias": "tiny", "category": "nonsense", "rating": 4}).status_code == 400
    assert client.delete("/api/v1/badges?alias=tiny&category=coding").json()["badges"] == []
    assert client.get("/api/v1/downloads").json() == {"jobs": []}
    assert client.post("/api/v1/downloads", json={"url": "ftp://x"}).status_code == 400
    assert "history" in client.get("/api/v1/host").json()
    assert client.get("/api/v1/backends/nope/diagnose").status_code == 404
    assert client.get("/api/v1/benchmark").json()["categories"][0]["key"] == "coding"


def test_model_keys_cannot_escape_the_models_dir(client):
    assert client.get("/api/v1/models/detail?key=../etc").status_code == 400
    assert client.get("/api/v1/models/detail?key=tiny/tiny-Q4_K_M.gguf").json()["summary"]["arch"] == "llama"
    assert client.post("/api/v1/models/delete", json={"models": ["../x"]}).json()["results"][0]["ok"] is False


def test_hugging_face_cache_layout_is_listed_configured_and_deleted(client):
    import os
    from conftest import _gguf
    repo = ROOT / "models" / "models--acme--cache-GGUF"
    snap = repo / "snapshots" / "abc123"
    (repo / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    blob = repo / "blobs" / "sha-model"
    blob.write_bytes(_gguf({"general.architecture": "llama", "llama.context_length": 4096, "llama.embedding_length": 256,
                            "llama.block_count": 2, "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
                            "tokenizer.chat_template": "x"}) + b"\0" * 2048)
    (repo / "blobs" / "sha-proj").write_bytes(b"GGUF" + b"\0" * 64)
    os.symlink("../../blobs/sha-model", snap / "cache-Q4_K_M.gguf")
    os.symlink("../../blobs/sha-proj", snap / "mmproj-F16.gguf")
    listed = client.get("/api/v1/models").json()
    entry = next(m for m in listed["models"] if m["name"] == "cache-Q4_K_M.gguf")
    assert entry["subdir"] == "models--acme--cache-GGUF/snapshots/abc123" and entry["projector"]["name"] == "mmproj-F16.gguf"
    assert "cache-Q4_K_M" in listed["unregistered"]
    defaults = client.get("/api/v1/sections/cache-Q4_K_M?defaults=true").json()
    assert defaults["values"]["model"] == "/models/models--acme--cache-GGUF/snapshots/abc123/cache-Q4_K_M.gguf"
    assert defaults["values"]["mmproj"] == "/models/models--acme--cache-GGUF/snapshots/abc123/mmproj-F16.gguf"
    assert client.get(f"/api/v1/models/detail?key={entry['key']}").json()["summary"]["arch"] == "llama"
    result = client.post("/api/v1/models/delete", json={"models": [entry["key"]]}).json()["results"][0]
    assert result["ok"] and result["freed"] >= 2048
    assert not repo.exists()


def _fresh_download(stem: str, ctx: int, head: bool = False) -> None:
    from conftest import _gguf
    folder = ROOT / "models" / stem
    folder.mkdir(exist_ok=True)
    (folder / f"{stem}-Q4_K_M.gguf").write_bytes(_gguf({
        "general.architecture": "llama", "llama.context_length": ctx, "llama.embedding_length": 256,
        "llama.block_count": 4, "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
        "tokenizer.chat_template": "{{ messages }}"}) + b"\0" * 4096)
    if head:
        (folder / f"{stem}-mtp-Q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 64)


def test_safe_defaults_register_once_with_capped_context_and_detected_mtp(client):
    _fresh_download("big", 131072, head=True)
    r = client.post("/api/v1/sections/big-Q4_K_M/safe-defaults")
    assert r.status_code == 200, r.text
    section = r.json()["section"]
    assert section["ctx-size"] == "8192" and section["jinja"] == "true"
    assert section["spec-type"] == "draft-mtp" and section["spec-draft-model"].endswith("big-mtp-Q8_0.gguf")
    assert not any(k in section for k in ("temp", "top-k", "top-p", "min-p"))
    assert client.post("/api/v1/sections/big-Q4_K_M/safe-defaults").status_code == 409


def test_safe_defaults_keep_small_context_and_skip_mtp_without_a_head(client):
    _fresh_download("small", 2048)
    section = client.post("/api/v1/sections/small-Q4_K_M/safe-defaults").json()["section"]
    assert section["ctx-size"] == "2048" and "spec-type" not in section


def test_safe_defaults_never_touch_existing_or_unrelated_files(client):
    before = (ROOT / "models" / "models.ini").read_text()
    assert client.post("/api/v1/sections/tiny/safe-defaults").status_code == 409
    assert client.post("/api/v1/sections/tiny-Q4_K_M/safe-defaults").status_code == 409
    assert client.post("/api/v1/sections/absent/safe-defaults").status_code == 404
    assert client.post("/api/v1/sections/big-mtp-Q8_0/safe-defaults").status_code == 400
    assert (ROOT / "models" / "models.ini").read_text() == before
