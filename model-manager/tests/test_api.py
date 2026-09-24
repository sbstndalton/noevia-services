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
    assert sections["raw"].startswith("version = 1") and "[tiny]" in sections["raw"]
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


def test_overview_reports_models_folder_disk_space(client):
    models_dir = client.get("/api/v1/overview").json()["modelsDir"]
    assert models_dir["hostPath"] is None
    disk = models_dir["disk"]
    assert disk["free"] > 0 and disk["total"] >= disk["free"] and disk["freeH"]


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


def test_engine_log_lines_are_scrubbed_before_leaving_the_server(client, monkeypatch):
    from app import api, services
    leaked = "\n".join([
        "srv  load_model: loading /models/q/Qwen.gguf",
        "request headers: Authorization: Bearer abcdefghijklmnop123456",
        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz0123",
        'config {"api_key": "sk-live-abcdefghijklmnopqrstuv", "ctx": 8192}',
        "remote https://admin:hunter2secret@nextcloud.example/remote.php",
        "cookie cowork_session=deadbeefcafebabe1234 ok",
    ])
    monkeypatch.setattr(services, "_effective_container_names", lambda: ["cowork-llama-1"])
    monkeypatch.setattr(services, "container_logs", lambda name, tail=400: (True, leaked))
    body = client.get("/api/v1/backends/cowork-llama-1/logs").json()
    text = "\n".join(body["lines"])
    for secret in ("abcdefghijklmnop123456", "hf_abcdefghijklmnopqrstuvwxyz0123", "sk-live-abcdefghijklmnopqrstuv", "hunter2secret", "deadbeefcafebabe1234"):
        assert secret not in text
    assert "loading /models/q/Qwen.gguf" in text and '"ctx": 8192' in text
    # Filtering on a secret's value must not reveal that the line held it.
    assert client.get("/api/v1/backends/cowork-llama-1/logs?q=hunter2").json()["lines"] == []
    assert api.redact_log_line("token: abc123xyz") == "token: [redacted]"
    # Counters and numeric fields are not secrets.
    for line in ("n_tokens = 512", "prompt tokens: 40", "slot session: 3", "n_ctx_slot = 8192"):
        assert api.redact_log_line(line) == line


def test_download_targets_are_limited_to_declared_folders_inside_models(client, monkeypatch):
    from app.config import settings
    (ROOT / "models" / "archive").mkdir(exist_ok=True)
    monkeypatch.setattr(settings, "model_download_targets", "archive,missing,../etc,.hidden")
    targets = client.get("/api/v1/download-targets").json()["targets"]
    assert [t["id"] for t in targets] == ["", "archive"]
    queued = []
    from app import api
    monkeypatch.setattr(api.manager, "enqueue_url", lambda **kw: queued.append(kw["filename"]))
    assert client.post("/api/v1/downloads", json={"url": "https://example.invalid/x.gguf", "target": "archive"}).status_code == 200
    assert queued == ["archive/x/x.gguf"]
    for bad in ("missing", "../etc", "/abs", "tiny"):
        assert client.post("/api/v1/downloads", json={"url": "https://example.invalid/x.gguf", "target": bad}).status_code == 400


def test_hugging_face_cache_layout_never_surfaces_hex_names(client):
    """models--org--repo/snapshots/<commit>/file.gguf are symlinks into blobs/<sha256>."""
    import os
    from conftest import _gguf
    repo = ROOT / "models" / "models--synthetic--cache-GGUF"
    commit = "0123456789abcdef0123456789abcdef01234567"
    sha = "a" * 64
    (repo / "blobs").mkdir(parents=True, exist_ok=True)
    (repo / "snapshots" / commit).mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs" / "main").write_text(commit)
    (repo / "blobs" / sha).write_bytes(_gguf({"general.architecture": "llama", "llama.context_length": 4096, "llama.embedding_length": 64,
        "llama.block_count": 2, "llama.attention.head_count": 2, "llama.attention.head_count_kv": 1}) + b"\0" * 1024)
    link = repo / "snapshots" / commit / "cache-model-Q4_K_M.gguf"
    if not link.exists():
        os.symlink(os.path.join("..", "..", "blobs", sha), link)
    hexlike = lambda s: bool(__import__("re").fullmatch(r"[0-9a-f]{32,64}", s or ""))
    models = client.get("/api/v1/models").json()["models"]
    names = [m["name"] for m in models] + [m.get("modelId") or "" for m in models] + [s for m in models for s in m.get("sections", [])]
    assert not any(hexlike(n) or hexlike(n.rsplit(".", 1)[0]) for n in names), names
    unregistered = client.get("/api/v1/sections").json()["unregistered"]
    assert not any(hexlike(n) for n in unregistered), unregistered
    print("HF-CACHE", [(m["name"], m.get("modelId"), m.get("subdir")) for m in models], unregistered)


def test_token_guards_every_route_except_health(client, monkeypatch):
    from app.config import settings
    token = "synthetic-loader-token-with-32-characters"
    monkeypatch.setattr(settings, "model_loader_token", token)
    assert client.get("/api/v1/health").status_code == 200
    for path in ("/api/v1/sections", "/api/v1/backends", "/api/v1/downloads", "/", "/containers"):
        assert client.get(path).status_code == 401, path
    assert client.post("/api/v1/backends/x/restart").status_code == 401
    assert client.get("/api/v1/sections", headers={"X-Model-Loader-Token": "wrong"}).status_code == 401
    assert client.get("/api/v1/sections", headers={"X-Model-Loader-Token": token}).status_code == 200
    monkeypatch.setattr(settings, "model_loader_token", "")
    assert client.get("/api/v1/sections").status_code == 200


def test_token_configuration_rejects_short_or_whitespace_secrets():
    from pydantic import ValidationError
    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(model_loader_token="   ")
    with pytest.raises(ValidationError):
        Settings(model_loader_token="too-short")
    assert Settings(model_loader_token="  " + "x" * 32 + "  ").model_loader_token == "x" * 32


def test_search_reports_a_hub_failure_instead_of_raising(client, monkeypatch):
    """Discover must catch what search_models actually raises.

    search_models used to surface httpx errors; it now raises HfSearchError with a message
    that names the fix. An `except httpx.…` here would match nothing, so a rate-limited or
    unreachable hub would 500 instead of returning a readable error — and it would fail
    silently, because the types simply stop lining up.
    """
    from app import hf

    async def boom(*args, **kwargs):
        raise hf.HfSearchError("Hugging Face returned HTTP 429. Hugging Face is rate limiting this server; add a token to raise the limit.")

    monkeypatch.setattr(hf, "search_models", boom)
    r = client.get("/api/v1/search", params={"q": "qwen"})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    assert "429" in body["error"] and "token" in body["error"]


def test_search_survives_an_avatar_failure(client, monkeypatch):
    """Avatars are decoration and must never fail the search that carries them."""
    from app import hf

    async def no_models(*args, **kwargs):
        return []

    async def bad_avatars(*args, **kwargs):
        raise RuntimeError("avatar cache is unavailable")

    monkeypatch.setattr(hf, "search_models", no_models)
    monkeypatch.setattr(hf, "owner_avatars", bad_avatars)
    r = client.get("/api/v1/search", params={"q": "qwen"})
    assert r.status_code == 200 and r.json()["results"] == []


def _fake_repo(monkeypatch, files):
    from app import api, hf
    detail = hf.HfRepoDetail(id="synthetic/repo-GGUF", files=[
        hf.HfFile(path=p, size=10, quant=None, shard_base=b, shard_index=None, shard_total=None) for p, b in files],
        readme_snippet=None)

    async def repo_detail(repo):
        return detail
    monkeypatch.setattr(hf, "repo_detail", repo_detail)
    queued = []
    monkeypatch.setattr(api.manager, "enqueue", lambda **kw: queued.append(kw["filename"]))
    return queued


def test_a_shard_base_the_repo_does_not_list_is_refused_and_queues_nothing(client, monkeypatch):
    queued = _fake_repo(monkeypatch, [("m-Q4.gguf", "m-Q4.gguf"), ("mmproj-f16.gguf", "mmproj-f16.gguf")])
    for bad in ("..", "../..", "/etc/passwd", "nope.gguf", "."):
        r = client.post("/api/v1/downloads", json={"repo": "synthetic/repo-GGUF", "shardBase": bad})
        assert r.status_code == 400, bad
    assert queued == []
    ok = client.post("/api/v1/downloads", json={"repo": "synthetic/repo-GGUF", "shardBase": "m-Q4.gguf"})
    assert ok.status_code == 200 and queued == ["m-Q4/m-Q4.gguf", "m-Q4/mmproj-f16.gguf"]


def test_a_listed_shard_base_that_escapes_is_still_refused(client, monkeypatch):
    queued = _fake_repo(monkeypatch, [("x/...gguf", "..")])
    r = client.post("/api/v1/downloads", json={"repo": "synthetic/repo-GGUF", "shardBase": ".."})
    assert r.status_code == 400 and queued == []


def test_the_downloader_refuses_any_destination_outside_the_models_dir():
    from app.downloader import manager, safe_dest
    for bad in ("../x.gguf", "a/../../x.gguf", "/etc/x.gguf", "", "."):
        with pytest.raises(ValueError):
            safe_dest(bad)
        with pytest.raises(ValueError):
            manager._make_job(repo_id="r", filename=bad, url="https://example.invalid/x", total_bytes=0)
    assert safe_dest("archive/m/m.gguf").name == "m.gguf"


def test_hub_errors_are_not_echoed(client, monkeypatch):
    from app import hf

    import httpx

    class Boom:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **kw): raise httpx.ConnectError("secret-internal-detail 10.0.0.9:443")
    monkeypatch.setattr(hf.httpx, "AsyncClient", Boom)
    r = client.get("/api/v1/search?q=x-unique-query").json()
    assert "secret-internal-detail" not in r["error"] and "Could not reach" in r["error"] and r["results"] == []
