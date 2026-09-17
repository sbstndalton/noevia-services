"""Tune must not propose a context this machine cannot use (2026-09-17: live Easy mode saved 262K)."""
import pytest
from fastapi.testclient import TestClient

from conftest import INI, ROOT, _gguf
from app import autoconfig, gguf_meta
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_ini():
    (ROOT / "models" / "models.ini").write_text(INI)

CANDS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]


def test_cap_context_uses_the_best_evidence_available():
    assert autoconfig.cap_context(262144, CANDS) == (32768, "prompt speed not measured yet; measure context to go higher")
    ctx, why = autoconfig.cap_context(262144, CANDS, prompt_tps=515, prompt_budget_s=120)
    assert ctx == 32768 and "515 tokens/s" in why          # 515 × 120 = 61 800 → largest candidate below
    ctx, why = autoconfig.cap_context(262144, CANDS, prompt_tps=2000, prompt_budget_s=120)
    assert ctx == 131072                                    # fast machines keep long windows
    assert autoconfig.cap_context(262144, CANDS + [24576], verified_ctx=24576)[0] == 24576
    assert autoconfig.cap_context(16384, CANDS) == (16384, "")  # memory binds first: no reason given
    assert autoconfig.cap_context(0, CANDS) == (0, "")


def test_verified_context_is_an_upper_bound_even_when_speed_would_allow_more():
    ctx, why = autoconfig.cap_context(262144, CANDS + [24576], verified_ctx=24576, prompt_tps=5000)
    assert ctx == 24576 and "verified" in why


def test_analyze_on_a_long_native_window_recommends_a_usable_context():
    path = ROOT / "models" / "tiny" / "long-native.gguf"
    path.write_bytes(_gguf({
        "general.architecture": "llama", "llama.context_length": 262144, "llama.embedding_length": 256,
        "llama.block_count": 4, "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
        "tokenizer.chat_template": "{{ messages }}"}) + b"\0" * 4096)
    summary = gguf_meta.summarize(gguf_meta.read_raw(path))
    backend = [{"name": "engine", "vendor": "unknown", "vram_gb": 14.0, "gpu_count": 1, "card_vram_gb": [14.0], "host_ram_gb": 29.0, "baseline": {}}]
    rec = autoconfig.analyze(summary=summary, file_size=4096, backends=backend, vision=False)
    assert rec.estimated_ctx == 262144
    assert rec.recommended_ctx == 32768 and rec.values["ctx-size"] == "32768"
    assert "not measured" in rec.ctx_cap_reason
    measured = autoconfig.analyze(summary=summary, file_size=4096, backends=backend, vision=False, prompt_tps=515)
    assert 32768 <= measured.recommended_ctx <= 515 * 120 and "515" in measured.ctx_cap_reason
    verified = autoconfig.analyze(summary=summary, file_size=4096, backends=backend, vision=False, verified_ctx=16384)
    assert verified.recommended_ctx == 16384 and verified.values["ctx-size"] == "16384"


def _nextn_model(name, layers):
    folder = ROOT / "models" / name
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.gguf"
    path.write_bytes(_gguf({
        "general.architecture": "qwen35", "qwen35.context_length": 8192, "qwen35.embedding_length": 256,
        "qwen35.block_count": 4, "qwen35.attention.head_count": 4, "qwen35.attention.head_count_kv": 2,
        "qwen35.nextn_predict_layers": layers, "tokenizer.chat_template": "{{ messages }}"}) + b"\0" * 4096)
    return path


def test_builtin_mtp_layers_enable_mtp_by_default_without_a_head_file(client):
    path = _nextn_model("builtin-nextn", 1)
    summary = gguf_meta.summarize(gguf_meta.read_raw(path))
    assert summary["model"]["nextn_predict_layers"] == 1
    backend = [{"name": "engine", "vendor": "unknown", "vram_gb": 14.0, "gpu_count": 1, "card_vram_gb": [14.0], "host_ram_gb": 29.0, "baseline": {}}]
    rec = autoconfig.analyze(summary=summary, file_size=4096, backends=backend, vision=False,
                             models_dir=ROOT / "models", section_name="builtin-nextn", model_subdir="builtin-nextn", mode="code")
    assert rec.values["spec-type"] == "draft-mtp,ngram-simple"          # code → Coding profile
    assert rec.values["spec-draft-model"] == ""
    chat = autoconfig.analyze(summary=summary, file_size=4096, backends=backend, vision=False,
                              models_dir=ROOT / "models", section_name="builtin-nextn", model_subdir="builtin-nextn", mode="chat")
    assert chat.values["spec-type"] == "draft-mtp"
    r = client.post("/api/v1/sections/builtin-nextn/safe-defaults").json()
    assert r["mtp"] is True and r["section"]["spec-type"] == "draft-mtp" and "spec-draft-model" not in r["section"]
    heads = client.get("/api/v1/sections/builtin-nextn/draft-heads").json()
    assert heads["available"] is True and heads["builtinLayers"] == 1 and heads["local"] == "" and heads["repo"] is None


def test_no_head_and_no_builtin_layers_keeps_speculation_off(client):
    path = _nextn_model("no-mtp", 0)
    summary = gguf_meta.summarize(gguf_meta.read_raw(path))
    backend = [{"name": "engine", "vendor": "unknown", "vram_gb": 14.0, "gpu_count": 1, "card_vram_gb": [14.0], "host_ram_gb": 29.0, "baseline": {}}]
    rec = autoconfig.analyze(summary=summary, file_size=4096, backends=backend, vision=False,
                             models_dir=ROOT / "models", section_name="no-mtp", model_subdir="no-mtp", mode="code")
    assert rec.values.get("spec-type", "") == ""
    r = client.post("/api/v1/sections/no-mtp/safe-defaults").json()
    assert r["mtp"] is False and "spec-type" not in r["section"]
    assert client.get("/api/v1/sections/no-mtp/draft-heads").json()["available"] is False


def test_an_mtp_named_model_is_not_its_own_draft_head_but_a_small_head_beside_it_is(client):
    folder = ROOT / "models" / "Big-MTP-Q4"
    folder.mkdir(exist_ok=True)
    main = folder / "Big-MTP-Q4.gguf"
    main.write_bytes(b"\0")
    import os
    os.truncate(main, autoconfig.HEAD_MAX_BYTES + 1)     # sparse: large on paper, no disk
    assert autoconfig._find_mtp(ROOT / "models", "Big-MTP-Q4", "Big-MTP-Q4") == ""
    head = folder / "mtp-Big-Q8_0.gguf"
    head.write_bytes(b"\0" * 1024)
    assert autoconfig._find_mtp(ROOT / "models", "Big-MTP-Q4", "Big-MTP-Q4") == "/models/Big-MTP-Q4/mtp-Big-Q8_0.gguf"
    assert client.post("/api/v1/sections/mtp-Big-Q8_0/safe-defaults").status_code == 400


def test_a_large_mtp_build_registers_as_a_model(client):
    folder = ROOT / "models" / "Huge-MTP-Q4"
    folder.mkdir(exist_ok=True)
    main = folder / "Huge-MTP-Q4.gguf"
    main.write_bytes(_gguf({"general.architecture": "llama", "llama.context_length": 8192, "llama.embedding_length": 256,
                            "llama.block_count": 4, "llama.attention.head_count": 4, "tokenizer.chat_template": "x"}))
    import os
    os.truncate(main, autoconfig.HEAD_MAX_BYTES + 4096)
    r = client.post("/api/v1/sections/Huge-MTP-Q4/safe-defaults")
    assert r.status_code == 200, r.text


def test_draft_head_download_only_takes_a_head_from_the_models_own_repo(client, monkeypatch):
    from types import SimpleNamespace
    from app import api, db
    _nextn_model("repo-model", 0)
    queued = []
    monkeypatch.setattr(db, "repo_for_file", lambda base: "synthetic/repo-model-GGUF" if base == "repo-model.gguf" else None)
    files = [SimpleNamespace(path="mtp-repo-model-Q8_0.gguf", size=120_000_000),
             SimpleNamespace(path="repo-model-MTP-Q4.gguf", size=autoconfig.HEAD_MAX_BYTES + 1),
             SimpleNamespace(path="mmproj-BF16.gguf", size=900_000_000)]

    async def repo_detail(repo, revision="main"):
        assert repo == "synthetic/repo-model-GGUF"
        return SimpleNamespace(files=files)
    monkeypatch.setattr(api.hf, "repo_detail", repo_detail)
    monkeypatch.setattr(api.manager, "enqueue", lambda **kw: queued.append(kw))
    heads = client.get("/api/v1/sections/repo-model/draft-heads").json()
    assert [h["path"] for h in heads["remote"]] == ["mtp-repo-model-Q8_0.gguf"]
    ok = client.post("/api/v1/sections/repo-model/draft-heads/download", json={"path": "mtp-repo-model-Q8_0.gguf"})
    assert ok.status_code == 200 and queued[0]["filename"] == "repo-model/mtp-repo-model-Q8_0.gguf"
    for bad in ("repo-model-MTP-Q4.gguf", "mmproj-BF16.gguf", "../../etc/passwd"):
        assert client.post("/api/v1/sections/repo-model/draft-heads/download", json={"path": bad}).status_code == 400
    assert len(queued) == 1
