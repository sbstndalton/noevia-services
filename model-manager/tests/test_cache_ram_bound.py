"""#697: every section write leaves an explicit, bounded prompt cache (--cache-ram), autoconfig
never suggests one above the cap, and autoconfig sizes against noevia's inference budget."""
import pytest
from fastapi.testclient import TestClient

from conftest import INI, ROOT, _gguf
from app import api, autoconfig, gguf_meta, ini
from app.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_ini():
    (ROOT / "models" / "models.ini").write_text(INI)


def _save(client, values, extras=""):
    current = client.get("/api/v1/sections/tiny").json()
    r = client.put("/api/v1/sections/tiny", json={"baseRevision": current["revision"], "values": {**current["values"], **values}, "extras": extras})
    assert r.status_code == 200, r.text
    return r.json()["section"]


def test_limits_default_to_1024_and_2048_and_the_cap_never_exceeds_the_maximum(monkeypatch):
    assert settings.cache_ram_limits == (1024, 2048)
    monkeypatch.setattr(settings, "llamacpp_autoconfig_cache_ram_max_mib", 4096)
    assert settings.cache_ram_limits == (2048, 2048)


def test_clamp_turns_unbounded_and_oversized_values_into_the_hard_maximum():
    assert ini.clamp_cache_ram("8192") == "2048"
    assert ini.clamp_cache_ram("-1") == "2048"
    assert ini.clamp_cache_ram(" 512 ") == "512"
    assert ini.clamp_cache_ram("0") == "0"
    assert ini.clamp_cache_ram("lots") == "lots"  # not ours to reinterpret


def test_a_saved_section_without_cache_ram_gets_the_cap(client):
    assert "cache-ram" not in ini.get_section("tiny")
    assert _save(client, {"ctx-size": "8192"})["cache-ram"] == "1024"


@pytest.mark.parametrize("given,stored", [("8192", "2048"), ("-1", "2048"), ("2048", "2048"), ("512", "512"), ("0", "0")])
def test_a_saved_cache_ram_is_clamped_to_the_hard_maximum(client, given, stored):
    assert _save(client, {"cache-ram": given})["cache-ram"] == stored


def test_extras_cannot_slip_an_unbounded_cache_past_the_clamp(client):
    assert _save(client, {"cache-ram": ""}, extras="cache-ram = 16384")["cache-ram"] == "2048"


def test_a_bounded_global_default_is_inherited_instead_of_duplicated(client):
    (ROOT / "models" / "models.ini").write_text("version = 1\n\n[*]\ncache-ram = 512\n" + INI.split("version = 1\n", 1)[1])
    assert "cache-ram" not in _save(client, {"ctx-size": "8192"})
    (ROOT / "models" / "models.ini").write_text("version = 1\n\n[*]\ncache-ram = 8192\n" + INI.split("version = 1\n", 1)[1])
    assert _save(client, {"ctx-size": "8192"})["cache-ram"] == "1024"


def test_safe_defaults_registration_writes_the_cap(client):
    folder = ROOT / "models" / "fresh-Q4"
    folder.mkdir(exist_ok=True)
    (folder / "fresh-Q4.gguf").write_bytes(_gguf({
        "general.architecture": "llama", "llama.context_length": 32768, "llama.embedding_length": 256,
        "llama.block_count": 4, "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
        "tokenizer.chat_template": "{{ messages }}"}) + b"\0" * 4096)
    r = client.post("/api/v1/sections/fresh-Q4/safe-defaults")
    assert r.status_code == 200, r.text
    assert r.json()["section"]["cache-ram"] == "1024"


def test_autoconfig_never_suggests_a_cache_above_the_cap():
    path = ROOT / "models" / "tiny" / "tiny-Q4_K_M.gguf"
    summary = gguf_meta.summarize(gguf_meta.read_raw(path))
    # Plenty of host RAM: before #697 this suggested llama-server's 8192 MiB default or more.
    backend = [{"name": "engine", "vendor": "unknown", "vram_gb": 14.0, "gpu_count": 1, "card_vram_gb": [14.0], "host_ram_gb": 128.0, "baseline": {}}]
    rec = autoconfig.analyze(summary=summary, file_size=4096, backends=backend, vision=False)
    assert rec.values.get("cache-ram") and int(rec.values["cache-ram"]) <= 1024


def test_autoconfig_backends_are_sized_against_the_budget_less_the_prompt_cache():
    backends = [{"name": "engine", "vram_gb": 30.0, "card_vram_gb": [20.0, 10.0], "host_ram_gb": 64.0}]
    sized = api.budget_backends(backends, 16)
    assert sized[0]["vram_gb"] == 15.0                       # 16 GiB less the 1 GiB cache cap
    assert sized[0]["card_vram_gb"] == [10.0, 5.0]
    assert backends[0]["vram_gb"] == 30.0                    # the discovered list is not mutated
    assert api.budget_backends(backends, 0) is backends      # no budget sent: unchanged
    assert api.budget_backends(backends, "junk") is backends
    small = api.budget_backends([{"name": "e", "vram_gb": 8.0}], 16)
    assert small[0]["vram_gb"] == 8.0                        # never raised above the hardware


@pytest.mark.parametrize("extras,stored", [
    ("LLAMA_ARG_CACHE_RAM = 8192", "2048"),
    ("cram = -1", "2048"),
    ("LLAMA_ARG_CACHE_RAM = 512", "512"),
])
def test_cache_ram_aliases_are_folded_into_the_canonical_key_before_clamping(client, extras, stored):
    section = _save(client, {"cache-ram": ""}, extras=extras)
    assert section["cache-ram"] == stored
    assert not any(k.lower() in ("cram", "llama_arg_cache_ram") for k in section), section


def test_an_alias_beside_the_canonical_key_cannot_hide_a_larger_value(client):
    section = _save(client, {"cache-ram": "512"}, extras="LLAMA_ARG_CACHE_RAM = 16384")
    assert section["cache-ram"] == "2048" and "LLAMA_ARG_CACHE_RAM" not in section


def test_startup_migration_bounds_chat_sections_zeroes_embed_rerank_and_keeps_a_backup(monkeypatch):
    path = ROOT / "models" / "models.ini"
    original = (
        "version = 1\n\n"
        "[chat-a]\nmodel = /models/a.gguf\nctx-size = 8192\n\n"
        "[chat-b]\nmodel = /models/b.gguf\nLLAMA_ARG_CACHE_RAM = 8192\n\n"
        "[chat-ok]\nmodel = /models/c.gguf\ncache-ram = 1024\n\n"
        "[nomic-embed]\nmodel = /models/n.gguf\nembedding = true\n\n"
        "[reranker]\nmodel = /models/r.gguf\nreranking = true\n\n"
        "[laya_multilingual_f16]\nmodel = /models/laya_multilingual_f16/laya.gguf\n")
    path.write_text(original)
    backed_up = []
    real = ini._backup_current
    monkeypatch.setattr(ini, "_backup_current", lambda p: (backed_up.append(p.read_text()), real(p)))
    assert ini.migrate_cache_ram() == ["chat-a", "chat-b", "nomic-embed", "reranker"]
    assert ini.get_section("chat-a")["cache-ram"] == "1024"
    assert ini.get_section("chat-b")["cache-ram"] == "2048" and "LLAMA_ARG_CACHE_RAM" not in ini.get_section("chat-b")
    # #723: no prompt cache for embedding/reranking, written explicitly; Laya stays untouched.
    for non_chat in ("nomic-embed", "reranker"):
        assert ini.get_section(non_chat)["cache-ram"] == "0", non_chat
    assert "cache-ram" not in ini.get_section("laya_multilingual_f16")
    assert backed_up == [original], "the normal writer backed up the file first"
    text = path.read_text()
    assert ini.migrate_cache_ram() == [] and path.read_text() == text, "idempotent: nothing rewritten"


def test_container_memory_reports_anonymous_memory_without_page_cache():
    from app import hw

    class Fake:
        def __init__(self, stats): self._s = stats
        def stats(self, stream=False): return self._s
    gib = 1024 ** 3
    v2 = hw._read_container_runtime(Fake({"memory_stats": {"usage": 9 * gib, "limit": 14 * gib, "stats": {"inactive_file": 5 * gib, "anon": 3 * gib}}}))
    assert v2.mem_anon_gb == 3.0 and v2.mem_used_gb == 4.0
    v1 = hw._read_container_runtime(Fake({"memory_stats": {"usage": 9 * gib, "stats": {"cache": 5 * gib, "rss": 2 * gib}}}))
    assert v1.mem_anon_gb == 2.0
    assert hw._read_container_runtime(Fake({"memory_stats": {"usage": gib, "stats": {}}})).mem_anon_gb is None
