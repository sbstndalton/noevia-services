"""#568: presets must not override the GGUF's own rope scaling, and duplicate [sections] are an error."""
import pytest
from fastapi.testclient import TestClient

from conftest import INI, ROOT
from app import autoconfig, gguf_meta, ini
from app.main import app

BACKEND = [{"name": "engine", "vendor": "unknown", "vram_gb": 24.0, "gpu_count": 1, "card_vram_gb": [24.0],
            "host_ram_gb": 64.0, "baseline": {}}]
ROPE_KEYS = ("rope-scaling", "rope-scale", "rope-freq-base", "rope-freq-scale")


def _raw(arch, ctx=131072, **extra):
    raw = {"general.architecture": arch, f"{arch}.context_length": ctx, f"{arch}.embedding_length": 256,
           f"{arch}.block_count": 4, f"{arch}.attention.head_count": 4, f"{arch}.attention.head_count_kv": 2,
           "tokenizer.chat_template": "{{ messages }}"}
    raw.update({f"{arch}.{k}": v for k, v in extra.items()})
    return raw


GEMMA3 = _raw("gemma3", **{"rope.scaling.type": "linear", "rope.scaling.factor": 8.0})
PLAIN = _raw("llama", ctx=8192)
DUP = "[a]\nctx-size = 1\n\n[b]\nctx-size = 2\n\n[a]\nctx-size = 3\n"


@pytest.fixture(autouse=True)
def reset_ini():
    (ROOT / "models" / "models.ini").write_text(INI)


def _rope(values):
    return [k for k in ROPE_KEYS if values.get(k)]


def test_gemma3_scaling_metadata_yields_no_rope_keys_in_defaults_or_autoconfig():
    summary = gguf_meta.summarize(GEMMA3)
    assert summary["model"]["rope_scaling_type"] == "linear" and summary["model"]["rope_scaling_factor"] == 8.0
    values, _ = ini.suggest_defaults(summary)
    assert not _rope(values)
    rec = autoconfig.analyze(summary=summary, file_size=4096, backends=BACKEND, vision=False)
    assert not _rope(rec.values)


def test_gemma3_gets_no_rope_keys_even_without_scaling_metadata():
    summary = gguf_meta.summarize(_raw("gemma3"))
    assert gguf_meta.rope_owned_by_gguf(summary["model"])
    values, _ = ini.suggest_defaults(summary)
    assert not _rope(values)


def test_declared_scaling_on_another_architecture_is_left_alone():
    summary = gguf_meta.summarize(_raw("qwen2", **{"rope.scaling.type": "yarn", "rope.scaling.factor": 4.0}))
    values, _ = ini.suggest_defaults(summary)
    assert not _rope(values)
    rec = autoconfig.analyze(summary=summary, file_size=4096, backends=BACKEND, vision=False)
    assert not _rope(rec.values)


def test_model_without_scaling_metadata_and_ctx_below_train_gets_no_rope_keys():
    summary = gguf_meta.summarize(PLAIN)
    assert not gguf_meta.rope_owned_by_gguf(summary["model"])
    values, _ = ini.suggest_defaults(summary)
    assert values["ctx-size"] == "8192" and not _rope(values)
    rec = autoconfig.analyze(summary=summary, file_size=4096, backends=BACKEND, vision=False)
    assert rec.recommended_ctx <= 8192
    assert not _rope(rec.values)


def test_duplicate_detection_on_load_and_validation():
    assert ini.duplicate_sections(DUP) == ["a"]
    assert ini.duplicate_sections("[a]\n[b]\n[a]\n[b]\n[a]\n") == ["a", "b"]
    assert ini.duplicate_sections(INI) == []
    with pytest.raises(ini.DuplicateSectionsError, match=r"\[a\]"):
        ini.parse_ini_text(DUP)
    # Read-only views stay tolerant so a damaged file can still be shown and repaired.
    assert ini.parse_ini_text(DUP, allow_duplicates=True).sections() == ["a", "b"]


def test_writers_refuse_a_file_that_already_has_duplicate_sections():
    path = ROOT / "models" / "models.ini"
    path.write_text(DUP)
    for call in (lambda: ini.upsert_section("a", {"ctx-size": "9"}, ""),
                 lambda: ini.delete_section("b"),
                 lambda: ini.rename_section("b", "c")):
        with pytest.raises(ini.DuplicateSectionsError):
            call()
    assert path.read_text() == DUP


def test_upsert_replaces_an_existing_section_instead_of_appending_a_second():
    ini.upsert_section("tiny", {"ctx-size": "2048"}, "")
    text = (ROOT / "models" / "models.ini").read_text()
    assert text.count("[tiny]") == 1 and ini.duplicate_sections(text) == []
    assert ini.get_section("tiny")["ctx-size"] == "2048"


def test_put_models_ini_rejects_duplicate_sections_and_writes_nothing():
    with TestClient(app) as c:
        r = c.put("/api/v1/models-ini", json={"text": DUP, "baseRevision": "x"})
        assert r.status_code == 400 and "duplicate sections: [a]" in r.json()["detail"]
    assert (ROOT / "models" / "models.ini").read_text() == INI
