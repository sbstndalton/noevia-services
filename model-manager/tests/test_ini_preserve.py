"""#800: section saves, renames, deletes, safe defaults and the startup cache-ram migration edit
models.ini in place. Comments, blank lines, the preamble and every untouched section stay
byte-for-byte. Synthetic sections only."""
import shutil

import pytest
from fastapi.testclient import TestClient

from conftest import INI, ROOT, _gguf
from app import ini
from app.main import app

PATH = ROOT / "models" / "models.ini"

ORIGINAL = (
    "; models.ini for the synthetic test host\n"
    "version = 1\n"
    "\n"
    "# global defaults\n"
    "[*]\n"
    "cache-ram = 1024 \n"
    "\n"
    "[tiny]\n"
    "; the tiny model - keep ctx small\n"
    "model = /models/tiny/tiny-Q4_K_M.gguf\n"
    "ctx-size = 4096\n"
    "cache-ram = 512\n"
    "# trailing note inside tiny\n"
    "\n"
    "# --- section B: operator notes ---\n"
    "[b]\n"
    "model = /models/b.gguf\n"
    "; a comment in section B\n"
    "temp = 0.6\n"
    "temp = 0.7\n"
    "cache-ram = 256\n"
    "\n"
    "# end of file\n"
)


@pytest.fixture(autouse=True)
def _restore():
    PATH.write_text(ORIGINAL)
    yield
    PATH.write_text(INI)
    shutil.rmtree(ROOT / "models" / "fresh-dl", ignore_errors=True)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _rev(client):
    return client.get("/api/v1/sections").json()["revision"]


def test_saving_one_section_keeps_every_comment_and_the_other_section_verbatim(client):
    current = client.get("/api/v1/sections/tiny").json()
    r = client.put("/api/v1/sections/tiny", json={"baseRevision": current["revision"],
                                                   "values": {**current["values"], "ctx-size": "8192"},
                                                   "extras": current["extras"]})
    assert r.status_code == 200, r.text
    assert PATH.read_text() == ORIGINAL.replace("ctx-size = 4096\n", "ctx-size = 8192\n")


def test_adding_a_key_lands_after_the_sections_last_key():
    ini.upsert_section("tiny", {"model": "/models/tiny/tiny-Q4_K_M.gguf", "ctx-size": "4096", "cache-ram": "512",
                                "ngl": "99"}, "")
    assert PATH.read_text() == ORIGINAL.replace("cache-ram = 512\n", "cache-ram = 512\nngl = 99\n")


def test_removing_a_key_removes_its_line_and_nothing_else():
    ini.upsert_section("tiny", {"model": "/models/tiny/tiny-Q4_K_M.gguf", "cache-ram": "512"}, "")
    assert PATH.read_text() == ORIGINAL.replace("ctx-size = 4096\n", "")


def test_earlier_duplicate_keys_survive_an_unrelated_edit_and_the_effective_line_is_the_one_changed():
    ini.upsert_section("b", {"model": "/models/b.gguf", "cache-ram": "128"}, "temp = 0.7")
    assert PATH.read_text() == ORIGINAL.replace("cache-ram = 256\n", "cache-ram = 128\n")
    ini.upsert_section("b", {"model": "/models/b.gguf", "cache-ram": "128"}, "temp = 0.9")
    text = PATH.read_text()
    assert "temp = 0.6\ntemp = 0.9\n" in text and ini.get_section("b")["temp"] == "0.9"


def test_dropping_a_duplicated_key_removes_every_copy():
    ini.upsert_section("b", {"model": "/models/b.gguf", "cache-ram": "256"}, "")
    assert PATH.read_text() == ORIGINAL.replace("temp = 0.6\ntemp = 0.7\n", "")
    assert "temp" not in ini.get_section("b")


def test_cache_ram_alias_fold_keeps_comments():
    PATH.write_text(ORIGINAL.replace("cache-ram = 256\n", "# big cache on purpose\nLLAMA_ARG_CACHE_RAM = 16384\n"))
    ini.upsert_section("b", {"model": "/models/b.gguf"}, "temp = 0.7\nLLAMA_ARG_CACHE_RAM = 16384")
    assert PATH.read_text() == ORIGINAL.replace("cache-ram = 256\n", "# big cache on purpose\ncache-ram = 2048\n")


def test_rename_changes_only_the_header(client):
    r = client.post("/api/v1/sections/b/rename", json={"newName": "b-renamed", "baseRevision": _rev(client)})
    assert r.status_code == 200, r.text
    assert PATH.read_text() == ORIGINAL.replace("[b]\n", "[b-renamed]\n")


def test_delete_removes_the_section_and_keeps_the_neighbours_comments(client):
    r = client.delete(f"/api/v1/sections/tiny?baseRevision={_rev(client)}")
    assert r.status_code == 200, r.text
    text = PATH.read_text()
    assert text == ORIGINAL.replace(
        "[tiny]\n; the tiny model - keep ctx small\nmodel = /models/tiny/tiny-Q4_K_M.gguf\n"
        "ctx-size = 4096\ncache-ram = 512\n", "")
    assert "# --- section B: operator notes ---" in text and "; a comment in section B" in text


def test_deleting_the_last_section_leaves_no_dangling_gap():
    PATH.write_text("version = 1\n\n# keep me\n[a]\nx = 1\n\n[b]\ny = 2\n")
    assert ini.delete_section("b")
    assert PATH.read_text() == "version = 1\n\n# keep me\n[a]\nx = 1\n"
    assert ini.delete_section("a")
    assert PATH.read_text() == "version = 1\n\n# keep me\n"


def test_safe_defaults_append_a_section_without_touching_the_rest(client):
    folder = ROOT / "models" / "fresh-dl"
    folder.mkdir(exist_ok=True)
    (folder / "fresh-dl-Q4_K_M.gguf").write_bytes(_gguf({
        "general.architecture": "llama", "llama.context_length": 4096, "llama.embedding_length": 256,
        "llama.block_count": 2, "llama.attention.head_count": 4, "llama.attention.head_count_kv": 2,
        "tokenizer.chat_template": "{{ messages }}"}) + b"\0" * 512)
    r = client.post("/api/v1/sections/fresh-dl-Q4_K_M/safe-defaults")
    assert r.status_code == 200, r.text
    text = PATH.read_text()
    assert text.startswith(ORIGINAL) and "\n[fresh-dl-Q4_K_M]\n" in text[len(ORIGINAL):]
    assert ini.get_section("fresh-dl-Q4_K_M")["ctx-size"] == "4096"


def test_startup_migration_keeps_comments_and_untouched_sections():
    original = (
        "version = 1\n\n"
        "; operator header comment\n"
        "[chat-a]\n# chat A runs the 8k preset\nmodel = /models/a.gguf\nctx-size = 8192\n\n"
        "[chat-b]\nmodel = /models/b.gguf\n; folded below\nLLAMA_ARG_CACHE_RAM = 8192\n\n"
        "# section B notes survive\n"
        "[chat-ok]\nmodel = /models/c.gguf\n; already bounded\ncache-ram = 1024\n\n"
        "[nomic-embed]\nmodel = /models/n.gguf\nembedding = true\n")
    PATH.write_text(original)
    assert ini.migrate_cache_ram() == ["chat-a", "chat-b", "nomic-embed"]
    assert PATH.read_text() == (
        "version = 1\n\n"
        "; operator header comment\n"
        "[chat-a]\n# chat A runs the 8k preset\nmodel = /models/a.gguf\nctx-size = 8192\ncache-ram = 1024\n\n"
        "[chat-b]\nmodel = /models/b.gguf\n; folded below\ncache-ram = 2048\n\n"
        "# section B notes survive\n"
        "[chat-ok]\nmodel = /models/c.gguf\n; already bounded\ncache-ram = 1024\n\n"
        "[nomic-embed]\nmodel = /models/n.gguf\nembedding = true\ncache-ram = 0\n")


def test_file_without_a_final_newline_gets_one_before_an_added_key():
    PATH.write_text("version = 1\n\n[a]\n# note\nmodel = /models/a.gguf")
    ini.upsert_section("a", {"model": "/models/a.gguf", "ctx-size": "2048"}, "")
    assert PATH.read_text() == "version = 1\n\n[a]\n# note\nmodel = /models/a.gguf\nctx-size = 2048\ncache-ram = 1024\n"


def test_multi_line_value_in_another_section_is_left_alone():
    PATH.write_text("[a]\nmodel = /models/a.gguf\n\n[b]\nmodel = /models/b.gguf\nnote = first\n  second\n")
    ini.upsert_section("a", {"model": "/models/a.gguf", "cache-ram": "64"}, "")
    assert PATH.read_text() == "[a]\nmodel = /models/a.gguf\ncache-ram = 64\n\n[b]\nmodel = /models/b.gguf\nnote = first\n  second\n"


def test_missing_file_is_created(tmp_path, monkeypatch):
    target = tmp_path / "models.ini"
    monkeypatch.setattr(ini.settings, "models_ini_path", target)
    ini.upsert_section("solo", {"model": "/models/solo.gguf", "cache-ram": "64"}, "")
    assert target.read_text() == "[solo]\nmodel = /models/solo.gguf\ncache-ram = 64\n"


def test_render_preserving_refuses_rather_than_guess(monkeypatch):
    # If the in-place edit would not reproduce the parser's state, the full rewrite is used.
    monkeypatch.setattr(ini, "_raw_layout", lambda text: [])
    ini.upsert_section("tiny", {"model": "/models/tiny/tiny-Q4_K_M.gguf", "ctx-size": "1024"}, "")
    assert ini.get_section("tiny")["ctx-size"] == "1024" and ini.get_section("b")["temp"] == "0.7"
