"""#1179 / #1596: a system routing (Laya) section is marked by noevia's own models.roles.json beside
models.ini, never by a key in models.ini (llama-server reads that file and its preset loader throws
on an option it does not know). The name/path rule is a deprecated fallback and it is logged.
Synthetic sections only."""
import configparser
import json
import logging

import pytest

from conftest import ROOT  # noqa: F401  (sets up the synthetic model folders before app imports)
from fastapi.testclient import TestClient

from app import ini
from app.config import settings
from app.main import app


@pytest.fixture(autouse=True)
def roles_file(tmp_path, monkeypatch):
    """A private models.ini / models.roles.json pair per test."""
    monkeypatch.setattr(settings, "models_ini_path", tmp_path / "models.ini")
    ini._roles_bad_logged.clear()
    return tmp_path / "models.roles.json"


def _cp(text: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None)
    cp.read_string(text)
    return cp


def test_the_roles_file_sits_beside_models_ini():
    assert ini.roles_path() == settings.models_ini_path.with_name("models.roles.json")


def test_an_explicit_role_decides_whatever_the_name_or_path_says(roles_file):
    roles_file.write_text(json.dumps({"routing-box": "laya", "laya-named-chat": "chat"}))
    cp = _cp(
        "[routing-box]\nmodel = /models/routing-box/synthetic.gguf\n"
        "[laya-named-chat]\nmodel = /models/laya-named-chat/chat.gguf\n"
    )
    assert ini._is_laya_section(cp, "routing-box") is True, "the role marks Laya whatever its name"
    assert ini._is_chat_section(cp, "routing-box") is False
    assert ini._is_laya_section(cp, "laya-named-chat") is False, "an explicit non-Laya role beats the name"
    assert ini._is_chat_section(cp, "laya-named-chat") is True


def test_the_role_is_case_and_space_tolerant(roles_file):
    roles_file.write_text(json.dumps({"box": "  LAYA "}))
    assert ini._is_laya_section(_cp("[box]\nmodel = /models/box/synthetic.gguf\n"), "box") is True


def test_a_role_key_inside_models_ini_is_not_a_marker():
    cp = _cp("[box]\nmodel = /models/box/synthetic.gguf\nrole = laya\n")
    assert ini._is_laya_section(cp, "box") is False, "models.ini is llama.cpp's file; the key is not read"


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"box": 3, "x": ""}'])
def test_an_unusable_roles_file_is_ignored_not_fatal(roles_file, content, caplog):
    roles_file.write_text(content)
    with caplog.at_level(logging.WARNING, logger="app.ini"):
        assert ini.read_roles() == {}
        assert ini._is_laya_section(_cp("[box]\nmodel = /models/box/synthetic.gguf\n"), "box") is False
    if content != '{"box": 3, "x": ""}':
        assert any("models.roles.json is ignored" in r.getMessage() for r in caplog.records)


def test_a_section_without_a_role_keeps_the_name_rule_and_logs_the_new_marker_once(caplog):
    ini._name_rule_logged.discard("laya_multilingual_f16")
    cp = _cp("[laya_multilingual_f16]\nmodel = /models/laya_multilingual_f16/synthetic.gguf\n")
    with caplog.at_level(logging.WARNING, logger="app.ini"):
        assert ini._is_laya_section(cp, "laya_multilingual_f16") is True
        assert ini._is_laya_section(cp, "laya_multilingual_f16") is True
    warnings = [r for r in caplog.records if "deprecated name/path rule" in r.getMessage()]
    assert len(warnings) == 1, "one log line per section, not one per call"
    message = warnings[0].getMessage()
    assert "models.roles.json" in message and '"laya_multilingual_f16": "laya"' in message
    assert "role = laya" not in message, "the log must not tell operators to put a key llama.cpp rejects into models.ini"


def test_the_path_rule_still_applies_without_a_role():
    assert ini._is_laya_section(_cp("[routing]\nmodel = /models/laya-small/synthetic.gguf\n"), "routing") is True


def test_an_ordinary_section_is_not_laya_and_is_not_logged(caplog):
    cp = _cp("[tiny]\nmodel = /models/tiny/synthetic.gguf\n")
    with caplog.at_level(logging.WARNING, logger="app.ini"):
        assert ini._is_laya_section(cp, "tiny") is False
        assert ini._is_chat_section(cp, "tiny") is True
    assert not [r for r in caplog.records if "deprecated" in r.getMessage()]
    assert ini._is_laya_section(cp, "*") is False
    assert ini._is_laya_section(cp, "missing-section") is False


def test_renaming_and_deleting_a_section_moves_and_drops_its_role(roles_file):
    settings.models_ini_path.write_text("[box]\nmodel = /models/box/synthetic.gguf\n[keep]\nmodel = /models/keep/synthetic.gguf\n")
    roles_file.write_text(json.dumps({"box": "laya", "keep": "chat"}))
    assert ini.rename_section("box", "renamed") is True
    assert json.loads(roles_file.read_text()) == {"keep": "chat", "renamed": "laya"}
    assert ini.delete_section("renamed") is True
    assert json.loads(roles_file.read_text()) == {"keep": "chat"}
    assert ini.delete_section("keep") is True
    assert not roles_file.exists(), "an empty roles file is removed rather than left behind"


def test_models_ini_never_gains_a_key_llama_cpp_does_not_accept(roles_file):
    """The preset loader of llama-server throws on an unknown key, so every write path keeps models.ini to
    the keys noevia's own field schema (the llama.cpp options) knows, plus `model`. A role marker is
    beside it, not in it."""
    settings.models_ini_path.write_text(
        "[*]\ncache-ram = 1024\n\n[routing-box]\nmodel = /models/routing-box/synthetic.gguf\n\n"
        "[chat]\nmodel = /models/tiny/tiny-Q4_K_M.gguf\nctx-size = 4096\n")
    roles_file.write_text(json.dumps({"routing-box": "laya"}))
    ini.upsert_section("chat", {"ctx-size": "8192", "threads": "4"}, "")
    ini.rename_section("chat", "chat2")
    ini.migrate_cache_ram()
    assert ini._is_laya_section(ini.read_ini(), "routing-box") is True
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(settings.models_ini_path.read_text())
    accepted = ini.ALL_KNOWN_KEYS | {"model"}
    unknown = {(s, k) for s in parsed.sections() for k in parsed[s] if k not in accepted}
    assert not unknown, f"keys outside the llama.cpp option set in models.ini: {sorted(unknown)}"
    assert "role" not in settings.models_ini_path.read_text()
    assert "role" not in ini.ALL_KNOWN_KEYS


# noevia#2093 phase 1: the services reader accepts the system-routing role key as well as the legacy
# role key (noevia-core#227 and noevia-rs#271 read it the same way). The legacy key is accepted until
# phase 3 of noevia#2093. Synthetic sections only.
LEGACY_ROLE = ini.LAYA_ROLE
NEW_ROLE = "system-routing"


@pytest.fixture
def api_client():
    with TestClient(app) as c:
        yield c


def _api_code(client, name: str):
    return client.get(f"/api/v1/sections/{name}/autoconfig").json().get("code")


def test_the_system_routing_role_is_read_after_trim_and_case(roles_file):
    roles_file.write_text(json.dumps({"system-box": " System-Routing "}))
    settings.models_ini_path.write_text("[system-box]\nmodel = /models/system-box/synthetic.gguf\n")
    assert ini.is_laya_section("system-box") is True
    assert ini._is_laya_section(ini.read_ini(), "system-box") is True
    assert ini._is_chat_section(ini.read_ini(), "system-box") is False


def test_the_system_routing_role_returns_the_laya_section_code_from_the_api(roles_file, api_client):
    roles_file.write_text(json.dumps({"system-box": " System-Routing "}))
    settings.models_ini_path.write_text("[system-box]\nmodel = /models/system-box/synthetic.gguf\n")
    r = api_client.get("/api/v1/sections/system-box/autoconfig").json()
    assert r["code"] == "laya_section" and "No model file" not in r["error"]


def test_the_legacy_role_key_still_marks_a_system_section(roles_file, api_client):
    roles_file.write_text(json.dumps({"legacy-box": LEGACY_ROLE}))
    settings.models_ini_path.write_text("[legacy-box]\nmodel = /models/legacy-box/synthetic.gguf\n")
    assert ini.is_laya_section("legacy-box") is True
    assert _api_code(api_client, "legacy-box") == "laya_section"


@pytest.mark.parametrize("role", ["system-routing-x", "system_routing", "systemrouting"])
def test_roles_that_only_resemble_the_system_routing_key_do_not_count(roles_file, api_client, role):
    roles_file.write_text(json.dumps({"near-box": role}))
    settings.models_ini_path.write_text("[near-box]\nmodel = /models/near-box/synthetic.gguf\n")
    assert ini.is_laya_section("near-box") is False
    assert ini._is_chat_section(ini.read_ini(), "near-box") is True
    assert _api_code(api_client, "near-box") != "laya_section"


def test_a_legacy_prefixed_name_with_a_chat_role_is_not_a_system_section(roles_file, api_client):
    name = f"{LEGACY_ROLE}-prefixed-chat"
    roles_file.write_text(json.dumps({name: "chat"}))
    settings.models_ini_path.write_text(f"[{name}]\nmodel = /models/{name}/synthetic.gguf\n")
    assert ini.is_laya_section(name) is False, "an explicit role beats the name-prefix fallback"
    assert ini._is_chat_section(ini.read_ini(), name) is True
    assert _api_code(api_client, name) != "laya_section"


def test_renaming_and_deleting_a_system_routing_section_moves_and_drops_its_exact_role(roles_file):
    settings.models_ini_path.write_text("[box]\nmodel = /models/box/synthetic.gguf\n[keep]\nmodel = /models/keep/synthetic.gguf\n")
    roles_file.write_text(json.dumps({"box": NEW_ROLE, "keep": "chat"}, indent=2, sort_keys=True) + "\n")
    assert ini.rename_section("box", "renamed") is True
    assert roles_file.read_bytes() == b'{\n  "keep": "chat",\n  "renamed": "system-routing"\n}\n'
    assert ini.is_laya_section("renamed") is True
    assert ini.delete_section("renamed") is True
    assert roles_file.read_bytes() == b'{\n  "keep": "chat"\n}\n'
    assert ini.delete_section("keep") is True
    assert not roles_file.exists()
