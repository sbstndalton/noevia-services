"""#1179: a system routing (Laya) section is marked by an explicit `role = laya` key. The name/path
rule is a deprecated fallback for sections with no role key, and it is logged. Synthetic sections only."""
import configparser
import logging

from conftest import ROOT  # noqa: F401  (sets up the synthetic model folders before app imports)
from app import ini


def _cp(text: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None)
    cp.read_string(text)
    return cp


def test_an_explicit_role_decides_whatever_the_name_or_path_says():
    cp = _cp(
        "[routing-box]\nmodel = /models/routing-box/synthetic.gguf\nrole = laya\n"
        "[laya-named-chat]\nmodel = /models/laya-named-chat/chat.gguf\nrole = chat\n"
    )
    assert ini._is_laya_section(cp, "routing-box") is True, "the role key marks Laya whatever its name"
    assert ini._is_chat_section(cp, "routing-box") is False
    assert ini._is_laya_section(cp, "laya-named-chat") is False, "an explicit non-Laya role beats the name"
    assert ini._is_chat_section(cp, "laya-named-chat") is True


def test_the_role_key_is_case_and_space_tolerant():
    cp = _cp("[box]\nmodel = /models/box/synthetic.gguf\nrole =   LAYA  \n")
    assert ini._is_laya_section(cp, "box") is True


def test_a_section_without_a_role_keeps_the_name_rule_and_logs_it_once(caplog):
    ini._name_rule_logged.discard("laya_multilingual_f16")
    cp = _cp("[laya_multilingual_f16]\nmodel = /models/laya_multilingual_f16/synthetic.gguf\n")
    with caplog.at_level(logging.WARNING, logger="app.ini"):
        assert ini._is_laya_section(cp, "laya_multilingual_f16") is True
        assert ini._is_laya_section(cp, "laya_multilingual_f16") is True
    warnings = [r for r in caplog.records if "deprecated name/path rule" in r.getMessage()]
    assert len(warnings) == 1, "one log line per section, not one per call"
    assert "role = laya" in warnings[0].getMessage()


def test_the_path_rule_still_applies_without_a_role_key():
    cp = _cp("[routing]\nmodel = /models/laya-small/synthetic.gguf\n")
    assert ini._is_laya_section(cp, "routing") is True


def test_an_ordinary_section_is_not_laya_and_is_not_logged(caplog):
    cp = _cp("[tiny]\nmodel = /models/tiny/synthetic.gguf\n")
    with caplog.at_level(logging.WARNING, logger="app.ini"):
        assert ini._is_laya_section(cp, "tiny") is False
        assert ini._is_chat_section(cp, "tiny") is True
    assert not [r for r in caplog.records if "deprecated" in r.getMessage()]
    assert ini._is_laya_section(cp, "*") is False
    assert ini._is_laya_section(cp, "missing-section") is False


def test_the_engine_command_never_receives_the_role_key():
    line = ini.to_cli("box", [("model", "/models/box/synthetic.gguf"), ("role", "laya"), ("ctx-size", "4096")])
    assert "--role" not in line and "laya" not in line
    assert "--ctx-size 4096" in line
