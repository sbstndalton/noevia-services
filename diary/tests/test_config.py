import copy

from agent.config import Config, _apply_env, _resolve_aux_model


def test_neutral_corpus_environment(monkeypatch):
    monkeypatch.setenv("CORPUS_BACKEND", "webdav")
    monkeypatch.setenv("CORPUS_ROOT", "Notes/Journal")
    monkeypatch.setenv("CORPUS_LOCAL_ROOT", "/data/corpus")
    cfg = Config({})
    _apply_env(cfg)
    assert cfg.get("corpus.backend") == "webdav"
    assert cfg.get("corpus.root") == "Notes/Journal"
    assert cfg.get("corpus.local.root") == "/data/corpus"


def test_legacy_remote_root_alias(monkeypatch, caplog):
    monkeypatch.setenv("CORPUS_REMOTE_ROOT", "Legacy/Diary")
    cfg = Config({})
    _apply_env(cfg)
    assert cfg.get("corpus.root") == "Legacy/Diary"
    assert "deprecated" in caplog.text


def test_config_deepcopy_does_not_share_nested_state():
    cfg = Config({"corpus": {"backend": "local"}})

    cloned = copy.deepcopy(cfg)
    cloned.as_dict()["corpus"]["backend"] = "webdav"

    assert cfg.corpus.backend == "local"
    assert cloned.corpus.backend == "webdav"


def test_daily_layout_environment(monkeypatch):
    monkeypatch.setenv("DIARY_ENTRY_LAYOUT", "daily")
    monkeypatch.setenv("DIARY_ENTRIES_PREFIX", "Entries")
    cfg = Config({})
    _apply_env(cfg)
    assert cfg.get("corpus.entry_layout") == "daily"
    assert cfg.get("corpus.entries_prefix") == "Entries"


# The live deployment ran for days with LLM_AUX_MODEL="default" — the placeholder
# every compose file falls back to. Lemonade does not serve a model by that name,
# so every summariser call 404d and entries silently lost their topic headers.


def test_placeholder_aux_model_falls_back_to_the_chat_model(caplog):
    cfg = Config({"llm": {"chat_model": "Gemma-4-E4B-it-GGUF", "aux": {"model": "default"}}})
    _resolve_aux_model(cfg)
    assert cfg.get("llm.aux.model") == "Gemma-4-E4B-it-GGUF"
    assert "DIARY_AUX_MODEL" in caplog.text


def test_missing_aux_model_falls_back_to_the_chat_model():
    cfg = Config({"llm": {"chat_model": "Gemma-4-E4B-it-GGUF", "aux": {}}})
    _resolve_aux_model(cfg)
    assert cfg.get("llm.aux.model") == "Gemma-4-E4B-it-GGUF"


def test_explicit_aux_model_is_never_overridden():
    cfg = Config({"llm": {"chat_model": "big-model", "aux": {"model": "small-model"}}})
    _resolve_aux_model(cfg)
    assert cfg.get("llm.aux.model") == "small-model"


def test_both_unset_is_left_alone_rather_than_invented():
    # Nothing configured at all: a 404 naming "default" is a legible symptom,
    # and guessing a model name here would only hide it differently.
    cfg = Config({"llm": {"chat_model": "default", "aux": {"model": "default"}}})
    _resolve_aux_model(cfg)
    assert cfg.get("llm.aux.model") == "default"


def test_explicit_fallback_model_environment_keys_2099(monkeypatch):
    monkeypatch.setenv("LLM_CHAT_FALLBACK_MODEL", "synthetic-chat-fallback")
    monkeypatch.setenv("LLM_AUX_FALLBACK_MODEL", "synthetic-aux-fallback")
    cfg = Config({})
    _apply_env(cfg)
    assert cfg.get("llm.chat_fallback_model") == "synthetic-chat-fallback"
    assert cfg.get("llm.aux.fallback_model") == "synthetic-aux-fallback"


def test_app_state_gives_each_role_its_own_setting_keys_and_fallback_2099(monkeypatch, tmp_path):
    import agent.app as appmod

    cfg = appmod.load_config()
    cfg.as_dict().setdefault("llm", {}).update({"chat_model": "synthetic-chat", "chat_fallback_model": "synthetic-chat-b"})
    cfg.as_dict()["llm"].setdefault("aux", {}).update({"model": "synthetic-aux", "fallback_model": "synthetic-aux-b"})
    cfg.as_dict().setdefault("retrieval", {})["db_path"] = str(tmp_path / "r.db")
    cfg.as_dict().setdefault("corpus", {}).update({"backend": "local", "local": {"root": str(tmp_path / "c")}})
    state = appmod.AppState(cfg)
    try:
        assert (state.llm_main.setting_key, state.llm_main.fallback_model) == ("LLM_CHAT_MODEL", "synthetic-chat-b")
        assert (state.llm_aux.setting_key, state.llm_aux.fallback_model) == ("LLM_AUX_MODEL", "synthetic-aux-b")
    finally:
        state.llm_main.close()
        state.llm_aux.close()
