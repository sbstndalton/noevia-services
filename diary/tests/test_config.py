import copy

from agent.config import Config, _apply_env


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
