"""Only exact native confirmation removes the Python schema initializer. No live services."""
import pytest
from app import db, hw, bench
from app import main
from app.config import settings

ENV = "NOEVIA_NATIVE_MODEL_MANAGER_DB_INIT_CONFIRMED"

@pytest.mark.parametrize("value", [None, "", "0", "true", "rust", " 1", "1 ", "1"])
def test_startup_schema_owner_is_exact_and_other_startup_work_is_unchanged(monkeypatch, value):
    calls = []
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    monkeypatch.delenv("NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED", raising=False)
    monkeypatch.delenv("NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_RUNNER_CONFIRMED", raising=False)
    monkeypatch.delenv("NOEVIA_NATIVE_MODEL_MANAGER_SECTION_WRITES_CONFIRMED", raising=False)
    monkeypatch.setattr(main.model_files, "require_binary", lambda: calls.append("binary"))
    monkeypatch.setattr(db, "init", lambda: calls.append("schema"))
    monkeypatch.setattr(db, "seed_bench_prompts", lambda: calls.append("seed"))
    monkeypatch.setattr(hw, "start_sampler", lambda: calls.append("sampler"))
    monkeypatch.setattr(bench, "reap_orphans", lambda: calls.append("orphans"))
    monkeypatch.setattr(settings, "migrate_cache_ram_on_start", True)
    monkeypatch.setattr(main, "_migrate_cache_ram", lambda: calls.append("ini"))
    main._startup()
    assert calls == (["binary"] + ([] if value == "1" else ["schema"])
                     + ["seed", "sampler", "orphans", "ini"])

def test_native_seed_and_orphan_guards_remain_independent(monkeypatch):
    calls = []
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_PROMPTS_CONFIRMED", "1")
    monkeypatch.setenv("NOEVIA_NATIVE_MODEL_MANAGER_BENCHMARK_RUNNER_CONFIRMED", "1")
    monkeypatch.setattr(main.model_files, "require_binary", lambda: calls.append("binary"))
    monkeypatch.setattr(db, "init", lambda: pytest.fail("second schema initializer"))
    monkeypatch.setattr(db, "seed_bench_prompts", lambda: pytest.fail("second seed"))
    monkeypatch.setattr(hw, "start_sampler", lambda: calls.append("sampler"))
    monkeypatch.setattr(bench, "reap_orphans", lambda: pytest.fail("second orphan cleanup"))
    monkeypatch.setattr(settings, "migrate_cache_ram_on_start", False)
    main._startup()
    assert calls == ["binary", "sampler"]
