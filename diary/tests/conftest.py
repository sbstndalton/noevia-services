import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import pytest


@pytest.fixture(autouse=True)
def _fresh_storage_gates(tmp_path):
    """Back-off gates are shared per credential fingerprint (#1166); tests reuse fake credentials."""
    from agent import storage_backoff

    storage_backoff.configure_shared(tmp_path / 'storage-gates.db')
    yield
    storage_backoff.configure_shared(None)


# P0 (noevia#1071): CI builds every Diary Rust leaf at the Dockerfile's pinned noevia-rs ref and sets
# DIARY_RUST_LEAVES_REQUIRED=1. The real-binary differential cases in these modules skip when their binary
# variable is unset; in required mode a skip there is a failure, so a dropped variable cannot pass silently.
RUST_LEAF_MODULES = frozenset({
    'test_corpus_write_native', 'test_corpus_parse_native', 'test_score_gap_native', 'test_temporal_native',
    'test_workspace_export_native', 'test_workspace_import_native', 'test_workspace_restore_differential',
    'test_workspace_restore_switch', 'test_d2_native', 'test_d2_regressions',
})


def rust_leaves_required():
    return os.environ.get('DIARY_RUST_LEAVES_REQUIRED') == '1'


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped and rust_leaves_required() and item.module.__name__.rsplit('.', 1)[-1] in RUST_LEAF_MODULES:
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = 'failed'
        report.longrepr = f'DIARY_RUST_LEAVES_REQUIRED=1: a real-binary leaf case skipped ({reason})'


@pytest.fixture(autouse=True)
def _d2_test_binaries(monkeypatch, request):
    if request.node.name == 'test_each_leaf_is_its_adapters_fixed_installed_path':
        return
    folder = os.environ.get('DIARY_D2_TEST_BIN_DIR')
    if folder:
        from agent import entity_focus_native, index_transform_native, storage_backoff_native, workspace_rules_native, backup_format_native, backup_native
        for module in (entity_focus_native, index_transform_native, storage_backoff_native, workspace_rules_native, backup_format_native, backup_native):
            monkeypatch.setattr(module, 'DEFAULT_BIN', str(Path(folder) / Path(module.DEFAULT_BIN).name))


@pytest.fixture(autouse=True, scope="module")
def _native_entity_fixture_vec_loading(request):
    # sqlite-vec 0.1.9 no longer enables extension loading itself. Authorize only
    # each disposable test connection, then revoke it, so native entity CI really
    # exercises the existing synthetic retrieval fixture instead of skipping it.
    if (os.environ.get('DIARY_D2_TEST_BIN_DIR') and
            request.module.__name__.rsplit('.', 1)[-1] == 'test_entity_focus'):
        import sqlite_vec
        load = sqlite_vec.load
        def load_fixture_connection(conn):
            conn.enable_load_extension(True)
            try:
                return load(conn)
            finally:
                conn.enable_load_extension(False)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sqlite_vec, 'load', load_fixture_connection)
            yield
    else:
        yield
