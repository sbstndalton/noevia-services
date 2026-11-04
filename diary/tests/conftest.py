import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


import pytest


@pytest.fixture(autouse=True)
def _fresh_storage_gates():
    """Back-off gates are shared per credential fingerprint (#1166); tests reuse fake credentials."""
    from agent import storage_backoff

    with storage_backoff._gates_lock:
        storage_backoff._gates.clear()
    yield


# P0 (noevia#1071): CI builds every Diary Rust leaf at the Dockerfile's pinned noevia-rs ref and sets
# DIARY_RUST_LEAVES_REQUIRED=1. The real-binary differential cases in these modules skip when their binary
# variable is unset; in required mode a skip there is a failure, so a dropped variable cannot pass silently.
RUST_LEAF_MODULES = frozenset({
    'test_corpus_write_native', 'test_corpus_parse_native', 'test_score_gap_native', 'test_temporal_native',
    'test_workspace_export_native', 'test_workspace_import_native', 'test_workspace_restore_differential',
    'test_workspace_restore_switch',
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
