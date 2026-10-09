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
