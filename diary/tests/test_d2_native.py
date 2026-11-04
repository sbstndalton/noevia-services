"""Real binary parity and fail-closed transport checks for the pure D2 leaves."""
import importlib.util
import os
from pathlib import Path
import pytest
from agent import leaf_native

MODULE = Path(__file__).resolve().parents[1] / 'scripts/smoke_diary_leaves.py'
spec = importlib.util.spec_from_file_location('d2_smoke', MODULE)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
NAMES = ['diary-entity-focus','diary-index-transform','diary-storage-backoff',
         'diary-workspace-rules','diary-backup-format','diary-backup-run']

@pytest.mark.parametrize('name', NAMES)
def test_real_binary_parity(name, tmp_path, monkeypatch):
    folder = os.environ.get('DIARY_D2_TEST_BIN_DIR')
    if not folder:
        pytest.skip('DIARY_D2_TEST_BIN_DIR is not set')
    module, attr, run = smoke.LEAVES[name]
    binary = str(Path(folder) / name)
    assert os.access(binary, os.X_OK), 'required executable missing'
    monkeypatch.setattr(module, attr, binary)
    run(tmp_path)

@pytest.mark.parametrize('name', NAMES[:-1])
def test_missing_binary_never_falls_back(name, monkeypatch):
    module, attr, _ = smoke.LEAVES[name]
    monkeypatch.setattr(module, attr, '/missing/diary-leaf')
    monkeypatch.setenv(module.SELECTOR, 'rust')
    with pytest.raises(leaf_native.LeafUnavailable):
        module.invoke(op='missing')

@pytest.mark.parametrize('body', ['import time; time.sleep(5)', 'print("bad json")', 'raise SystemExit(1)'])
def test_bad_children_reaped(body, tmp_path):
    import sys
    executable = tmp_path / 'bad-leaf'
    executable.write_text('#!' + sys.executable + '\n' + body + '\n')
    executable.chmod(0o755)
    with pytest.raises(leaf_native.LeafUnavailable):
        leaf_native.call(str(executable), {}, timeout=.1)
