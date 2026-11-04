"""P0 packaging (noevia#1071): the leaf smoke, the image's /out/ bundle and the adapters name the same binaries."""
from pathlib import Path
import re
import sys

import pytest

from tests.conftest import RUST_LEAF_MODULES

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import smoke_diary_leaves as smoke  # noqa: E402

SELF_TESTED = {'tenant-assertion', 'workspace-restore-verify'}


def dockerfile():
    return (ROOT / 'Dockerfile').read_text()


def test_each_leaf_is_its_adapters_fixed_installed_path():
    for name, (module, attribute, _) in smoke.LEAVES.items():
        assert getattr(module, attribute) == f'/usr/local/bin/{name}', name


def test_the_image_builds_and_bundles_every_leaf_and_smokes_them_all():
    text = dockerfile()
    copied = re.search(r'&& cp ([a-z \\\n-]+) /out/', text)
    assert copied, 'the Rust stage copies its binaries into /out/'
    assert set(copied.group(1).replace('\\', ' ').split()) == set(smoke.LEAVES) | SELF_TESTED
    final = text.split('\nFROM ')[-1]
    assert 'COPY --from=tenant-assertion /out/ /usr/local/bin/' in final
    assert 'COPY --from=tenant-assertion /src/' not in final
    assert 'RUN python scripts/smoke_diary_leaves.py --all' in final
    # The smoke runs after scripts/ is in the image, and no selector is set for the service.
    assert final.index('COPY scripts/ ./scripts/') < final.index('smoke_diary_leaves.py')
    assert not re.search(r'(?m)^ENV .*_IMPL', text)


def test_every_named_package_is_built():
    build = dockerfile().split('cargo build', 1)[1].split('&& ./target', 1)[0]
    packages = set(re.findall(r'-p ([a-z-]+)', build))
    assert packages == {'tenant-assertion-cli', 'workspace-restore-verify-cli', 'diary-corpus-write-cli',
                        'diary-corpus-parse', 'diary-score-gap', 'diary-temporal-focus',
                        'diary-workspace-export', 'diary-workspace-import-apply'}


@pytest.mark.parametrize('name', sorted(smoke.LEAVES))
def test_a_missing_or_failing_leaf_fails_the_smoke(name, tmp_path, monkeypatch, capsys):
    module, attribute, _ = smoke.LEAVES[name]
    monkeypatch.setattr(module, attribute, getattr(module, attribute))  # restored after main() overrides it
    assert smoke.main([name, '--bin-dir', str(tmp_path)]) == 1
    assert f'{name}: FAILED' in capsys.readouterr().err
    broken = tmp_path / name
    broken.write_text('#!/bin/sh\ncat >/dev/null\nexit 3\n')
    broken.chmod(0o755)
    assert smoke.main([name, '--bin-dir', str(tmp_path)]) == 1


def test_unknown_or_no_leaf_is_a_usage_error():
    for argv in ([], ['diary-unknown']):
        with pytest.raises(SystemExit) as exited:
            smoke.main(argv)
        assert exited.value.code == 2


def test_required_mode_covers_existing_leaf_modules():
    for module in RUST_LEAF_MODULES:
        assert (ROOT / 'tests' / f'{module}.py').is_file(), module
