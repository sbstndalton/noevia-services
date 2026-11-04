"""P0 packaging (noevia#1071): the leaf smoke, the image's /out/ bundle and the adapters name the same binaries."""
from pathlib import Path
import os
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
    # The build tree and cargo registry do not stay in the builder layer (review P3-2).
    rust_stage = text.split('\nFROM python:')[0]
    assert rust_stage.rstrip().endswith('&& rm -rf /src/target /usr/local/cargo/registry')
    assert rust_stage.index('/out/ \\') < rust_stage.index('rm -rf /src/target')
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
                        'diary-workspace-export', 'diary-workspace-import-apply',
                        'diary-entity-focus', 'diary-index-transform', 'diary-storage-backoff',
                        'diary-workspace-rules', 'diary-backup-format', 'diary-backup-run', 'noevia-diary'}


def native_stage_env():
    """{NAME: value} of the image's own ENV instructions in the final stage."""
    final = dockerfile().split('\nFROM ')[-1].replace('\\\n', ' ')
    env = {}
    for line in final.splitlines():
        if line.startswith('ENV '):
            env.update(re.findall(r'([A-Z_]+)=(\S*)', line))
    return env


def test_the_image_sets_the_three_native_bin_defaults_to_the_paths_the_code_already_uses():
    from agent import rust_forward, tenant_assertion, workspace_export_native
    env = native_stage_env()
    assert env['DIARY_RUST_SERVER_BIN'] == rust_forward.DEFAULT_BIN == '/usr/local/bin/noevia-diary'
    assert env['TENANT_ASSERTION_BIN'] == tenant_assertion.DEFAULT_BIN == '/usr/local/bin/tenant-assertion'
    assert env['DIARY_WORKSPACE_EXPORT_BIN'] == workspace_export_native.BINARY == '/usr/local/bin/diary-workspace-export'
    # Each one is a bin the same /out/ bundle installs, and the build checks they are executable.
    for key in ('DIARY_RUST_SERVER_BIN', 'DIARY_WORKSPACE_EXPORT_BIN', 'TENANT_ASSERTION_BIN'):
        assert Path(env[key]).name in set(smoke.LEAVES) | {'tenant-assertion'}, key
    assert 'RUN test -x "$DIARY_RUST_SERVER_BIN" && test -x "$DIARY_WORKSPACE_EXPORT_BIN" && test -x "$TENANT_ASSERTION_BIN"' in dockerfile()


def test_the_image_leaves_every_native_switch_off():
    # An empty route and writer list means the Rust server is never started and no selector is rust: the
    # image behaves as the Python-only one until the owner sets them.
    text = dockerfile()
    assert not re.search(r'(?m)^\s*(ENV|ARG)\b[^\n]*\b(DIARY_RUST_ROUTES|DIARY_RUST_WRITES)\b', text)
    assert not re.search(r'(?m)^ENV .*_IMPL', text)
    ours = {key for key in native_stage_env() if key.startswith(('DIARY_', 'TENANT_', 'WORKSPACE_'))}
    assert ours == {'DIARY_RUST_SERVER_BIN', 'DIARY_WORKSPACE_EXPORT_BIN', 'TENANT_ASSERTION_BIN'}
    from agent import rust_forward
    assert rust_forward.configured_routes('') == {}
    if hasattr(rust_forward, 'configured_writers'):  # D5 (DIARY_RUST_WRITES); absent on the D4b base
        assert rust_forward.configured_writers('') == frozenset()


def test_the_smoke_names_noevia_diary_with_its_fixed_path_and_the_start_up_probe():
    module, attribute, run = smoke.LEAVES['noevia-diary']
    assert (module.__name__, attribute) == ('agent.rust_forward', 'DEFAULT_BIN')
    assert run is smoke.noevia_diary_leaf


@pytest.mark.skipif(not os.environ.get('DIARY_RUST_SERVER_BIN') and os.environ.get('DIARY_RUST_SERVER_REQUIRED') != '1',
                    reason='noevia-diary not built (set DIARY_RUST_SERVER_BIN)')
def test_the_real_noevia_diary_binary_passes_the_start_up_smoke(tmp_path, monkeypatch, capsys):
    # In required mode (CI) an unset DIARY_RUST_SERVER_BIN is a failure (KeyError), not a skip.
    (tmp_path / 'noevia-diary').symlink_to(Path(os.environ['DIARY_RUST_SERVER_BIN']).resolve())
    monkeypatch.setattr(smoke.rust_forward, 'DEFAULT_BIN', smoke.rust_forward.DEFAULT_BIN)
    assert smoke.main(['noevia-diary', '--bin-dir', str(tmp_path)]) == 0
    assert 'noevia-diary: ok' in capsys.readouterr().out


@pytest.mark.parametrize('name', sorted(smoke.LEAVES))
def test_a_missing_or_failing_leaf_fails_the_smoke(name, tmp_path, monkeypatch, capsys):
    module, attribute, _ = smoke.LEAVES[name]
    monkeypatch.setattr(module, attribute, getattr(module, attribute))  # restored after main() overrides it
    assert smoke.main([name, '--bin-dir', str(tmp_path)]) == 1
    assert f'{name}: FAILED' in capsys.readouterr().err
    broken = tmp_path / name
    broken.write_text('#!/bin/sh\nexit 3\n')
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
