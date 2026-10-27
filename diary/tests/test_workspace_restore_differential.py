"""The shared differential fixtures (tests/fixtures/workspace-restore.v1.json, generated in
sbstndalton/noevia-rs by tools/gen-workspace-restore.py from this module) against the live Python
reference, and, when WORKSPACE_RESTORE_VERIFY_BIN names a built binary, against the Rust leaf
through the real switch (noevia#982). CI copies nothing: it `cmp`s this file with noevia-rs' copy
at the pinned ref. Every archive in the table is synthetic."""
import base64
import json
import os
from pathlib import Path

import pytest

from agent import workspace_restore as wr

FIXTURES = json.loads((Path(__file__).parent / 'fixtures' / 'workspace-restore.v1.json').read_text())['cases']
CODES = {message: code for code, message in wr.MESSAGES.items()}


def decision(body):
    try:
        files, dirs, _ = wr.verify(body)
    except ValueError as exc:
        return 'reject: ' + CODES.get(str(exc), 'UNMAPPED ' + str(exc))
    return 'accept ' + wr.listing_sha256(files, dirs)


def test_fixture_table_is_present_and_safe():
    assert len(FIXTURES) > 250
    for c in FIXTURES:
        # The Rust side never accepts what Python refuses, nor accepts with another listing.
        if c['rust'].startswith('accept'):
            assert c['python'] == c['rust'], c['name']
        if c['python'] != c['rust']:
            assert c.get('stricter') or (c['name'].startswith('mutant/') and c['rust'] == 'reject: invalid'), c['name']
    covered = {c['python'] for c in FIXTURES if c['python'] == c['rust']}
    for code in wr.MESSAGES:
        if code != 'archive_too_large':  # 272 MiB does not fit a fixture; test_workspace_restore covers it
            assert 'reject: ' + code in covered, code


@pytest.mark.parametrize('case', FIXTURES, ids=[c['name'] for c in FIXTURES])
def test_python_reference_still_decides_as_recorded(case, monkeypatch):
    monkeypatch.delenv('WORKSPACE_RESTORE_VERIFY_IMPL', raising=False)
    assert decision(base64.b64decode(case['archive_b64'])) == case['python']


@pytest.mark.skipif(not os.environ.get('WORKSPACE_RESTORE_VERIFY_BIN'), reason='WORKSPACE_RESTORE_VERIFY_BIN not set (CI builds it)')
def test_rust_through_the_switch_matches_and_never_accepts_more(monkeypatch):
    from agent import workspace_decode as wd
    monkeypatch.setenv('WORKSPACE_RESTORE_VERIFY_IMPL', 'rust')
    monkeypatch.setattr(wd, 'DEFAULT_BIN', os.environ['WORKSPACE_RESTORE_VERIFY_BIN'])
    for c in FIXTURES:
        got = decision(base64.b64decode(c['archive_b64']))
        # Native decode-v1 alone decides acceptance; the reference stays test-only.
        assert got == c['rust'], c['name']
        if got.startswith('accept'):
            assert got == c['python'], c['name']
