"""The shared differential fixtures (tests/fixtures/tenant-assertion.v1.json, generated in
sbstndalton/noevia-rs by tools/gen-tenant-assertion.py from this module) against the live Python
reference, and, when TENANT_ASSERTION_BIN names a built binary, against the Rust leaf through the
real switch. CI copies nothing: it `cmp`s this file with noevia-rs' copy at the pinned ref."""
import json
import os
from pathlib import Path

import pytest

from agent import tenant_assertion as ta

FIXTURES = json.loads((Path(__file__).parent / 'fixtures' / 'tenant-assertion.v1.json').read_text())['cases']


def python_decision(i):
    ta._reset_for_tests()
    if i['op'] == 'secret_ref':
        return 'accept' if ta.secret_ref_matches(i['key'], i['user_id'], i['secret'], i['ref']) else 'reject: bad signature'
    headers = {'X-Cowork-User-ID': i['user_id'], ta.HEADER: i['assertion'], 'X-Cowork-Storage': i['storage'],
               'X-Cowork-Legacy-Owner': i['legacy_owner'], 'X-Cowork-Storage-Blocked': i['blocked']}
    reason = ta.verify(i['key'], headers, i['method'], i['path'], now=float(i['now']),
                       query=bytes.fromhex(i['query_hex']), body_hash=i['body_hash'])
    return 'accept' if reason is None else 'reject: ' + reason


def test_fixture_table_is_present_and_safe():
    assert len(FIXTURES) > 400
    for c in FIXTURES:
        assert not (c['rust'] == 'accept' and c['python'] != 'accept'), c['name']


@pytest.mark.parametrize('case', FIXTURES, ids=[c['name'] for c in FIXTURES])
def test_python_reference_still_decides_as_recorded(case, monkeypatch):
    monkeypatch.delenv('TENANT_ASSERTION_IMPL', raising=False)
    assert python_decision(case['input']) == case['python']


@pytest.mark.skipif(not os.environ.get('TENANT_ASSERTION_BIN'), reason='TENANT_ASSERTION_BIN not set (CI builds it)')
def test_rust_through_the_switch_matches_and_never_accepts_more(monkeypatch):
    monkeypatch.setenv('TENANT_ASSERTION_IMPL', 'rust')
    for c in FIXTURES:
        got = python_decision(c['input'])
        both = c['python'] == 'accept' and c['rust'] == 'accept'
        assert (got == 'accept') == both, c['name']
