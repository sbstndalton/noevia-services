"""Tenant assertion (TENANT_ASSERTION_IMPL, retired in #1071: always the Rust leaf).

tests/fixtures/tenant-assertion.v1.json (generated in sbstndalton/noevia-rs by tools/gen-tenant-assertion.py from the
former Python module; CI `cmp`s it with noevia-rs' copy at the pinned ref) holds, per synthetic case, what Python decided
and what the Rust leaf decided. tests/fixtures/tenant-assertion-py.v1.json freezes what the old module answered for 6000
more seeded synthetic requests (shape and signature mutations, clock skews, odd tenants, paths, queries, bodies and
headers), for the restart guard and for secretRef: Python's stateless decision (`py`), the leaf's (`rust`) and the live
composition's (`live`: both had to accept, what ran). Both files are frozen and sha256-pinned: the Python rules are gone,
and neither may be regenerated through the Rust path (that would erase the independent reference). Provenance (also in the
frozen file's `note`): frozen from noevia-services at e5f0c04 with the tenant-assertion binary built from the noevia-rs ref
the Dockerfile pins.

What must hold, and is counted below: the module (which now asks the leaf alone for the signature, the clock window and the
bounds, and keeps the header shape, the restart guard and the nonce cache) accepts a request exactly when the live
composition did, so nothing is accepted that Python refused; the leaf refuses some inputs Python accepted (non-ASCII or
over-long tenants, whitespace or newlines in the storage headers, over-long fields), which stays refused. The binary is
needed: skipped without it unless TENANT_ASSERTION_REQUIRED=1 (CI builds it and sets it)."""
import hashlib
import json
from pathlib import Path

import pytest

from agent import tenant_assertion as ta
from tests.tenant_binary import needs_binary

FIXTURE = Path(__file__).parent / 'fixtures' / 'tenant-assertion.v1.json'
FROZEN = Path(__file__).parent / 'fixtures' / 'tenant-assertion-py.v1.json'
# The frozen references are pinned: changing either is a deliberate act that also changes these.
PINS = {FIXTURE: '46f7441df59bfb57bad188add02c29ed6950675d4de1906db6ad412a86b01864', FROZEN: '76ca446f8e3e2fbb4e1ef0a4cb4fb789dfa9e3fa8b1a2b18a51bbd16acb64e43'}
CASES = json.loads(FIXTURE.read_text())['cases']
FROZEN_DATA = json.loads(FROZEN.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(i, started_at=None):
    """The module's decision for one fixture input, with a fresh nonce cache (accept, or reject: reason)."""
    ta._reset_for_tests(started_at)
    if i['op'] == 'secret_ref':
        return 'accept' if ta.secret_ref_matches(i['key'], i['user_id'], i['secret'], i['ref']) else 'reject: bad signature'
    headers = {'X-Cowork-User-ID': i['user_id'], ta.HEADER: i['assertion'], 'X-Cowork-Storage': i['storage'],
               'X-Cowork-Legacy-Owner': i['legacy_owner'], 'X-Cowork-Storage-Blocked': i['blocked']}
    reason = ta.verify(i['key'], headers, i['method'], i['path'], now=float(i['now']),
                       query=bytes.fromhex(i['query_hex']), body_hash=i['body_hash'])
    return 'accept' if reason is None else 'reject: ' + reason


def test_the_frozen_references_are_the_pinned_files():
    for path, pin in PINS.items():
        assert sha(path) == pin, path.name
    assert len(CASES) > 400
    assert len(FROZEN_DATA['cases']) == 6000 and len(FROZEN_DATA['guard']) == 8 and len(FROZEN_DATA['refs']) == 1500
    assert 'freeze script (sha256 ' in FROZEN_DATA['note'] and 'sha256 ' in FROZEN_DATA['note'].split('binary built')[1]
    for c in CASES:
        assert not (c['rust'] == 'accept' and c['python'] != 'accept'), c['name']


@needs_binary
@pytest.mark.parametrize('case', CASES, ids=[c['name'] for c in CASES])
def test_fixture_cases_are_decided_by_the_leaf_and_never_accept_more_than_python_did(case):
    got = decide(case['input'])
    # Accepted exactly when the leaf accepted (the restart guard is off here); never when Python refused.
    assert (got == 'accept') == (case['rust'] == 'accept'), case['name']
    assert got == 'accept' or got.startswith('reject: '), got
    if got == 'accept':
        assert case['python'] == 'accept', case['name']


def frozen_headers(c):
    return {**c['headers'], ta.HEADER: c['assertion']}


def verify_frozen(c, started_at=None):
    ta._reset_for_tests(started_at)
    return ta.verify(c['key'], frozen_headers(c), c['method'], c['path'], now=float(c['now']),
                     query=bytes.fromhex(c['query_hex']), body_hash=c['body_hash'])


@needs_binary
def test_frozen_requests_are_accepted_exactly_when_the_live_composition_accepted_them():
    count = {'total': 0, 'both_accept': 0, 'python_only': 0, 'leaf_only': 0, 'neither': 0, 'accepted_now': 0}
    stricter = {}
    for c in FROZEN_DATA['cases']:
        py, rust, live = c['result']['py'], c['result']['rust'], c['result']['live']
        reason = verify_frozen(c)
        label = f"case {c['i']} ({c['mutation']})"
        # Exactly what ran: live accepted when Python and the leaf both did.
        assert (reason is None) == (live == 'accept'), label
        # The leaf never accepts what Python refused, so nothing is accepted that Python refused.
        assert not (rust == 'accept' and py != 'accept'), label
        if reason is None:
            assert py == 'accept' and rust == 'accept', label
        count['total'] += 1
        if py == 'accept' and rust == 'accept': count['both_accept'] += 1
        elif py == 'accept': count['python_only'] += 1; stricter[c['mutation']] = stricter.get(c['mutation'], 0) + 1
        elif rust == 'accept': count['leaf_only'] += 1
        else: count['neither'] += 1
        if reason is None: count['accepted_now'] += 1
    assert count == {'total': 6000, 'both_accept': 566, 'python_only': 310, 'leaf_only': 0, 'neither': 5124, 'accepted_now': 566}, count


@needs_binary
def test_frozen_restart_guard_cases_are_decided_as_the_old_composition_decided_them():
    for c in FROZEN_DATA['guard']:
        reason = verify_frozen(c, started_at=c['started_at'])
        assert (reason is None) == (c['result']['live'] == 'accept'), c['i']
    # The leaf knows nothing of the process start: it accepts all eight, and the guard alone refuses the one signed before.
    assert [c['result']['rust'] for c in FROZEN_DATA['guard']] == ['accept'] * 8
    assert [c['result']['live'] for c in FROZEN_DATA['guard']].count('accept') == 7
    refused = [c for c in FROZEN_DATA['guard'] if c['result']['live'] != 'accept']
    assert len(refused) == 1 and verify_frozen(refused[0], started_at=refused[0]['started_at']) == 'signed before this process started'


@needs_binary
def test_frozen_secret_refs_are_decided_by_the_leaf_and_a_non_ascii_ref_is_a_mismatch_not_an_error():
    count = {'total': 0, 'accept': 0, 'python_only': 0, 'raised_before': 0}
    for r in FROZEN_DATA['refs']:
        res = r['result']
        got = ta.secret_ref_matches(r['key'], r['user_id'], r['secret'], r['ref'])
        label = f"ref {r['i']} ({r['mutation']})"
        # Never accepted when the leaf or Python refused; the old code raised on a non-ASCII ref (a 500), which is now a mismatch.
        assert got == (res['live'] == 'accept'), label
        assert not (res['rust'] == 'accept' and res['py'] != 'accept'), label
        count['total'] += 1
        if got: count['accept'] += 1
        if res['py'] == 'accept' and res['rust'] != 'accept': count['python_only'] += 1
        if res['live'].startswith('raised'): count['raised_before'] += 1; assert got is False, label
    assert count == {'total': 1500, 'accept': 399, 'python_only': 138, 'raised_before': 122}, count
