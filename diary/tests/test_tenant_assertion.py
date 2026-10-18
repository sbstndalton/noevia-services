"""M2 tenant assertion, open-mode refusal and secretRef storage descriptors (#291, #292).

Synthetic tenants and keys only; nothing here touches a real Diary."""
import base64
import json
import secrets
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

import agent.app as appmod
from agent import tenant_assertion as ta
from tests.test_dedicated_storage import volume, A, B  # noqa: F401
from tests.tenant_binary import needs_binary

# Since #1071 the signature, the clock window and the bounds are the Rust leaf's alone (reason: rust refused).
pytestmark = needs_binary

KEY = 'synthetic-tenant-key-for-tests'
C = '33333333-3333-4333-8333-333333333333'  # deleted by one test; the tombstone is process-wide


@pytest.fixture(autouse=True)
def fresh_nonces():
    ta._reset_for_tests()
    yield
    ta._reset_for_tests()


def signed(user, method, path, extra=None, key=KEY, ts=None, nonce=None, query=b'', body=b''):
    headers = {'X-Cowork-User-ID': user, **(extra or {})}
    ts = int(time.time()) if ts is None else ts
    nonce = nonce or secrets.token_hex(16)
    body_hash = ta.sha256_hex(body) if ta.body_is_hashed(headers.get('Content-Type', '')) else ta.STREAM
    headers[ta.HEADER] = ta.sign(key, user, method, path, headers, ts, nonce, query, body_hash)
    return headers


def flip_last_char(value):
    """Always differs from `value` (a fixed replacement collides with it ~1/1024 for a random signature, #847).
    'A' and 'E' are both canonical final base64url chars of a 32-byte digest."""
    return value[:-1] + ('E' if value[-1] == 'A' else 'A')


def b64(obj):
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip('=')


# ── verify() ────────────────────────────────────────────────────────────────

def test_flip_last_char_always_changes_the_value():
    for v in ('v2.abcAA', 'v2.abcAE', 'v2.abcAB'):
        assert flip_last_char(v) != v and flip_last_char(v)[:-1] == v[:-1]


def test_valid_assertion_verifies_once_and_replay_is_refused():
    h = signed(A, 'GET', '/api/day')
    assert ta.verify(KEY, h, 'GET', '/api/day') is None
    assert ta.verify(KEY, h, 'GET', '/api/day') == 'replayed'


@pytest.mark.parametrize('mutate', [
    lambda h: {**h, 'X-Cowork-User-ID': B},                       # another tenant, same assertion
    lambda h: {**h, 'X-Cowork-Storage': b64({'kind': 'local'})},  # swapped storage descriptor
    lambda h: {**h, 'X-Cowork-Legacy-Owner': '1'},                # added legacy-owner marker
    lambda h: {**h, ta.HEADER: flip_last_char(h[ta.HEADER])},     # forged signature (always differs)
    lambda h: {k: v for k, v in h.items() if k != ta.HEADER},     # missing
    lambda h: {**h, ta.HEADER: 'v2.garbage'},                     # malformed
    lambda h: {**h, ta.HEADER: 'v1' + h[ta.HEADER][2:]},          # v1 is not accepted
])
def test_tampered_or_missing_assertion_is_refused(mutate):
    h = signed(A, 'GET', '/api/day', {'X-Cowork-Storage': b64({'kind': 'webdav', 'secretRef': 'a' * 32})})
    assert ta.verify(KEY, mutate(h), 'GET', '/api/day') is not None


def test_assertion_is_bound_to_method_path_and_key():
    h = signed(A, 'GET', '/api/day')
    assert ta.verify(KEY, h, 'DELETE', '/api/day') == 'rust refused'
    assert ta.verify(KEY, h, 'GET', '/api/internal/tenant') == 'rust refused'
    assert ta.verify('another-key', h, 'GET', '/api/day') == 'rust refused'


def test_assertion_is_bound_to_query_and_body():
    h = signed(A, 'POST', '/api/file', query=b'path=a', body=b'{"path":"a"}')
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=b', body_hash=ta.sha256_hex(b'{"path":"a"}')) == 'rust refused'
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=a', body_hash=ta.sha256_hex(b'{"path":"b"}')) == 'rust refused'
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=a', body_hash=ta.STREAM) == 'rust refused'
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=a', body_hash=ta.sha256_hex(b'{"path":"a"}')) is None


def test_json_bodies_are_hashed_and_other_media_types_sign_stream():
    assert ta.body_is_hashed('') and ta.body_is_hashed('application/json; charset=utf-8')
    assert not ta.body_is_hashed('application/zip') and not ta.body_is_hashed('multipart/form-data; boundary=x')


def test_more_signed_requests_than_the_cache_cap_are_all_accepted_and_replays_stay_refused(monkeypatch):
    # #1680: over the cap the Diary used to raise NonceCacheFull (503) for fresh, legitimate requests.
    monkeypatch.setattr(ta, '_nonce_cap', 5)
    now = 1_000_000.0
    sent = []
    for i in range(40):  # 8x the cap inside one replay window, spread over 20 seconds
        t = now + i * 0.5
        h = signed(A, 'GET', '/api/storage-status', ts=int(t))
        assert ta.verify(KEY, h, 'GET', '/api/storage-status', now=t) is None, f'fresh request {i} refused'
        sent.append(h)
    assert ta._nonce_count() <= 5 + 2, 'the cache must stay bounded (cap plus the newest second)'
    # Every replay inside the window is still caught, including nonces evicted to make room.
    replay_at = now + 20
    for i, h in enumerate(sent):
        assert ta.verify(KEY, h, 'GET', '/api/storage-status', now=replay_at) is not None, f'replay {i} accepted'


def test_expired_nonces_are_purged_first_and_never_raise_the_floor_for_live_traffic(monkeypatch):
    monkeypatch.setattr(ta, '_nonce_cap', 3)
    now = 1_000_000.0
    for i in range(3):
        assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now)), 'GET', '/x', now=now) is None
    later = now + ta.SKEW_S + 1  # all three are outside the clock window now
    h = signed(A, 'GET', '/x', ts=int(later) - ta.SKEW_S + 1)  # oldest timestamp still valid
    assert ta.verify(KEY, h, 'GET', '/x', now=later) is None
    assert ta._nonce_count() == 1


def test_a_replay_after_the_clock_steps_back_is_refused(monkeypatch):
    now = 1_000_000.0
    h = signed(A, 'GET', '/x', ts=int(now))
    assert ta.verify(KEY, h, 'GET', '/x', now=now) is None
    # Purge it (clock far ahead), then step the clock back into its window: still a replay.
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) + 500), 'GET', '/x', now=now + 500) is None
    assert ta.verify(KEY, h, 'GET', '/x', now=now + 1) is not None


def test_over_the_cap_http_answers_200_and_replay_answers_401(keyed, monkeypatch):
    monkeypatch.setattr(ta, '_nonce_cap', 1)
    first = signed(B, 'GET', '/api/storage-status')
    assert keyed.get('/api/storage-status', headers=first).status_code == 200
    for _ in range(3):
        assert keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')).status_code == 200
    assert keyed.get('/api/storage-status', headers=first).status_code == 401


def test_path_is_signed_percent_encoded_as_on_the_wire():
    # Invariant shared with diary-tenant-assertion.test.cjs: the signed path is the
    # encoded request-target path (ASGI raw_path / URL.pathname), never the decoded one.
    assert ta.wire_path({'path': '/api/a b', 'raw_path': b'/api/a%20b'}) == '/api/a%20b'
    assert ta.wire_path({'path': '/api/x'}) == '/api/x'
    h = signed(A, 'GET', '/api/a%20b')
    assert ta.verify(KEY, h, 'GET', '/api/a b') == 'rust refused'
    assert ta.verify(KEY, h, 'GET', ta.wire_path({'path': '/api/a b', 'raw_path': b'/api/a%20b'})) is None


def test_clock_skew_window():
    now = time.time()
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) - ta.SKEW_S - 5), 'GET', '/x', now=now) == 'rust refused'
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) + ta.SKEW_S + 5), 'GET', '/x', now=now) == 'rust refused'
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) - ta.SKEW_S + 5), 'GET', '/x', now=now) is None


def test_web_and_sidecar_derive_the_same_signature():
    # Golden vector shared with apps/web/server/diary-tenant-assertion.test.cjs.
    h = {'X-Cowork-User-ID': A, 'X-Cowork-Storage': 'eyJraW5kIjoibG9jYWwifQ'}
    assert ta.sign('k', A, 'post', '/api/file', h, 1700000000, '0' * 32, b'path=x', ta.sha256_hex(b'{"path":"a"}')) == \
        'v2.1700000000.00000000000000000000000000000000.' + GOLDEN_SIG
    assert ta.storage_secret_ref('k', A, 's3cret') == GOLDEN_REF


GOLDEN_SIG = 'nfewKW0P1sYJ_oPynPoP1cOgZio8VF0OmX3Q5gIvQbc'
GOLDEN_REF = '3bc946714e8992b28033c5c925ede7b1'


# ── the running app ─────────────────────────────────────────────────────────

@pytest.fixture
def keyed(volume, monkeypatch):  # noqa: F811
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    return TestClient(appmod.app)


def test_key_set_refuses_tenant_requests_without_a_valid_assertion(keyed):
    assert keyed.get('/api/storage-status', headers={'X-Cowork-User-ID': B}).status_code == 401
    ok = keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status'))
    assert ok.status_code == 200 and ok.json()['mode'] == 'managed'
    # Valid bearer-level access but another tenant's id under B's assertion.
    stolen = {**signed(B, 'GET', '/api/storage-status'), 'X-Cowork-User-ID': A}
    assert keyed.get('/api/storage-status', headers=stolen).status_code == 401
    replay = signed(B, 'GET', '/api/storage-status')
    assert keyed.get('/api/storage-status', headers=replay).status_code == 200
    assert keyed.get('/api/storage-status', headers=replay).status_code == 401


def test_signed_query_and_json_body_are_enforced_end_to_end(keyed):
    body = b'{"path":"notes.md"}'
    ct = {'Content-Type': 'application/json'}
    ok = keyed.get('/api/storage-status?x=1', headers=signed(B, 'GET', '/api/storage-status', query=b'x=1'))
    assert ok.status_code == 200
    swapped = signed(B, 'GET', '/api/storage-status', query=b'x=1')
    assert keyed.get('/api/storage-status?x=2', headers=swapped).status_code == 401
    tampered = signed(B, 'POST', '/api/file', ct, body=body)
    assert keyed.post('/api/file', content=b'{"path":"other.md"}', headers=tampered).status_code == 401
    # A JSON call cannot downgrade to the unhashed "stream" marker.
    downgraded = signed(B, 'POST', '/api/file', {'Content-Type': 'application/zip'}, body=body)
    assert keyed.post('/api/file', content=body, headers={**downgraded, **ct}).status_code == 401
    # The genuine body reaches the handler (replayed after hashing): not an auth failure.
    assert keyed.post('/api/file', content=body, headers=signed(B, 'POST', '/api/file', ct, body=body)).status_code not in (401, 503)


def test_health_without_tenant_still_passes_with_key(keyed):
    assert keyed.get('/api/health').status_code == 200


def test_tenant_delete_always_requires_the_assertion(keyed, tmp_path):
    assert keyed.get('/api/storage-status', headers=signed(C, 'GET', '/api/storage-status')).status_code == 200
    root = tmp_path / 'state' / 'users' / C
    assert root.exists()
    assert keyed.delete('/api/internal/tenant', headers={'X-Cowork-User-ID': C}).status_code == 401
    assert keyed.delete('/api/internal/tenant', headers=signed(C, 'GET', '/api/internal/tenant')).status_code == 401
    assert root.exists()
    assert keyed.delete('/api/internal/tenant', headers=signed(C, 'DELETE', '/api/internal/tenant')).status_code == 200
    assert not root.exists()


def test_in_handler_backstop_refuses_unasserted_tenant(volume, monkeypatch):  # noqa: F811
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    req = Request({'type': 'http', 'headers': [(b'x-cowork-user-id', B.encode())]})
    with pytest.raises(HTTPException) as exc:
        appmod._tenant_state(req)
    assert exc.value.status_code == 401


@pytest.mark.parametrize('bearer', ['', 'synthetic-service-token'])
def test_header_less_legacy_fallback_is_refused_when_keyed(volume, monkeypatch, bearer):  # noqa: F811
    # With a bearer set the fallback used to act as DIARY_LEGACY_USER_ID unsigned.
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    monkeypatch.setenv('DIARY_LEGACY_USER_ID', B)
    monkeypatch.setattr(appmod.get_state(), 'auth_token', bearer)
    with pytest.raises(HTTPException) as exc:
        appmod._tenant_state(Request({'type': 'http', 'headers': []}))
    assert exc.value.status_code == 401


def test_header_less_legacy_route_is_401_with_bearer_and_key(keyed, monkeypatch):
    monkeypatch.setenv('DIARY_LEGACY_USER_ID', B)
    monkeypatch.setattr(appmod.get_state(), 'auth_token', 'synthetic-service-token')
    auth = {'Authorization': 'Bearer synthetic-service-token'}
    assert keyed.get('/api/storage-status', headers=auth).status_code == 401
    assert keyed.get('/api/day', headers=auth).status_code == 401
    assert keyed.get('/api/health', headers=auth).status_code == 200  # probe, no tenant detail


def test_key_unset_accepts_unasserted_requests_and_logs_once(volume, monkeypatch, caplog):  # noqa: F811
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    monkeypatch.delenv('DIARY_TENANT_KEY', raising=False)
    monkeypatch.setattr(appmod, '_open_tenant_logged', False)
    client = TestClient(appmod.app)
    with caplog.at_level('WARNING', logger='diary'):
        for _ in range(3):
            assert client.get('/api/storage-status', headers={'X-Cowork-User-ID': B}).status_code == 200
    assert sum('DIARY_TENANT_KEY is unset' in r.getMessage() for r in caplog.records) == 1


# ── open mode ───────────────────────────────────────────────────────────────

def test_open_mode_refuses_to_start_unless_declared(monkeypatch):
    monkeypatch.delenv('DIARY_TENANT_KEY', raising=False)
    monkeypatch.delenv('DIARY_ALLOW_OPEN', raising=False)
    with pytest.raises(RuntimeError, match='DIARY_ALLOW_OPEN'):
        appmod.enforce_open_mode_policy('')
    monkeypatch.setenv('DIARY_ALLOW_OPEN', 'true')  # only the literal 1 declares it
    with pytest.raises(RuntimeError):
        appmod.enforce_open_mode_policy('')
    monkeypatch.setenv('DIARY_ALLOW_OPEN', '1')
    appmod.enforce_open_mode_policy('')
    monkeypatch.delenv('DIARY_ALLOW_OPEN')
    appmod.enforce_open_mode_policy('service-token')
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    appmod.enforce_open_mode_policy('')


def test_lifespan_applies_the_open_mode_policy(monkeypatch):
    monkeypatch.delenv('DIARY_TENANT_KEY', raising=False)
    monkeypatch.delenv('DIARY_ALLOW_OPEN', raising=False)
    monkeypatch.setattr(appmod.get_state(), 'auth_token', '')
    with pytest.raises(RuntimeError):
        with TestClient(appmod.app):
            pass


# ── secretRef descriptors (#292) ────────────────────────────────────────────

def _req(user, storage):
    return Request({'type': 'http', 'headers': [(b'x-cowork-user-id', user.encode()), (b'x-cowork-storage', b64(storage).encode())]})


def test_ref_only_descriptor_needs_the_secret_only_to_build_state(volume, monkeypatch):  # noqa: F811
    base = {'kind': 'webdav', 'baseUrl': 'http://offline.invalid/dav', 'username': 'synthetic', 'corpusRoot': 'Diary'}
    monkeypatch.setenv('DIARY_TENANT_KEY', KEY)
    monkeypatch.setattr(appmod, '_require_tenant_assertion', lambda r: None)
    ref = ta.storage_secret_ref(KEY, B, 'synthetic-secret')
    with pytest.raises(HTTPException) as exc:
        appmod._tenant_state(_req(B, {**base, 'secretRef': ref}), recover=False)
    assert exc.value.status_code == 428 and exc.value.detail == {'code': 'storage_credential_required'}
    with pytest.raises(HTTPException) as bad:
        appmod._tenant_state(_req(B, {**base, 'secretRef': ref, 'secret': 'not-the-secret'}), recover=False)
    assert bad.value.status_code == 400
    built = appmod._tenant_state(_req(B, {**base, 'secretRef': ref, 'secret': 'synthetic-secret'}), recover=False)
    assert appmod._tenant_state(_req(B, {**base, 'secretRef': ref}), recover=False) is built
    assert not any('synthetic-secret' in key for key in appmod._tenant_states)
    rotated = ta.storage_secret_ref(KEY, B, 'rotated-secret')
    with pytest.raises(HTTPException) as again:
        appmod._tenant_state(_req(B, {**base, 'secretRef': rotated}), recover=False)
    assert again.value.status_code == 428


def test_ref_only_descriptor_never_needs_a_secret_for_an_app_diary(volume, monkeypatch):  # noqa: F811
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    appmod._tenant_state(_req(B, {'kind': 'local'}), recover=False)  # fresh account → app diary
    state = appmod._tenant_state(_req(B, {'kind': 'webdav', 'baseUrl': 'http://offline.invalid', 'secretRef': 'b' * 32}), recover=False)
    assert state.cfg.get('corpus.backend') == 'managed'


def test_backup_without_the_secret_asks_for_it(volume, monkeypatch):  # noqa: F811
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    client = TestClient(appmod.app)
    assert client.get('/api/storage-status', headers={'X-Cowork-User-ID': B}).json()['mode'] == 'managed'
    storage = b64({'kind': 'webdav', 'baseUrl': 'https://cloud.example/dav', 'username': 'synthetic', 'corpusRoot': 'Diary', 'secretRef': 'c' * 32})
    r = client.post('/api/storage-backup', headers={'X-Cowork-User-ID': B, 'X-Cowork-Storage': storage})
    assert r.status_code == 428
    assert r.json()['detail'] == {'code': 'storage_credential_required'}


def test_real_cap_sustains_2000_signed_requests_per_second_for_a_full_window():
    # #1680 at the shipped cap: 2,000 req/s for 60 s is 120,000 nonces, over the 100k cap.
    now = 1_000_000.0
    for i in range(120_000):
        t = now + i / 2000
        assert ta._remember_nonce(f'{i:032x}', int(t), t) is None, f'request {i} refused'
    assert ta._nonce_count() <= ta._nonce_cap + 2000
    assert ta._remember_nonce(f'{119_999:032x}', int(now + 59.9995), now + 60) is not None


# ── replay-cache hardening (#1811) ──────────────────────────────────────────

def test_future_dated_flood_does_not_lock_out_legitimate_traffic(monkeypatch):
    monkeypatch.setattr(ta, '_nonce_cap', 20)
    now = 1_000_000.0
    # A flood across every future second the clock window accepts overflows the cap many times over.
    for i in range(400):
        ts = int(now) + 1 + i % ta.SKEW_S
        ta.verify(KEY, signed(A, 'GET', '/x', ts=ts), 'GET', '/x', now=now)
    assert ta._floor < int(now), 'a future flood must not raise the floor above now'
    for _ in range(5):
        assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now)), 'GET', '/x', now=now) is None
    assert ta._nonce_count() <= 20 + 6


def test_flood_at_exactly_now_plus_skew_then_legitimate_now(monkeypatch):
    monkeypatch.setattr(ta, '_nonce_cap', 10)
    now = 1_000_000.0
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) - 1), 'GET', '/x', now=now) is None
    for _ in range(50):
        ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) + ta.SKEW_S), 'GET', '/x', now=now)
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now)), 'GET', '/x', now=now) is None
    assert ta._nonce_count() <= 10 + 2
    # The evicted future second is still refused, not silently reopened for replay.
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) + ta.SKEW_S), 'GET', '/x', now=now) is not None


def test_assertion_signed_before_process_start_is_refused_but_a_fresh_one_is_not():
    start = 1_000_000.4
    ta._reset_for_tests(started_at=start)
    old = signed(A, 'GET', '/x', ts=1_000_000 - ta.RESTART_SKEW_S - 1)  # inside the replay window, signed pre-restart
    assert ta.verify(KEY, old, 'GET', '/x', now=start + 0.1) == 'signed before this process started'
    assert ta.verify(KEY, old, 'GET', '/x', now=start + 30) == 'signed before this process started'
    # Web signs per request with a truncated ts: the first request right after the start is accepted,
    # even one signed in the second the process started in (and the RESTART_SKEW_S allowance).
    for ts in (1_000_000 - ta.RESTART_SKEW_S, 1_000_000, 1_000_001):
        assert ta.verify(KEY, signed(A, 'GET', '/x', ts=ts), 'GET', '/x', now=start + 0.6) is None
    assert ta.RESTART_SKEW_S <= 2


def test_real_process_start_does_not_refuse_current_traffic():
    ta._reset_for_tests(started_at=time.time())
    assert ta.verify(KEY, signed(A, 'GET', '/x'), 'GET', '/x') is None


def test_concurrent_verify_accepts_each_nonce_exactly_once(monkeypatch):
    """Nonces spread over several seconds with a tiny cap: the cap is exceeded while other threads are
    verifying, so eviction (past and future seconds) and _lost pruning run under contention."""
    import threading
    monkeypatch.setattr(ta, '_nonce_cap', 6)
    # This test is about the cache under contention, not about the bound on leaf children (switch tests): let every thread through.
    monkeypatch.setattr(ta, '_slots', threading.BoundedSemaphore(64))
    now = time.time()
    base = int(now)
    # Past seconds, the current one and future ones (future evictions land in _lost).
    seconds = [base - 8, base - 6, base - 4, base - 2, base, base + 2, base + 4, base + 6, base + 8]
    unique = [signed(A, 'GET', '/x', ts=seconds[i % len(seconds)]) for i in range(360)]
    dup = signed(A, 'GET', '/x', ts=base)
    assert ta.verify(KEY, dup, 'GET', '/x', now=now) is None  # accepted once, before the storm
    results, errors = [], []
    lock = threading.Lock()
    barrier = threading.Barrier(8)
    evicted = []
    real_drop = ta._drop_second

    def counting_drop(second, at):
        evicted.append(second)
        return real_drop(second, at)

    monkeypatch.setattr(ta, '_drop_second', counting_drop)

    def work(slice_):
        try:
            barrier.wait()
            local = []
            for i, h in enumerate(slice_):
                local.append(('u', h, ta.verify(KEY, h, 'GET', '/x', now=now)))
                if i % 8 == 0:
                    local.append(('d', dup, ta.verify(KEY, dup, 'GET', '/x', now=now)))
            with lock:
                results.extend(local)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(unique[i::8],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # Eviction really ran (it could not with one second), and the future ones were remembered.
    assert evicted and ta._lost
    # A nonce is never accepted twice, whatever was evicted.
    accepted = [h[ta.HEADER] for k, h, r in results if r is None]
    assert len(accepted) == len(set(accepted))
    # The pre-accepted nonce is never accepted again, even once its bucket has been evicted (the floor / _lost
    # refuse it instead).
    assert not [1 for k, h, r in results if k == 'd' and r is None]
    # Every refusal is a replay-cache refusal, never a crash or another reason.
    assert {r for k, h, r in results if r is not None} <= {'replayed', 'replayed or older than the replay cache'}
    assert ta._nonce_count() <= ta._nonce_cap + 360


def test_concurrent_verify_same_second_exactly_once(monkeypatch):
    """The plain exactly-once case: one second, one duplicate hammered from every thread."""
    import threading
    now = time.time()
    dup = signed(A, 'GET', '/x', ts=int(now))
    results = []
    barrier = threading.Barrier(8)

    def work():
        barrier.wait()
        results.extend(ta.verify(KEY, dup, 'GET', '/x', now=now) for _ in range(40))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for r in results if r is None) == 1
