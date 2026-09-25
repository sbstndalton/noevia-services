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


def b64(obj):
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip('=')


# ── verify() ────────────────────────────────────────────────────────────────

def test_valid_assertion_verifies_once_and_replay_is_refused():
    h = signed(A, 'GET', '/api/day')
    assert ta.verify(KEY, h, 'GET', '/api/day') is None
    assert ta.verify(KEY, h, 'GET', '/api/day') == 'replayed'


@pytest.mark.parametrize('mutate', [
    lambda h: {**h, 'X-Cowork-User-ID': B},                       # another tenant, same assertion
    lambda h: {**h, 'X-Cowork-Storage': b64({'kind': 'local'})},  # swapped storage descriptor
    lambda h: {**h, 'X-Cowork-Legacy-Owner': '1'},                # added legacy-owner marker
    lambda h: {**h, ta.HEADER: h[ta.HEADER][:-2] + 'AA'},         # forged signature
    lambda h: {k: v for k, v in h.items() if k != ta.HEADER},     # missing
    lambda h: {**h, ta.HEADER: 'v2.garbage'},                     # malformed
    lambda h: {**h, ta.HEADER: 'v1' + h[ta.HEADER][2:]},          # v1 is not accepted
])
def test_tampered_or_missing_assertion_is_refused(mutate):
    h = signed(A, 'GET', '/api/day', {'X-Cowork-Storage': b64({'kind': 'webdav', 'secretRef': 'a' * 32})})
    assert ta.verify(KEY, mutate(h), 'GET', '/api/day') is not None


def test_assertion_is_bound_to_method_path_and_key():
    h = signed(A, 'GET', '/api/day')
    assert ta.verify(KEY, h, 'DELETE', '/api/day') == 'bad signature'
    assert ta.verify(KEY, h, 'GET', '/api/internal/tenant') == 'bad signature'
    assert ta.verify('another-key', h, 'GET', '/api/day') == 'bad signature'


def test_assertion_is_bound_to_query_and_body():
    h = signed(A, 'POST', '/api/file', query=b'path=a', body=b'{"path":"a"}')
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=b', body_hash=ta.sha256_hex(b'{"path":"a"}')) == 'bad signature'
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=a', body_hash=ta.sha256_hex(b'{"path":"b"}')) == 'bad signature'
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=a', body_hash=ta.STREAM) == 'bad signature'
    assert ta.verify(KEY, h, 'POST', '/api/file', query=b'path=a', body_hash=ta.sha256_hex(b'{"path":"a"}')) is None


def test_json_bodies_are_hashed_and_other_media_types_sign_stream():
    assert ta.body_is_hashed('') and ta.body_is_hashed('application/json; charset=utf-8')
    assert not ta.body_is_hashed('application/zip') and not ta.body_is_hashed('multipart/form-data; boundary=x')


def test_full_nonce_cache_purges_expired_then_refuses_instead_of_evicting(monkeypatch):
    monkeypatch.setattr(ta, '_nonce_cap', 2)
    now = 1_000_000.0
    assert ta._remember_nonce('a' * 32, now)
    assert ta._remember_nonce('b' * 32, now + 1)
    with pytest.raises(ta.NonceCacheFull):
        ta._remember_nonce('c' * 32, now + 2)
    assert not ta._remember_nonce('a' * 32, now + 3), 'a live nonce was evicted'
    # Once the oldest expires it is purged and there is room again.
    assert ta._remember_nonce('c' * 32, now + 2 * ta.SKEW_S + 0.5)


def test_full_nonce_cache_answers_503_and_logs_once(keyed, monkeypatch, caplog):
    monkeypatch.setattr(ta, '_nonce_cap', 1)
    monkeypatch.setattr(appmod, '_nonce_full_logged', False)
    assert keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')).status_code == 200
    with caplog.at_level('ERROR', logger='diary'):
        for _ in range(2):
            assert keyed.get('/api/storage-status', headers=signed(B, 'GET', '/api/storage-status')).status_code == 503
    assert sum('nonce cache full' in r.getMessage() for r in caplog.records) == 1


def test_path_is_signed_percent_encoded_as_on_the_wire():
    # Invariant shared with diary-tenant-assertion.test.cjs: the signed path is the
    # encoded request-target path (ASGI raw_path / URL.pathname), never the decoded one.
    assert ta.wire_path({'path': '/api/a b', 'raw_path': b'/api/a%20b'}) == '/api/a%20b'
    assert ta.wire_path({'path': '/api/x'}) == '/api/x'
    h = signed(A, 'GET', '/api/a%20b')
    assert ta.verify(KEY, h, 'GET', '/api/a b') == 'bad signature'
    assert ta.verify(KEY, h, 'GET', ta.wire_path({'path': '/api/a b', 'raw_path': b'/api/a%20b'})) is None


def test_clock_skew_window():
    now = time.time()
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) - ta.SKEW_S - 5), 'GET', '/x', now=now) == 'outside clock window'
    assert ta.verify(KEY, signed(A, 'GET', '/x', ts=int(now) + ta.SKEW_S + 5), 'GET', '/x', now=now) == 'outside clock window'
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
