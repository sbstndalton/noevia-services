"""#1166: storage 429/401/403/5xx become clean answers, and a rejected or throttled storage server
is left alone for a cool-down. Synthetic server and credentials only; requests are counted with a
fake transport."""
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import agent.app as appmod
from agent import storage_backoff
from agent.storage_backoff import StorageBackoff, StorageGate, parse_retry_after
from agent.webdav import WebDAVCorpusBackend
from tests.test_dedicated_storage import volume  # noqa: F401 (fixture)

B = '22222222-2222-4222-8222-222222222222'
BASE = 'https://dav.example.test/remote.php/dav/files/alice/'


class FakeDav(httpx.BaseTransport):
    """Answers every request with `status` (+ headers) and counts the calls that reached it."""

    def __init__(self, status=200, headers=None):
        self.status, self.headers, self.calls = status, headers or {}, 0

    def handle_request(self, request):
        self.calls += 1
        if self.status == 200 and request.method == 'PROPFIND':
            body = '<?xml version="1.0"?><d:multistatus xmlns:d="DAV:"/>'
            return httpx.Response(207, text=body, request=request)
        return httpx.Response(self.status, headers=self.headers, request=request)


@pytest.fixture
def dav(monkeypatch):
    fake = FakeDav()
    monkeypatch.setattr(httpx, 'HTTPTransport', lambda *a, **k: fake)
    return fake


def storage_header(secret='synthetic-secret'):
    descriptor = {'kind': 'webdav', 'baseUrl': BASE, 'username': 'alice', 'secret': secret, 'corpusRoot': 'Diary'}
    return base64.urlsafe_b64encode(json.dumps(descriptor).encode()).decode()


def headers(secret='synthetic-secret', **extra):
    return {'X-Cowork-User-ID': B, 'X-Cowork-Storage': storage_header(secret), **extra}


def client():
    return TestClient(appmod.app, raise_server_exceptions=False)


# ---- response mapping -------------------------------------------------------

def test_429_is_503_with_retry_after_from_upstream(volume, dav):
    dav.status, dav.headers = 429, {'Retry-After': '120'}
    r = client().get('/api/files', headers=headers())
    assert r.status_code == 503
    assert r.headers['Retry-After'] == '120'
    assert r.json()['code'] == 'storageThrottled'
    assert 'try again' in r.json()['detail'].lower()


def test_429_without_retry_after_defaults_to_60(volume, dav):
    dav.status = 429
    r = client().get('/api/files', headers=headers())
    assert (r.status_code, r.headers['Retry-After']) == (503, '60')


@pytest.mark.parametrize('status', [401, 403])
def test_login_refusal_is_424_on_listing_and_read(volume, dav, status):
    dav.status = status
    c = client()
    listing = c.get('/api/files', headers=headers())
    assert listing.status_code == 424 and listing.json()['code'] == 'storageLoginRejected'
    reading = c.post('/api/file', json={'path': 'a.md'}, headers=headers())
    assert reading.status_code == 424 and reading.json()['code'] == 'storageLoginRejected'


@pytest.mark.parametrize('status', [500, 502, 503, 507])
def test_other_5xx_is_502(volume, dav, status):
    dav.status = status
    r = client().get('/api/files', headers=headers())
    assert r.status_code == 502 and r.json()['code'] == 'storageUpstream'


def test_unreachable_storage_is_502(volume, monkeypatch):
    class Down(httpx.BaseTransport):
        def handle_request(self, request):
            raise httpx.ConnectError('synthetic refused', request=request)
    monkeypatch.setattr(httpx, 'HTTPTransport', lambda *a, **k: Down())
    r = client().get('/api/files', headers=headers())
    assert r.status_code == 502


def test_write_path_429_is_503_and_login_refusal_is_424(volume, dav, monkeypatch):
    # Saving a file goes through the unguarded route: the app-level handler maps it (#1166).
    monkeypatch.setattr(appmod, '_reindex_dirty', lambda st: None)
    body = {'path': 'new.md', 'content': 'x', 'version': None}
    dav.status, dav.headers = 429, {'Retry-After': '30'}
    r = client().put('/api/file', json=body, headers=headers())
    assert (r.status_code, r.headers['Retry-After']) == (503, '30')
    storage_backoff._gates.clear()
    dav.status, dav.headers = 401, {}
    r = client().put('/api/file', json=body, headers=headers('another-secret'))
    assert r.status_code == 424 and r.json()['code'] == 'storageLoginRejected'


def test_a_non_storage_httpx_error_is_not_mistaken_for_storage():
    request = httpx.Request('GET', 'https://llm.example.test/v1')
    err = httpx.HTTPStatusError('llm', request=request, response=httpx.Response(401, request=request))
    assert not storage_backoff.is_storage_error(err)


# ---- cool-down --------------------------------------------------------------

def test_rejected_login_suppresses_repeat_upstream_calls(volume, dav):
    dav.status = 401
    c = client()
    assert c.get('/api/files', headers=headers()).status_code == 424
    first = dav.calls
    assert first >= 1
    for _ in range(5):
        r = c.get('/api/files', headers=headers())
        assert r.status_code == 424 and r.json()['code'] == 'storageLoginRejected'
    assert dav.calls == first  # not one more request reached storage


def test_changed_credentials_are_tried_immediately(volume, dav):
    dav.status = 401
    c = client()
    c.get('/api/files', headers=headers())
    calls = dav.calls
    c.get('/api/files', headers=headers())
    assert dav.calls == calls
    dav.status = 200
    r = c.get('/api/files', headers=headers('a-new-secret'))
    assert r.status_code == 200 and dav.calls == calls + 1


def test_explicit_retry_resets_login_cooldown(volume, dav):
    dav.status = 401
    c = client()
    c.get('/api/files', headers=headers())
    calls = dav.calls
    c.get('/api/files', headers=headers())
    assert dav.calls == calls
    dav.status = 200
    r = c.get('/api/files', headers=headers(**{'X-Cowork-Storage-Retry': '1'}))
    assert r.status_code == 200 and dav.calls == calls + 1


def test_explicit_retry_does_not_skip_a_throttle(volume, dav):
    dav.status, dav.headers = 429, {'Retry-After': '90'}
    c = client()
    c.get('/api/files', headers=headers())
    calls = dav.calls
    r = c.get('/api/files', headers=headers(**{'X-Cowork-Storage-Retry': '1'}))
    assert r.status_code == 503 and dav.calls == calls
    assert 1 <= int(r.headers['Retry-After']) <= 90


def test_429_is_honoured_then_storage_is_called_again():
    now = [1000.0]
    fake = FakeDav(429, {'Retry-After': '10'})
    backend = WebDAVCorpusBackend(BASE, 'alice', 'synthetic', transport=fake)
    backend.storage_gate._clock = lambda: now[0]
    with pytest.raises(httpx.HTTPStatusError):
        backend.list_dir('')
    with pytest.raises(StorageBackoff) as held:
        backend.list_dir('')
    assert held.value.kind == 'throttle' and fake.calls == 1 and held.value.retry_after == 10
    now[0] += 11
    fake.status = 200
    assert backend.list_dir('') == [] and fake.calls == 2


def test_background_replay_respects_the_cooldown(volume, dav):
    # The journal replay / reindex use the same backend, so they stop too (and count as a transient outage).
    from agent.corpus_store import is_transient_failure
    dav.status = 401
    backend = WebDAVCorpusBackend(BASE, 'alice', 'bg-secret')
    with pytest.raises(httpx.HTTPStatusError):
        backend.get_text('Entries/a.md')
    calls = dav.calls
    with pytest.raises(StorageBackoff) as stopped:
        backend.get_text('Entries/a.md')
    with pytest.raises(StorageBackoff):
        backend.put('Entries/a.md', b'x')
    with pytest.raises(StorageBackoff):
        backend.exists('Entries/a.md')
    assert dav.calls == calls
    assert is_transient_failure(stopped.value)


def test_gate_is_thread_safe_and_shared_per_credential():
    import threading
    gate = StorageGate()
    errors = []

    def hammer():
        try:
            for i in range(200):
                gate.record(401 if i % 3 == 0 else 429, '5')
                try:
                    gate.check()
                except StorageBackoff:
                    pass
                gate.reset_login()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    assert storage_backoff.gate_for('a', 'b') is storage_backoff.gate_for('a', 'b')
    assert storage_backoff.gate_for('a', 'b') is not storage_backoff.gate_for('a', 'c')


def test_retry_after_parsing():
    assert parse_retry_after('120') == 120
    assert parse_retry_after(None) == 60
    assert parse_retry_after('garbage') == 60
    assert parse_retry_after('0') == 1
    assert parse_retry_after('999999') == 3600
    assert 1 <= parse_retry_after('Wed, 21 Oct 2099 07:28:00 GMT') <= 3600


# ---- review follow-ups: 403 scope, retry rate limit, gate table cap -----------

class PathDav(httpx.BaseTransport):
    def __init__(self, forbidden):
        self.forbidden, self.calls = set(forbidden), []

    def handle_request(self, request):
        self.calls.append(request.url.path)
        if request.url.path in self.forbidden:
            return httpx.Response(403, request=request)
        return httpx.Response(200, text='x', headers={'ETag': '"e"'}, request=request)


def test_403_on_one_path_does_not_block_other_paths():
    fake = PathDav({'/remote.php/dav/files/alice/Diary/secret.md'})
    backend = WebDAVCorpusBackend(BASE, 'alice', 'p403', transport=fake)
    with pytest.raises(httpx.HTTPStatusError):
        backend.get_text('Diary/secret.md')
    assert backend.get_text('Diary/ok.md')[0] == 'x'
    assert len(fake.calls) == 2


def test_403_on_the_account_root_closes_the_gate():
    fake = PathDav({'/remote.php/dav/files/alice/'})
    backend = WebDAVCorpusBackend(BASE, 'alice', 'proot', transport=fake)
    with pytest.raises(httpx.HTTPStatusError):
        backend.exists('')
    with pytest.raises(StorageBackoff):
        backend.get_text('Diary/ok.md')


def test_s3_403_on_one_key_does_not_block_another():
    from agent.s3_storage import S3CorpusBackend
    fake = PathDav({'/bkt/missing.md'})
    backend = S3CorpusBackend('https://s3.example.test', 'bkt', 'AK', 'SK', transport=fake) \
        if 'transport' in S3CorpusBackend.__init__.__code__.co_varnames else None
    if backend is None:
        backend = S3CorpusBackend('https://s3.example.test', 'bkt', 'AK', 'SK')
        backend._client = httpx.Client(base_url=backend.base, transport=storage_backoff.GatedTransport(fake, backend.storage_gate))
    with pytest.raises(httpx.HTTPStatusError):
        backend.get_text('missing.md')
    assert backend.get_text('other.md')[0] == 'x'
    assert fake.calls == ['/bkt/missing.md', '/bkt/other.md']


def test_explicit_retry_reset_is_rate_limited():
    now = [1000.0]
    gate = StorageGate(clock=lambda: now[0])
    gate.record(401)
    assert gate.reset_login() is True
    gate.record(401)
    now[0] += 10
    assert gate.reset_login() is False
    with pytest.raises(StorageBackoff):
        gate.check()
    now[0] += 25
    assert gate.reset_login() is True


def test_reset_outcomes_are_logged_with_hash_prefix_only(caplog):
    # #1185: live QA can see whether a Retry reset was accepted or ignored by the 30 s limit.
    import logging
    secret = 'synthetic-secret-value'
    gate = storage_backoff.gate_for('https://dav.example', 'acct', secret)
    now = [1000.0]
    gate._clock = lambda: now[0]
    gate.record(401)
    with caplog.at_level(logging.INFO, logger='agent.storage_backoff'):
        assert gate.reset_login() is True
        gate.record(401)
        now[0] += 5
        assert gate.reset_login() is False
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(messages) == 2, messages
    assert 'accepted' in messages[0] and 'ignored' in messages[1]
    assert all(r.levelno == logging.INFO for r in caplog.records)
    prefix = gate._label
    assert len(prefix) == 8 and all(prefix in m for m in messages)
    assert all(secret not in m and 'acct' not in m and 'dav.example' not in m for m in messages)


def test_reset_with_no_cooldown_logs_nothing(caplog):
    import logging
    gate = StorageGate()
    with caplog.at_level(logging.INFO, logger='agent.storage_backoff'):
        assert gate.reset_login() is False
    assert not caplog.records


def test_gate_table_is_hard_capped():
    for i in range(storage_backoff._MAX_GATES + 40):
        storage_backoff.gate_for('cred', str(i)).record(401)  # every gate closed: none idle
    assert len(storage_backoff._gates) <= storage_backoff._MAX_GATES
