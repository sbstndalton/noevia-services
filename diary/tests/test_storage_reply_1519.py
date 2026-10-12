"""#1519: a storage listing that answers 2xx with something that is not a usable listing (a PHP
notice in front of the XML, a multistatus entry with no path) is a storage failure answered as 502,
never an unhandled 500. Synthetic server and credentials only; the transport is a fake."""
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import agent.app as appmod
from agent.storage_backoff import StorageReply
from agent.webdav import WebDAVCorpusBackend
from tests.test_dedicated_storage import volume  # noqa: F401 (fixture)

B = '33333333-3333-4333-8333-333333333333'
BASE = 'https://dav.example.test/remote.php/dav/files/alice/'
GOOD = ('<?xml version="1.0"?><d:multistatus xmlns:d="DAV:">'
        '<d:response><d:href>/remote.php/dav/files/alice/Diary/AI%20Memory/</d:href><d:propstat><d:status>HTTP/1.1 200 OK</d:status>'
        '<d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>'
        '<d:response><d:href>/remote.php/dav/files/alice/Diary/AI%20Memory/note.md</d:href><d:propstat><d:status>HTTP/1.1 200 OK</d:status>'
        '<d:prop><d:resourcetype/><d:getetag>"e1"</d:getetag><d:getlastmodified>Mon, 12 Oct 2026 10:00:00 GMT</d:getlastmodified></d:prop></d:propstat></d:response>'
        '</d:multistatus>')


class ReplyDav(httpx.BaseTransport):
    """Answers each PROPFIND with `body` (status 207), counting the requests that reached it."""

    def __init__(self, body):
        self.body, self.calls = body, 0

    def handle_request(self, request):
        self.calls += 1
        return httpx.Response(207, text=self.body, request=request)


@pytest.fixture
def dav_with(monkeypatch):
    def use(body):
        fake = ReplyDav(body)
        monkeypatch.setattr(httpx, 'HTTPTransport', lambda *a, **k: fake)
        return fake
    return use


def headers(secret='synthetic-secret'):
    descriptor = {'kind': 'webdav', 'baseUrl': BASE, 'username': 'alice', 'secret': secret, 'corpusRoot': 'Diary'}
    storage = base64.urlsafe_b64encode(json.dumps(descriptor).encode()).decode()
    return {'X-Cowork-User-ID': B, 'X-Cowork-Storage': storage}


def client():
    return TestClient(appmod.app, raise_server_exceptions=False)


def test_a_usable_listing_still_lists(volume, dav_with):
    dav_with(GOOD)
    r = client().get('/api/files', params={'path': 'AI Memory'}, headers=headers())
    assert r.status_code == 200, r.text
    assert [f['name'] for f in r.json()['files']] == ['note.md']


@pytest.mark.parametrize('body', [
    '<br />\n<b>Warning</b>: synthetic PHP notice<br />\n' + GOOD,   # a notice printed before the XML
    '<?xml version="1.0"?><d:multistatus xmlns:d="DAV:"><d:response><d:propstat/></d:response></d:multistatus>',  # no href
    '<?xml version="1.0"?><d:other xmlns:d="DAV:"/>',              # not a multistatus
])
def test_an_unusable_2xx_listing_is_502_not_500(volume, dav_with, body):
    dav_with(body)
    r = client().get('/api/files', params={'path': 'AI Memory'}, headers=headers())
    assert r.status_code == 502, (r.status_code, r.text[:200])
    assert r.json()['code'] == 'storageUpstream'


def test_the_parser_still_raises_a_value_error_for_other_callers():
    assert issubclass(StorageReply, ValueError)
    backend = WebDAVCorpusBackend(BASE, 'alice', 'synthetic-secret', transport=ReplyDav('<not xml'))
    with pytest.raises(ValueError):
        backend.list_dir('AI Memory')
