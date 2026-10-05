import asyncio
from pathlib import Path

import httpx
import pytest
from app import downloader, hf


@pytest.mark.parametrize('url,allowed', [
    ('https://huggingface.co/a', True), ('https://HUGGINGFACE.CO:443/a', True),
    ('http://huggingface.co/a', False), ('https://huggingface.co:444/a', False),
    ('https://huggingface.co.evil.test/a', False), ('https://evil.test/huggingface.co', False),
    ('https://evil.test/?host=huggingface.co', False), ('https://huggingface.co@evil.test/a', False),
    ('https://user@huggingface.co/a', False), ('https://huggingface.co./a', False),
    ('https://huggingface.co:bad/a', False), ('https://[broken/a', False),
])
def test_token_origin(url, allowed):
    assert hf.is_token_origin(url) is allowed


@pytest.mark.parametrize('parallel', [False, True])
def test_download_redirect_headers_and_ranges(monkeypatch, tmp_path, parallel):
    captured = []
    payload = b'synthetic-model-bytes'
    def respond(req):
        captured.append(req)
        if req.url.host == 'huggingface.co':
            return httpx.Response(302, headers={'location': 'https://cdn.example/model'})
        if req.method == 'HEAD':
            return httpx.Response(200, headers={'content-length': str(len(payload)), 'accept-ranges': 'bytes'})
        part = req.headers.get('range')
        if part:
            start, end = part.removeprefix('bytes=').split('-')
            end = end or str(len(payload) - 1)  # an open range is answered with its real end (RFC 9110)
            data = payload[int(start):int(end)+1]
            return httpx.Response(206, content=data, headers={'content-range': f'bytes {start}-{end}/{len(payload)}'})
        return httpx.Response(200, content=payload)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(downloader.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(hf, 'get_token', lambda: 'synthetic-token')
    monkeypatch.setattr(downloader, 'PARALLEL_MIN_SIZE', 1 if parallel else 1000)
    monkeypatch.setattr(downloader, 'PARALLEL_CHUNKS', 2)
    job = downloader.DownloadJob('test', 'test/repo', 'model', 'https://huggingface.co/model',
                                 tmp_path/'model', tmp_path/'part', 0)
    if not parallel:
        job.temp_path.write_bytes(payload[:3])
        # A partial is only resumed when it names the upload it came from (If-Range).
        (tmp_path / 'part.validator').write_text('"etag-1"')
    asyncio.run(downloader.DownloadManager()._stream(job))
    assert job.temp_path.read_bytes() == payload
    assert any(r.method == 'GET' and 'range' in r.headers for r in captured)
    for req in captured:
        assert req.headers.get('authorization') == ('Bearer synthetic-token' if req.url.host == 'huggingface.co' else None)


def test_failed_parallel_retry_fetches_all_ranges(monkeypatch, tmp_path):
    payload = b'abcdefghijklmnop'
    requested = []
    attempt = 0

    def respond(req):
        requested.append((attempt, req.method, req.headers.get('range')))
        if req.method == 'HEAD':
            return httpx.Response(200, headers={'content-length': str(len(payload)), 'accept-ranges': 'bytes'})
        start, end = map(int, req.headers['range'].removeprefix('bytes=').split('-'))
        if attempt == 1 and start == 8:
            return httpx.Response(503)
        return httpx.Response(206, content=payload[start:end + 1],
                              headers={'content-range': f'bytes {start}-{end}/{len(payload)}'})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(downloader.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(hf, 'get_token', lambda: None)
    monkeypatch.setattr(downloader.db, 'record_download', lambda **kw: None)
    monkeypatch.setattr(downloader, 'PARALLEL_MIN_SIZE', 1)
    monkeypatch.setattr(downloader, 'PARALLEL_CHUNKS', 2)
    dest = tmp_path / 'model.gguf'
    temp = tmp_path / 'model.gguf.download'
    manager = downloader.DownloadManager()

    async def run_attempts():
        nonlocal attempt
        jobs = []
        for attempt in (1, 2):
            job = downloader.DownloadJob(str(attempt), 'fixture', 'model.gguf', 'https://example.test/model.gguf',
                                         dest, temp, 0)
            await manager._run(job)
            jobs.append(job)
        return jobs

    first, second = asyncio.run(run_attempts())
    assert first.status == 'error'
    assert second.status == 'done'
    assert dest.read_bytes() == payload
    assert [r for n, method, r in requested if n == 2 and method == 'GET'] == ['bytes=0-7', 'bytes=8-15']


@pytest.mark.parametrize('fault', ['ignored-range', 'wrong-range', 'short-body', 'long-body'])
def test_parallel_invalid_range_cannot_install_model(monkeypatch, tmp_path, fault):
    payload = b'abcdefghijklmnop'

    def respond(req):
        if req.method == 'HEAD':
            return httpx.Response(200, headers={'content-length': str(len(payload)), 'accept-ranges': 'bytes'})
        start, end = map(int, req.headers['range'].removeprefix('bytes=').split('-'))
        data = payload[start:end + 1]
        if start == 0:
            if fault == 'ignored-range':
                return httpx.Response(200, content=payload)
            if fault == 'wrong-range':
                return httpx.Response(206, content=data, headers={'content-range': 'bytes 1-8/16'})
            if fault == 'short-body':
                data = data[:-1]
            if fault == 'long-body':
                data += b'!'
        return httpx.Response(206, content=data, headers={'content-range': f'bytes {start}-{end}/{len(payload)}'})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(downloader.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(hf, 'get_token', lambda: None)
    monkeypatch.setattr(downloader.db, 'record_download', lambda **kw: None)
    monkeypatch.setattr(downloader, 'PARALLEL_MIN_SIZE', 1)
    monkeypatch.setattr(downloader, 'PARALLEL_CHUNKS', 2)
    job = downloader.DownloadJob('test', 'fixture', 'model.gguf', 'https://example.test/model.gguf',
                                 tmp_path / 'model.gguf', tmp_path / 'model.gguf.download', 0)
    asyncio.run(downloader.DownloadManager()._run(job))
    assert job.status == 'error'
    assert not job.dest_path.exists()


def test_socket_experiment_requires_token():
    compose = (Path(__file__).resolve().parents[3] / 'experiments/model-loader/compose.daserver.yaml').read_text()
    manager = compose.split('  model-loader:')[1]
    assert '/var/run/docker.sock:/var/run/docker.sock' in manager
    assert 'MODEL_LOADER_TOKEN: ${MODEL_LOADER_TOKEN:?' in manager
    assert 'context: ../../services/model-manager' in manager
    assert 'model-loader-test:e11a6ec' not in manager

@pytest.mark.parametrize('url', [
    'https://huggingface.co.evil.test/model', 'https://evil.test/huggingface.co/model',
    'https://evil.test/model?source=huggingface.co', 'http://huggingface.co/model',
])
def test_untrusted_download_never_sends_token(monkeypatch, tmp_path, url):
    captured = []
    def respond(req):
        captured.append(req)
        return httpx.Response(200, content=b'fixture' if req.method == 'GET' else b'')
    real_client = httpx.AsyncClient
    monkeypatch.setattr(downloader.httpx, 'AsyncClient', lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(hf, 'get_token', lambda: 'synthetic-token')
    job = downloader.DownloadJob('test', 'fixture', 'model', url, tmp_path/'model', tmp_path/'part', 0)
    asyncio.run(downloader.DownloadManager()._stream(job))
    assert [r.method for r in captured] == ['HEAD', 'GET']
    assert all('authorization' not in r.headers for r in captured)
