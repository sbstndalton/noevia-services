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
            data = payload[int(start):int(end)+1 if end else None]
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
    asyncio.run(downloader.DownloadManager()._stream(job))
    assert job.temp_path.read_bytes() == payload
    assert any(r.method == 'GET' and 'range' in r.headers for r in captured)
    for req in captured:
        assert req.headers.get('authorization') == ('Bearer synthetic-token' if req.url.host == 'huggingface.co' else None)


def test_socket_experiment_requires_token():
    compose = (Path(__file__).resolve().parents[3] / 'experiments/model-loader/compose.daserver.yaml').read_text()
    manager = compose.split('  model-loader:')[1]
    assert '/var/run/docker.sock:/var/run/docker.sock' in manager
    assert 'MODEL_LOADER_TOKEN: ${MODEL_LOADER_TOKEN:?' in manager

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
