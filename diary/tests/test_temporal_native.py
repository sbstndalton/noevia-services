"""Synthetic bounded native temporal transport and context fail-closed checks."""
from datetime import date
from types import SimpleNamespace
import json
import os
from pathlib import Path
import struct
import sys

import pytest

from agent import entity_focus, temporal_native as native
from agent.config import Config
from agent.context import ContextAssembler
from tests.test_retrieval_context import FakeStore

REAL_BIN = os.environ.get('DIARY_TEMPORAL_TEST_BIN')
FIXTURES = Path(__file__).parent / 'fixtures'
ANCHOR = (date(2030, 1, 20), True)
ROWS = [{'day': '2030-01-20', 'header': 'a', 'text': 'a', 'score': .9},
        {'day': '2030-01-21', 'header': 'b', 'text': 'b', 'score': .8}]


def frame(obj):
    body = json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return b'DTR1' + struct.pack('>Q', len(body)) + body + b'END1'


def fake(tmp_path, source):
    path = tmp_path / 'synthetic-temporal-child'
    path.write_text(f'#!{sys.executable}\n' + source)
    path.chmod(0o755)
    return str(path)


def emitting(tmp_path, payload, code=0, delay=0, stderr=b''):
    return fake(tmp_path, 'import sys,time\n'
                'sys.stdin.buffer.read()\n'
                f'sys.stdout.buffer.write({payload!r})\n'
                'sys.stdout.buffer.flush()\n'
                f'sys.stderr.buffer.write({stderr!r})\n'
                f'time.sleep({delay})\n'
                f'sys.exit({code})\n')


def use(monkeypatch, path):
    monkeypatch.setenv('DIARY_TEMPORAL_FOCUS_IMPL', 'rust')
    monkeypatch.setattr(native, 'DEFAULT_BIN', str(path))


@pytest.mark.skipif(not REAL_BIN, reason='paired Rust CLI binary required')
def test_actual_cli_python312_anchor_and_cutoff_fixtures(monkeypatch):
    use(monkeypatch, REAL_BIN)
    queries = json.loads((FIXTURES / 'diary-temporal-anchor-python312.json').read_text())
    for case in queries:
        actual = native.anchor(case['query'])
        expected = case['anchor']
        wanted = (date.fromisoformat(expected['day']), expected['inclusive']) if expected else None
        assert actual == wanted
    case = json.loads((FIXTURES / 'diary-temporal-cutoff-python312.json').read_text())
    anchor = native.anchor(case['query'])
    assert anchor == entity_focus.time_anchor(case['query'])
    kept, removed = native.cutoff(anchor, case['rows'])
    assert [case['rows'][rank - 1]['id'] for rank in kept] == case['kept_ids']
    assert [(rank, case['rows'][rank - 1]['id'], 'after_time_anchor') for rank in removed] == [
        (item['rank'], item['id'], item['reason']) for item in case['removed']]
    assert native.anchor('No dated question.') is None
    assert native.cutoff(None, ROWS) == ([1, 2], [])


@pytest.mark.parametrize('failure', ['constructor', 'spawn', 'blocking', 'register',
                                      'cancel', 'wait', 'selector_close'])
def test_complete_child_lifetime_cleanup(tmp_path, monkeypatch, caplog, failure):
    payload = frame({'version': 1, 'op': 'anchor', 'anchor': None})
    use(monkeypatch, emitting(tmp_path, payload))
    processes, selectors_seen = [], []
    popen = native.subprocess.Popen
    selector_type = native.selectors.DefaultSelector
    set_blocking = native.os.set_blocking

    class TrackedSelector(selector_type):
        def __init__(self):
            if failure == 'constructor':
                raise RuntimeError('private constructor detail')
            super().__init__()
            self.closed_seen = False
            selectors_seen.append(self)

        def register(self, *args, **kwargs):
            if failure == 'register':
                raise RuntimeError('private register detail')
            return super().register(*args, **kwargs)

        def select(self, *args, **kwargs):
            if failure == 'cancel':
                raise KeyboardInterrupt('private cancellation detail')
            return super().select(*args, **kwargs)

        def close(self):
            self.closed_seen = True
            result = super().close()
            if failure == 'selector_close':
                raise RuntimeError('private selector close detail')
            return result

    def tracked_spawn(*args, **kwargs):
        if failure == 'spawn':
            raise OSError('private spawn detail')
        child = popen(*args, **kwargs)
        processes.append(child)
        if failure == 'wait':
            wait = child.wait

            def completed_wait(*args, **kwargs):
                wait(*args, **kwargs)
                raise RuntimeError('private wait detail')

            child.wait = completed_wait
        return child

    def tracked_blocking(*args, **kwargs):
        if failure == 'blocking':
            raise RuntimeError('private setup detail')
        return set_blocking(*args, **kwargs)

    monkeypatch.setattr(native.selectors, 'DefaultSelector', TrackedSelector)
    monkeypatch.setattr(native.subprocess, 'Popen', tracked_spawn)
    monkeypatch.setattr(native.os, 'set_blocking', tracked_blocking)
    expected = KeyboardInterrupt if failure == 'cancel' else native.TemporalUnavailable
    with pytest.raises(expected) as raised:
        native.anchor('No dated question.')
    assert type(raised.value) is expected
    assert len(processes) == (0 if failure in ('constructor', 'spawn') else 1)
    assert len(selectors_seen) == (0 if failure == 'constructor' else 1)
    for child in processes:
        assert child.returncode is not None
        assert all(stream.closed for stream in (child.stdin, child.stdout, child.stderr))
    assert all(selector.closed_seen for selector in selectors_seen)
    assert 'private' not in caplog.text
    if failure != 'cancel':
        assert 'private' not in str(raised.value)


@pytest.mark.skipif(not REAL_BIN, reason='paired Rust CLI binary required')
def test_actual_cli_corrected_date_and_unicode_edges(monkeypatch):
    use(monkeypatch, REAL_BIN)
    for query in [
            'Dr\u0345. Before January 20, 2030, what?',
            'Sept\u0345. Before January 20, 2030, what?',
            'Dr. Before January 20, 2030, what?',
            'As of January \U00010d42\U00010d40, 2030, what happened?']:
        assert native.anchor(query) == entity_focus.time_anchor(query)
    rows = [{'day': day} for day in ['2030-1-20', '+10000-01-01', '9999-W52-7',
                                    '2030W011', '2030-W01-1', '0000-01-01',
                                    '2030-01-20', '2024-02-29', ' 2030-01-20']]
    kept, removed = [], []
    for rank, row in enumerate(rows, 1):
        try:
            day = date.fromisoformat(row['day'])
        except ValueError:
            day = None
        (removed if day is not None and day > ANCHOR[0] else kept).append(rank)
    assert native.cutoff(ANCHOR, rows) == (kept, removed)


@pytest.mark.parametrize('damage', ['truncated', 'magic', 'length', 'version', 'extra', 'nonzero',
                                     'wrong_anchor', 'wrong_rank', 'duplicate_rank', 'duplicate_key', 'stderr',
                                     'overflow', 'timeout'])
def test_native_faults_fail_closed(tmp_path, monkeypatch, damage):
    good = frame({'version': 1, 'op': 'cutoff', 'anchor': {'day': '2030-01-20', 'inclusive': True},
                  'kept_ranks': [1], 'removed': [{'rank': 2, 'reason': 'after_time_anchor'}]})
    if damage in ('version', 'wrong_anchor', 'wrong_rank', 'duplicate_rank'):
        obj = json.loads(good[12:-4])
        if damage == 'version':
            obj['version'] = 2
        elif damage == 'wrong_anchor':
            obj['anchor']['day'] = '2030-01-19'
        elif damage == 'wrong_rank':
            obj['kept_ranks'] = [3]
        else:
            obj['kept_ranks'] = [1, 1]
        payload = frame(obj)
    elif damage == 'duplicate_key':
        payload = frame({'version': 1, 'op': 'cutoff', 'anchor': {'day': '2030-01-20', 'inclusive': True},
                         'kept_ranks': [1], 'removed': [{'rank': 2, 'reason': 'after_time_anchor'}]})
        payload = payload.replace(b'"op":"cutoff"', b'"op":"anchor","op":"cutoff"')
        payload = payload[:4] + struct.pack('>Q', len(payload) - 16) + payload[12:]
    elif damage == 'truncated':
        payload = good[:-4]
    elif damage == 'magic':
        payload = b'BAD!' + good[4:]
    elif damage == 'length':
        payload = good[:4] + struct.pack('>Q', 999) + good[12:]
    elif damage == 'extra':
        payload = good + b'x'
    else:
        payload = good
    if damage == 'overflow':
        monkeypatch.setattr(native, 'MAX_RESPONSE', 20)
    if damage == 'timeout':
        monkeypatch.setattr(native, 'TIMEOUT_S', .2)
    use(monkeypatch, emitting(tmp_path, payload, code=2 if damage == 'nonzero' else 0,
                              delay=2 if damage == 'timeout' else 0,
                              stderr=b'private-source' if damage == 'stderr' else b''))
    with pytest.raises(native.TemporalUnavailable):
        native.cutoff(ANCHOR, ROWS)
    monkeypatch.setattr(native, 'MAX_RESPONSE', 1024 * 1024)
    monkeypatch.setattr(native, 'TIMEOUT_S', 5)


def test_capacity_missing_binary_and_no_anchor_success(tmp_path, monkeypatch, caplog):
    use(monkeypatch, tmp_path / 'missing')
    with pytest.raises(native.TemporalCapacity):
        native.anchor('x' * (native.MAX_QUERY + 1))
    with pytest.raises(native.TemporalCapacity):
        native.cutoff(ANCHOR, [{}] * (native.MAX_CANDIDATES + 1))
    with pytest.raises(native.TemporalCapacity):
        native.cutoff(ANCHOR, [{'day': 'x' * 65}])
    with pytest.raises(native.TemporalUnavailable):
        native.anchor('Before 2030-01-20, what happened?')
    use(monkeypatch, emitting(tmp_path, frame({'version': 1, 'op': 'anchor', 'anchor': None})))
    assert native.anchor('No anchor here.') is None
    use(monkeypatch, emitting(tmp_path, frame({'version': 1, 'op': 'cutoff', 'anchor': None,
                                             'kept_ranks': [1, 2], 'removed': []})))
    assert native.cutoff(None, ROWS) == ([1, 2], [])
    assert 'private-source' not in caplog.text


class Retriever:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, query, *, top_k, min_score, **kwargs):
        self.calls.append(top_k)
        return self.rows


def test_one_anchor_widens_pool_then_same_anchor_filters(monkeypatch):
    monkeypatch.setenv('DIARY_TEMPORAL_FOCUS_IMPL', 'rust')
    observed = []
    monkeypatch.setattr(native, 'anchor', lambda query: observed.append(('anchor', query)) or ANCHOR)
    monkeypatch.setattr(native, 'cutoff', lambda anchor, rows: observed.append(('cutoff', anchor, rows)) or ([1], [2]))
    monkeypatch.setattr(entity_focus, 'time_anchor', lambda _: pytest.fail('Python anchor reparsed'))
    retriever = Retriever(ROWS)
    cfg = Config({'retrieval': {'top_k': 2, 'min_score': 0, 'entity_focus': True}})
    assembler = ContextAssembler(FakeStore(), retriever, cfg)
    loaded, labels, notes = assembler._retrieve_focused('As of January 20, 2030, what happened?', [])
    assert retriever.calls == [8]
    assert observed[0][0] == 'anchor' and observed[1] == ('cutoff', ANCHOR, ROWS)
    assert [row['day'] for row in loaded] == ['2030-01-20']
    assert any('Time anchor:' in line for line in notes)


def test_native_failure_stops_context_before_search_or_model(monkeypatch):
    import agent.app as app
    monkeypatch.setenv('DIARY_TEMPORAL_FOCUS_IMPL', 'rust')
    retriever = Retriever(ROWS)
    cfg = Config({'retrieval': {'top_k': 2, 'min_score': 0, 'entity_focus': True}})
    assembler = ContextAssembler(FakeStore(), retriever, cfg)
    monkeypatch.setattr(native, 'anchor', lambda _: (_ for _ in ()).throw(native.TemporalUnavailable('unavailable')))
    with pytest.raises(native.TemporalUnavailable):
        assembler._retrieve_focused('Before 2030-01-20, what happened?', [])
    assert retriever.calls == []
    monkeypatch.setattr(native, 'anchor', lambda _: ANCHOR)
    monkeypatch.setattr(native, 'cutoff', lambda *_: (_ for _ in ()).throw(native.TemporalUnavailable('unavailable')))
    with pytest.raises(native.TemporalUnavailable):
        assembler._retrieve_focused('Before 2030-01-20, what happened?', [])
    assert retriever.calls == [8]
    calls = []
    class FakeState:
        store = SimpleNamespace(require_write_transform=lambda: None)
        assembler = type('FailingAssembler', (), {'build': lambda *a, **k: (_ for _ in ()).throw(native.TemporalUnavailable('unavailable'))})()
        llm_main = type('Model', (), {'chat': lambda *a, **k: calls.append('model')})()
    monkeypatch.setattr(app, '_reindex_dirty', lambda _: None)
    with pytest.raises(native.TemporalUnavailable):
        app._run_exchange(FakeState(), 'question', 'synthetic', background=False)
    assert calls == []


def test_native_context_failure_http_and_stream_are_unsaved(monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as app
    monkeypatch.setattr(app, 'check_auth', lambda _: True)
    monkeypatch.setattr(app, '_tenant_state', lambda *a, **k: SimpleNamespace(store=SimpleNamespace(require_write_transform=lambda: None)))
    monkeypatch.setattr(app, '_run_exchange',
                        lambda *a, **k: (_ for _ in ()).throw(native.TemporalUnavailable('private detail')))
    client = TestClient(app.app)
    body = {'messages': [{'role': 'user', 'content': 'Before 2030-01-20, what happened?'}]}
    response = client.post('/v1/chat/completions', json=body)
    assert response.status_code == 503
    assert response.json()['error']['code'] == 'context_unavailable'
    assert 'private detail' not in response.text
    with client.stream('POST', '/v1/chat/completions', json={**body, 'stream': True, 'diary_events': True}) as stream:
        events = '\n'.join(stream.iter_lines())
    assert 'contextUnavailable' in events and '"saved": false' in events
    assert 'private detail' not in events and '"type": "diary"' not in events


def test_local_exchange_native_failure_reports_unsaved_without_private_detail(monkeypatch):
    from fastapi.testclient import TestClient
    import agent.app as app
    monkeypatch.setattr(app, 'check_auth', lambda _: True)
    monkeypatch.setattr(app, '_tenant_state', lambda _: pytest.fail('local mode read remote storage'))
    monkeypatch.setattr(app, '_run_exchange',
                        lambda *a, **k: (_ for _ in ()).throw(native.TemporalUnavailable('private detail')))
    client = TestClient(app.app, raise_server_exceptions=False)
    request = {'files': {}, 'message': 'Before 2030-01-20, what happened?', 'history': []}
    headers = {'X-Cowork-User-ID': '11111111-1111-4111-8111-111111111111'}
    response = client.post('/api/local-exchange', headers=headers, json=request)
    assert response.status_code == 503
    assert response.json()['code'] == 'contextUnavailable' and response.json()['saved'] is False
    assert 'private detail' not in response.text
    stream = client.post('/api/local-exchange', headers=headers, json={**request, 'stream': True})
    assert stream.status_code == 200
    assert 'contextUnavailable' in stream.text and '"saved": false' in stream.text
    assert 'private detail' not in stream.text and '"type": "diary"' not in stream.text
