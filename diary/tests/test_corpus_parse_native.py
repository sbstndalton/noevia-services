"""Synthetic native parser transport and index durability checks."""
import json
import os
from pathlib import Path
import struct
import sys
from datetime import date

import pytest

from agent import corpus_parse as cp
from agent.config import Config
from agent.corpus_store import CorpusStore
from agent.journal import Journal
from agent.retrieval import Retriever
from tests.test_journal_store import FakeWebDAV
from tests.test_retrieval_context import StubLLM

BIN = os.environ.get('DIARY_CORPUS_PARSE_BIN')
SAMPLE = ('## Saturday, August 29, 2026\n\n### 09:15 — Morning\n\n'
          '**Me:** Synthetic note\n\n**Assistant:** Synthetic reply\n\n'
          '<!-- xid:11111111-1111-1111-1111-111111111111 -->\n')
EMPTY = '# No Diary day heading\n'


def fake(tmp_path, script):
    path = tmp_path / 'fake-parser'
    path.write_text(f'#!{sys.executable}\n' + script)
    path.chmod(0o755)
    return str(path)


def frame(payload):
    raw = json.dumps(payload, ensure_ascii=False).encode()
    return b'DCR1' + struct.pack('>Q', len(raw)) + raw + b'END1'


def parsed_frame():
    from agent import corpus as fmt
    days = fmt.parse_diary(SAMPLE)
    day = days[0]
    sub = day.subsections[0]
    rows = cp._chunks(days)
    return frame({'version': 1, 'days': [{
        'header': day.header, 'date': day.date.isoformat(), 'subsections': [{
            'header': sub.header, 'time': sub.time, 'topic': sub.topic,
            'exchanges': [vars(e) for e in sub.exchanges]}]}],
        'chunks': [{'day': a, 'header': b, 'text': c, 'content_hash': d}
                   for a, b, c, d in rows]})


def emitting(tmp_path, payload, exit_code=0, sleep=0):
    return fake(tmp_path, 'import sys, time\n'
                'sys.stdin.buffer.read()\n'
                f'sys.stdout.buffer.write(bytes.fromhex({payload.hex()!r}))\n'
                'sys.stdout.buffer.flush()\n'
                f'time.sleep({sleep})\n'
                f'sys.exit({exit_code})\n')


def use(monkeypatch, path):
    monkeypatch.setenv('DIARY_CORPUS_PARSE_IMPL', 'rust')
    monkeypatch.setattr(cp, 'DEFAULT_BIN', str(path))


@pytest.mark.skipif(not BIN, reason='paired Rust CLI path required')
def test_actual_cli_days_chunks_and_both_callers(tmp_path, monkeypatch):
    use(monkeypatch, BIN)
    text = SAMPLE + SAMPLE.replace('Morning', 'Evening').replace('Synthetic note', 'Second note')
    python = cp.parse(text)
    monkeypatch.delenv('DIARY_CORPUS_PARSE_IMPL')
    reference = cp.parse(text)
    assert python == reference
    use(monkeypatch, BIN)
    retriever = Retriever(tmp_path / 'idx.db', StubLLM())
    assert retriever._parse_chunks(text) == reference.chunks
    store = CorpusStore(Config({'corpus': {'webdav': {'remote_root': 'Diary'},
                                             'monthly_prefix': '', 'index_file': 'INDEX.md'}}),
                        FakeWebDAV(), Journal(tmp_path / 'journal.db'))
    store.dav.files[store.month_path(date(2026, 8, 29))] = (SAMPLE, '"1"')
    assert '<!-- xid:' not in store.get_day_text(date(2026, 8, 29))
    assert '<!-- xid:' in store.get_day_text(date(2026, 8, 29), include_markers=True)
    assert store.get_day_text(date(2026, 8, 29), max_chars=50).startswith('(earlier exchanges today truncated)')


@pytest.mark.skipif(not BIN, reason='paired Rust CLI path required')
def test_actual_cli_455_synthetic_python_differential(monkeypatch):
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'diary-corpus-parse.v1.json').read_text())
    assert fixture['source_commit'] == '6c1eb5891c75b6fb9f04c4484ba39da80a5b4b9e'
    assert len(fixture['fixtures']) == 455
    use(monkeypatch, BIN)
    for case in fixture['fixtures']:
        actual = cp.parse(case['input'])
        days = [{'header': d.header, 'date': d.date.isoformat() if d.date else None,
                 'subsections': [{'header': sub.header, 'time': sub.time, 'topic': sub.topic,
                                  'exchanges': [vars(ex) for ex in sub.exchanges]}
                                 for sub in d.subsections]}
                for d in actual.days]
        assert days == case['days'], case['name']
        assert [list(row) for row in actual.chunks] == case['chunks'], case['name']


@pytest.mark.parametrize('damage', ['truncated', 'magic', 'length', 'version', 'schema', 'hash', 'extra', 'nonzero', 'overflow', 'timeout'])
def test_fault_never_creates_plan_or_deletes_index(tmp_path, monkeypatch, damage):
    retriever = Retriever(tmp_path / 'idx.db', StubLLM())
    retriever.reindex_file('2026-08.md', SAMPLE)
    with retriever._conn:
        before = retriever._conn.execute(
            'SELECT id,day,file,header,body,content_hash FROM chunks ORDER BY id').fetchall()
        vector_tables = [r[0] for r in retriever._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_%'")]
        vectors = {name: retriever._conn.execute(f'SELECT * FROM "{name}"').fetchall()
                   for name in vector_tables}
    good = parsed_frame()
    if damage in ('version', 'schema', 'hash'):
        obj = json.loads(good[12:-4])
        if damage == 'version':
            obj['version'] = 2
        elif damage == 'schema':
            del obj['days'][0]['subsections'][0]['topic']
        else:
            obj['chunks'][0]['content_hash'] = '0' * 64
        payload = frame(obj)
    elif damage == 'truncated':
        payload = good[:-4]
    elif damage == 'magic':
        payload = b'BAD!' + good[4:]
    elif damage == 'length':
        payload = good[:4] + struct.pack('>Q', len(good)) + good[12:]
    elif damage == 'extra':
        payload = good + b'x'
    else:
        payload = good
    if damage == 'overflow':
        monkeypatch.setattr(cp, 'MAX_OUTPUT', len(good) - 1)
    if damage == 'timeout':
        monkeypatch.setattr(cp, 'TIMEOUT_S', 0.25)
    use(monkeypatch, emitting(tmp_path, payload, exit_code=3 if damage == 'nonzero' else 0,
                               sleep=2 if damage == 'timeout' else 0))
    with pytest.raises(cp.ParseUnavailable):
        retriever.prepare_reindex('2026-08.md', SAMPLE)
    after = retriever._conn.execute(
        'SELECT id,day,file,header,body,content_hash FROM chunks ORDER BY id').fetchall()
    assert [tuple(x) for x in after] == [tuple(x) for x in before]
    assert {name: retriever._conn.execute(f'SELECT * FROM "{name}"').fetchall()
            for name in vectors} == vectors
    if damage == 'overflow':
        monkeypatch.setattr(cp, 'MAX_OUTPUT', 512 * 1024 * 1024)
    use(monkeypatch, emitting(tmp_path, good))
    assert retriever.prepare_reindex('2026-08.md', SAMPLE).rows == cp._chunks(cp.parse(SAMPLE).days)
    assert retriever.reindex_file('2026-08.md', SAMPLE) == 0
    assert retriever.stats()['chunks'] == 1
    use(monkeypatch, BIN if BIN else emitting(tmp_path, frame({'version': 1, 'days': [], 'chunks': []})))
    assert retriever.reindex_file('2026-08.md', EMPTY) == 0
    assert retriever.stats()['chunks'] == 0


def test_capacity_missing_binary_and_read_failure_not_empty(tmp_path, monkeypatch):
    use(monkeypatch, tmp_path / 'missing')
    with pytest.raises(cp.ParseCapacity):
        monkeypatch.setattr(cp, 'MAX_INPUT', 1)
        cp.parse(SAMPLE)
    monkeypatch.setattr(cp, 'MAX_INPUT', 256 * 1024 * 1024)
    with pytest.raises(cp.ParseUnavailable):
        cp.parse(SAMPLE)
    use(monkeypatch, emitting(tmp_path, parsed_frame()[:-4]))
    store = CorpusStore(Config({'corpus': {'webdav': {'remote_root': 'Diary'},
                                             'monthly_prefix': '', 'index_file': 'INDEX.md'}}),
                        FakeWebDAV(), Journal(tmp_path / 'journal.db'))
    store.dav.files[store.month_path(date(2026, 8, 29))] = (SAMPLE, '"1"')
    with pytest.raises(cp.ParseUnavailable):
        store.get_day_text(date(2026, 8, 29))
    store.dav.files.clear()
    with pytest.raises(cp.ParseUnavailable):
        store.get_day_text(date(2026, 8, 29))



def test_child_capacity_and_stderr_overflow_are_typed_and_safe(tmp_path, monkeypatch, caplog):
    capacity = fake(tmp_path, 'import sys\n'
                    'sys.stdin.buffer.read()\n'
                    'sys.stderr.write("capacity\\n")\n'
                    'sys.exit(3)\n')
    use(monkeypatch, capacity)
    with pytest.raises(cp.ParseCapacity):
        cp.parse(SAMPLE)
    assert 'Synthetic note' not in caplog.text
    flooding = fake(tmp_path, 'import sys\n'
                    'sys.stdin.buffer.read()\n'
                    'sys.stderr.write("x" * 1000)\n'
                    'sys.exit(3)\n')
    use(monkeypatch, flooding)
    with pytest.raises(cp.ParseUnavailable):
        cp.parse(SAMPLE)
    assert 'x' * 100 not in caplog.text
