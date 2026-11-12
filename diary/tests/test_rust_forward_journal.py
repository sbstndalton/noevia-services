"""Native replay against the real recovery implementation, synthetic corpora only."""
import json
import sqlite3

import pytest
import agent.app as appmod
from agent import corpus as fmt, tenant_writer
from tests.test_rust_forward import world, DAILY  # noqa: F401
from tests.test_rust_forward import MONTHLY
from tests.test_rust_forward_d5 import (writers, _writer_registry, needs_binary, needs_server, needs_cw,
    request, snapshot, restore, storage, differential, Recorder)  # noqa: F401


@needs_binary
@needs_server
@needs_cw
@pytest.mark.parametrize('scenario', ['clean', 'pending', 'partial', 'quarantine', 'index', 'limit', 'missing'])
def test_real_replay_parity(writers, scenario):
    client, start, state = writers
    modes = start()
    tenant = state / 'users' / DAILY
    payload = {'day':'2030-01-06', 'xid':'abcdef12', 'sub_header':'12:00 | Synthetic',
               'body':fmt.render_exchange('Synthetic note', '', 'abcdef12')}
    with sqlite3.connect(tenant / 'managed-index.db') as db:
        if scenario != 'clean':
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       ('replay', '2030-01-10', 'exchange', json.dumps(payload)))
        if scenario == 'pending':
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       ('edit', '2000-01-01', 'exchange_edit', json.dumps({'xid':'abcdef12','new_me':'Synthetic newest','new_claude':''})))
        if scenario == 'quarantine':
            db.execute("UPDATE journal SET kind='broken'")
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       ('dependent', '2030', 'exchange_edit', json.dumps({'xid':'abcdef12'})))
        if scenario == 'index':
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       ('month', '2000', 'index_month', json.dumps({'label':'January 2030','month':'2030-01.md'})))
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       ('update', '1999', 'index_update', json.dumps({'today':'2030-01-06','open_questions':[{'action':'add','text':'Synthetic question?'}]})))
        if scenario == 'missing':
            db.execute("UPDATE journal SET payload='{}'")
        if scenario == 'limit':
            for n in range(105):
                db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                           (f'month-{n}', '2000', 'index_month', json.dumps({'label':f'Synthetic {n}','month':f'{n}.md'})))
    if scenario == 'partial':
        # Persisted corpus write with its completion record still pending.
        st = next(v for k, v in appmod._tenant_states.items() if k.startswith(DAILY + ":"))
        with tenant_writer.hold(DAILY):
            st.store._apply_entry(st.journal.unapplied()[0])
    with sqlite3.connect(tenant / 'managed-diary.db') as db:
        db.execute("INSERT OR IGNORE INTO workspace_index_outbox VALUES ('outbox.md')")
    before = snapshot(state, DAILY)
    modes.python()
    st = next(v for k, v in appmod._tenant_states.items() if k.startswith(DAILY + ":"))
    st.recovered = False
    appmod._recover_state(st)
    expected = storage(state, DAILY)
    restore(state, DAILY, before)
    # Python replay must never be invoked on this native request.
    modes.rust()
    original = st.store.apply_pending
    st.store.apply_pending = lambda *a, **k: pytest.fail('Python replay used')
    st.recovered = False
    Recorder.served = []
    try:
        response = request(client, DAILY, 'POST', '/api/directory', {'path':'Replay result'})
    finally:
        st.store.apply_pending = original
    assert response.status_code == 200, response.text
    assert Recorder.served == [True]
    actual = storage(state, DAILY)
    # Directory creation is the trigger; its one generation bump is separate from replay.
    assert actual[0] == expected[0]
    assert actual[3] == expected[3]
    assert int(actual[2]['generation']) == int(expected[2]['generation']) + 1
    with sqlite3.connect(tenant / 'managed-diary.db') as db:
        assert db.execute('SELECT COUNT(*) FROM workspace_index_outbox').fetchone()[0] == 0
    with sqlite3.connect(tenant / 'managed-index.db') as db:
        assert ('outbox.md',) in db.execute('SELECT document FROM dirty_documents').fetchall()

@needs_binary
@needs_server
@needs_cw
def test_crash_during_replay_then_native_resume(writers):
    import concurrent.futures
    import time
    client, start, state = writers
    modes = start()
    tenant = state / 'users' / DAILY
    with sqlite3.connect(tenant / 'managed-index.db') as db:
        for n in range(80):
            xid = f'{n:08x}'
            p = {'day':'2030-01-06','xid':xid,'sub_header':'12:00 | Synthetic',
                 'body':fmt.render_exchange(f'Synthetic {n}', '', xid)}
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       (f'replay-{n}', '2030', 'exchange', json.dumps(p)))
    # Kill after observing a real corpus/replay commit, while the pass is still running.
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pending = pool.submit(request, client, DAILY, 'POST', '/api/directory', {'path':'After replay'})
        deadline = time.monotonic() + 10
        saw_partial = False
        while time.monotonic() < deadline and not pending.done():
            with sqlite3.connect(tenant / 'managed-index.db') as db:
                count = db.execute("SELECT COUNT(*) FROM journal WHERE id LIKE 'replay-%' AND applied=1").fetchone()[0]
            if 0 < count < 80:
                modes.server._proc.kill()
                saw_partial = True
                break
            time.sleep(0.001)
        response = pending.result(timeout=20)
    assert saw_partial, 'kill must land during replay'
    assert response.status_code == 503
    modes.server.stop()  # reap and settle the exclusion barrier
    modes = start()
    st = next(v for k, v in appmod._tenant_states.items() if k.startswith(DAILY + ":"))
    st.store.apply_pending = lambda *a, **k: pytest.fail('resume used Python replay')
    response = request(client, DAILY, 'POST', '/api/directory', {'path':'After replay'})
    assert response.status_code == 200, response.text
    with sqlite3.connect(tenant / 'managed-index.db') as db:
        assert db.execute("SELECT COUNT(*) FROM journal WHERE id LIKE 'replay-%' AND applied=0").fetchone()[0] == 0
    with sqlite3.connect(tenant / 'managed-diary.db') as db:
        text = '\n'.join(row[0].decode() for row in db.execute("SELECT data FROM files WHERE path LIKE 'Entries/%'"))
    for n in range(80):
        assert text.count(f'<!-- xid:{n:08x} -->') == 1


@needs_binary
@needs_server
@needs_cw
def test_python_writer_waits_until_native_replay_settles(writers):
    import concurrent.futures
    import threading
    import time
    client, start, state = writers
    start()
    tenant = state / 'users' / DAILY
    with sqlite3.connect(tenant / 'managed-index.db') as db:
        db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                   ('blocked','2030','exchange',json.dumps({'day':'2030-01-06','xid':'abcdef12','sub_header':'12:00 | Synthetic','body':fmt.render_exchange('Synthetic', '', 'abcdef12')})))
    entered = threading.Event()
    def peer():
        with tenant_writer.hold(DAILY):
            entered.set()
    # Hold the corpus transaction so native replay cannot settle its first write.
    with sqlite3.connect(tenant / 'managed-diary.db') as db, concurrent.futures.ThreadPoolExecutor() as pool:
        db.execute('BEGIN IMMEDIATE')
        pending = pool.submit(request, client, DAILY, 'POST', '/api/directory', {'path':'After replay'})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with tenant_writer._cond:
                slot = tenant_writer._slots.get(DAILY)
                if slot is not None and slot.owner is not None:
                    break
            time.sleep(0.001)
        other = pool.submit(peer)
        assert not entered.wait(0.15)
        assert not pending.done()
        db.commit()
        assert pending.result(timeout=20).status_code == 200
        other.result(timeout=20)
    assert entered.is_set()


@needs_binary
@needs_server
@needs_cw
@pytest.mark.parametrize('case', [
    'nan', 'infinity', 'surrogate', 'known-nan', 'known-surrogate', 'malformed',
    'month-0', 'month-19', 'short-0', 'short-19',
    'year0-0', 'year0-19', 'day0',
])
@pytest.mark.parametrize('runtime', ['reference', 'forwarded'])
def test_noncanonical_replay_uses_reference_before_effects(writers, case, runtime):
    import sys
    client, start, state = writers
    modes = start()
    client._transport.raise_server_exceptions = False
    tenant = state / 'users' / DAILY
    st = next(v for k, v in appmod._tenant_states.items() if k.startswith(DAILY + ':'))
    kind = 'unknown'
    payload = {'nan':'{"extra":NaN}', 'infinity':'{"extra":Infinity}',
               'surrogate':'{"extra":"\\ud800"}', 'malformed':'{"extra":'}.get(case)
    attempts = 0
    if case.startswith('known-'):
        kind = 'index_month'
        extension = 'NaN' if case == 'known-nan' else '"\\ud800"'
        payload = '{"label":"Synthetic","month":"2030-01.md","extra":' + extension + '}'
    if case.startswith(('month-', 'short-', 'year0-')):
        kind = 'exchange_edit'
        attempts = int(case.split('-')[1])
        hint = {'month': '2030-13', 'short': 'x', 'year0': '0000-01'}[case.split('-')[0]]
        payload = json.dumps({'xid':'abcdef12','new_me':'Synthetic bad hint', 'new_claude':'',
                              'month':hint})
    if case == 'day0':
        # Legacy/manual row: fromisoformat refuses year 0, so the reference quarantines it at once.
        kind = 'exchange'
        payload = json.dumps({'day':'0000-01-01','xid':'abcdef12','sub_header':'12:00 | Synthetic',
                              'body':fmt.render_exchange('Synthetic year zero', '', 'abcdef12')})
    with sqlite3.connect(tenant / 'managed-index.db') as db:
        db.execute('INSERT INTO journal(id,created_at,kind,payload,attempts) VALUES (?,?,?,?,?)',
                   ('compat', '2030', kind, payload, attempts))
        if kind == 'exchange_edit':
            db.execute('INSERT INTO journal(id,created_at,kind,payload) VALUES (?,?,?,?)',
                       ('dependent','2030','exchange_edit',json.dumps({'xid':'abcdef12',
                        'month':'2030-01','new_me':'Synthetic valid hint','new_claude':''})))
    before = snapshot(state, DAILY)
    body = {'path':'Compatibility result'}
    modes.python()
    st.recovered = False
    expected_reply = request(client, DAILY, 'POST', '/api/directory', body)
    expected = storage(state, DAILY)
    print('reference diagnostic provenance:', sys.version, case, expected[3])
    rows = {row[0]: row[1:] for row in sqlite3.connect(tenant / 'managed-index.db').execute(
        "SELECT id,applied,attempts,last_error FROM journal WHERE id IN ('compat','dependent')")}
    if kind == 'exchange_edit':
        assert rows['compat'][1] == attempts + 1
        assert rows['compat'][0] == int(attempts == 19)
        assert rows['dependent'][:2] == ((1, 1) if attempts == 19 else (0, 0))
    elif case == 'malformed':
        assert rows['compat'] == (0, 0, None)
    else:
        assert rows['compat'][0] == 1
    if runtime == 'reference':
        return
    restore(state, DAILY, before)
    st.recovered = False
    modes.rust()
    Recorder.served = []
    reply = request(client, DAILY, 'POST', '/api/directory', body)
    assert Recorder.served == [False], 'must decline to real Python before any recovery effect'
    assert (reply.status_code, reply.content) == (expected_reply.status_code, expected_reply.content)
    assert storage(state, DAILY) == expected
    # Recoverable extensions retire and permit the next native request. An invalid
    # month at attempts 0 or malformed JSON must continue using reference recovery.
    Recorder.served = []
    follow = request(client, DAILY, 'POST', '/api/directory', {'path':'Compatibility followup'})
    pending = case == 'malformed' or case in ('month-0', 'short-0', 'year0-0')
    assert Recorder.served == [not pending]
    assert follow.status_code == (500 if case == 'malformed' else 200)


@needs_binary
@needs_server
@needs_cw
def test_year_zero_edit_hint_is_decided_by_the_reference_and_adjacent_years_stay_native(writers):
    """Y0-2: month "0000-01" matches MONTH_ID_RE but date(0, 1, 1) raises, so the reference answers
    502 'edit failed'; the native edit must decline to it before any effect, on both layouts.
    Years 1 and 9999 are valid dates and stay native (404: no such document)."""
    client, start, state = writers
    modes = start()
    client._transport.raise_server_exceptions = False
    xid = 'ffffffff-0000-4000-8000-000000000000'
    cases = []
    for label, user in (('daily', DAILY), ('monthly', MONTHLY)):
        cases += [
            (f'edit year-zero month, {label} (Python decides)', user, 'POST', '/api/entries/edit',
             {'xid': xid, 'me': 'x', 'month': '0000-01'}, 502, False),
            (f'edit year-zero month 12, {label} (Python decides)', user, 'POST', '/api/entries/edit',
             {'xid': xid, 'me': 'x', 'month': '0000-12'}, 502, False),
            (f'edit month 0000-13 stays a 400, {label}', user, 'POST', '/api/entries/edit',
             {'xid': xid, 'me': 'x', 'month': '0000-13'}, 400, True),
        ]
    cases += [
        ('edit year 1 stays native', DAILY, 'POST', '/api/entries/edit',
         {'xid': xid, 'me': 'x', 'month': '0001-01'}, 404, True),
        ('edit year 9999 stays native', DAILY, 'POST', '/api/entries/edit',
         {'xid': xid, 'me': 'x', 'month': '9999-12'}, 404, True),
    ]
    differential(client, modes, state, cases)
