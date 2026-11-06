"""Native replay against the real recovery implementation, synthetic corpora only."""
import json
import sqlite3

import pytest
import agent.app as appmod
from agent import corpus as fmt, tenant_writer
from tests.test_rust_forward import world, DAILY  # noqa: F401
from tests.test_rust_forward_d5 import (writers, _writer_registry, needs_binary, needs_server, needs_cw,
    request, snapshot, restore, storage, Recorder)  # noqa: F401


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
