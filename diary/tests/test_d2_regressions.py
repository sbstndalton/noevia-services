"""Native adapter/replay and broker-death regressions, synthetic data only."""
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import select
import signal
import sys
import time
from types import SimpleNamespace

import pytest
from agent import backup_native, corpus, index_transform_native, leaf_native
from agent.corpus_store import CorpusStore, is_transient_failure
from agent.journal import Journal
from agent.local_storage import LocalCorpusBackend
from agent.managed_storage import ManagedCorpusBackend

TENANT = '11111111-1111-4111-8111-111111111111'

@pytest.fixture
def binaries():
    folder = os.environ.get('DIARY_D2_TEST_BIN_DIR')
    if not folder:
        pytest.skip('DIARY_D2_TEST_BIN_DIR is not set')
    for name in ('diary-index-transform', 'diary-backup-run'):
        assert os.access(str(Path(folder) / name), os.X_OK)
    return Path(folder)


def store_at(path):
    journal = Journal(path / 'journal.db')
    backend = LocalCorpusBackend(str(path / 'corpus'))
    cfg = SimpleNamespace(get=lambda key, default=None: default)
    return CorpusStore(cfg, backend, journal)


QUESTIONS = [
    {'text':'ignored', 'date':{'unused':True}},
    {'action':False, 'text':'ignored'},
    {'action':'unknown', 'text':'ignored', 'date':[]},
    {'action':'add', 'text':' \t\x1c', 'replacement':[], 'date':{}},
    {'action':'add', 'text':'Where?', 'date':[]},
    {'action':'resolve', 'text':'Where?', 'replacement':{}, 'date':[]},
]
TIMELINE = [
    {'action':1, 'text':'ignored'},
    {'text':'ignored', 'date':[]},
    {'action':'add', 'text':' ', 'date':[], 'replacement':{}},
    {'action':'edit', 'text':'unused', 'replacement':'', 'date':[]},
    {'action':'add', 'text':'Arrived', 'date':None, 'replacement':{}},
    {'action':'add', 'text':'Departed', 'date':''},
]


def test_index_mixed_ignored_adapter_matches_python(binaries, monkeypatch):
    results = []
    for impl in ('python', 'rust'):
        monkeypatch.setenv(index_transform_native.SELECTOR, impl)
        idx = corpus.parse_index(None)
        changed = corpus.apply_index_edits(idx, QUESTIONS, TIMELINE, '2030-01-02')
        results.append((changed, idx, idx.render()))
    assert results[0] == results[1]
    assert '**2030-01-02** — Arrived' in results[1][2]


def test_index_journal_mixed_ignored_matches_python(binaries, monkeypatch, tmp_path):
    # These unused fields pass the actual journal sanitizer (string replacement when present).
    questions = [op for op in QUESTIONS if 'replacement' not in op or isinstance(op['replacement'],str)]
    timeline = [dict(op, replacement='unused') if isinstance(op.get('replacement'),dict) else op
                for op in TIMELINE if isinstance(op.get('text'),str)]
    results = []
    for impl in ('python', 'rust'):
        monkeypatch.setenv(index_transform_native.SELECTOR, impl)
        store = store_at(tmp_path / impl)
        try:
            jid = store.journal.enqueue('index_update', {'open_questions':questions, 'timeline':timeline, 'today':'2030-01-02'})
            assert store.apply_pending() == 1
            row = store.journal._conn.execute('SELECT applied,last_error FROM journal WHERE id=?',(jid,)).fetchone()
            assert tuple(row) == (1,None)
            results.append(store.backend.get_text(store.index_path())[0])
        finally:
            store.journal.close()
    assert results[0] == results[1]


@pytest.mark.parametrize('failure', ['missing', 'spawn', 'timeout'])
@pytest.mark.parametrize('kind', ['index_month', 'index_update'])
def test_native_outage_over_twenty_replays_then_recovers(binaries, monkeypatch, tmp_path, failure, kind):
    monkeypatch.setenv(index_transform_native.SELECTOR, 'rust')
    actual = str(binaries / 'diary-index-transform')
    broken = tmp_path / 'broken'
    if failure == 'spawn':
        broken.write_text('not executable')
    elif failure == 'timeout':
        broken.write_text('#!' + sys.executable + '\nimport time; time.sleep(5)\n')
        broken.chmod(0o755)
        monkeypatch.setattr(index_transform_native, 'call', lambda binary,payload: leaf_native.call(binary,payload,timeout=.01))
    monkeypatch.setattr(index_transform_native, 'DEFAULT_BIN', str(broken))
    assert is_transient_failure(leaf_native.LeafUnavailable('synthetic'))
    store = store_at(tmp_path / 'store')
    clock = [100.0]
    store._clock = lambda: clock[0]
    payload = ({'label':'January 2030','month':'2030-01.md'} if kind == 'index_month'
               else {'open_questions':[{'action':'add','text':'Retained?'}],'timeline':[], 'today':'2030-01-02'})
    try:
        jid = store.journal.enqueue(kind, payload)
        for _ in range(25):
            assert store.apply_pending(force=True) == 0
        entry, = store.journal.unapplied()
        assert entry.id == jid and entry.attempts == 25
        assert store.apply_pending() == 0  # real cooldown; no extra attempt
        assert store.journal.unapplied()[0].attempts == 25
        monkeypatch.setattr(index_transform_native, 'DEFAULT_BIN', actual)
        # The real binary gets its normal deadline after recovery.
        monkeypatch.setattr(index_transform_native, 'call', leaf_native.call)
        clock[0] += 31
        assert store.apply_pending() == 1
        row = store.journal._conn.execute('SELECT applied,last_error FROM journal WHERE id=?',(jid,)).fetchone()
        assert tuple(row) == (1,None)
        assert not store.journal.unapplied()
    finally:
        store.journal.close()


def _broker_process(root, remote_root, boundary, events):
    """Test-only scheduling shim: freeze the real child before killing its broker."""
    backend = ManagedCorpusBackend(root,TENANT)
    remote = LocalCorpusBackend(str(remote_root))
    real_popen = backup_native.subprocess.Popen
    real_write = backup_native.os.write
    real_verify = backend._create_verified
    child = [None]
    verified = [0]
    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        child[0] = proc
        events.send(('child',proc.pid))
        return proc
    def freeze():
        os.kill(child[0].pid,signal.SIGSTOP)
        events.send(('boundary',child[0].pid))
        # The test kills this broker; a hard deadline prevents accidental orphaning.
        signal.pause()
        raise AssertionError('broker unexpectedly resumed')
    def verify(remote,path,data):
        real_verify(remote,path,data)
        verified[0] += 1
        if boundary == 'verify' and verified[0] == 1:
            freeze()
    def write(fd,data):
        proc = child[0]
        if (boundary == 'commit' and proc is not None and fd == proc.stdin.fileno()
                and verified[0] == 2 and bytes(data) == b'{"result": "ok"}\n'):
            # Last verify acknowledgement is buffered while the child is stopped;
            # after resume its next action is the real success-row commit.
            os.kill(proc.pid,signal.SIGSTOP)
            written = real_write(fd,data)
            events.send(('boundary',proc.pid))
            signal.pause()
            return written
        return real_write(fd,data)
    backup_native.subprocess.Popen = popen
    backup_native.os.write = write
    backend._create_verified = verify
    try:
        backend.backup(remote, STORAGE, now=time.time()+100)
    finally:
        events.close()


STORAGE = {'kind':'webdav','baseUrl':'https://synthetic.invalid','username':'fixture','corpusRoot':'Diary'}

@pytest.mark.parametrize('boundary', ['verify', 'commit'])
def test_broker_death_keeps_lock_until_native_exit(binaries, monkeypatch, tmp_path, boundary):
    if not sys.platform.startswith('linux') or not hasattr(os,'pidfd_open'):
        pytest.skip('Linux pidfd required for deterministic orphan exit observation')
    monkeypatch.setenv(backup_native.SELECTOR, 'rust')
    backend = ManagedCorpusBackend(tmp_path/'managed',TENANT)
    backend.activate({'note.md':b'old'}, {})
    with backend.db() as db:
        backend.set_meta(db,'due',0)
        db.execute('INSERT INTO backups VALUES (?,?,?,0,0,NULL)',(backend.destination(STORAGE),0,1))
    context = multiprocessing.get_context('fork')
    receive, send = context.Pipe(duplex=False)
    broker = context.Process(target=_broker_process,args=(backend.root,tmp_path/'remote',boundary,send))
    child_pid = pidfd = None
    broker.start()
    send.close()
    try:
        assert receive.poll(10), 'broker did not launch worker'
        event,child_pid = receive.recv()
        assert event == 'child'
        pidfd = os.pidfd_open(child_pid)
        assert receive.poll(10), 'worker did not reach controlled boundary'
        assert receive.recv() == ('boundary',child_pid)
        broker.kill()
        broker.join(5)
        assert not broker.is_alive()
        with (backend.root/'managed-backup.lock').open('a') as locked:
            with pytest.raises(BlockingIOError):
                fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,version = backend.get('note.md')
        backend.put('note.md',b'new',if_match=version)
        remote = LocalCorpusBackend(str(tmp_path/'remote'))
        # Both replacement Python and native workers honor the inherited lock.
        for impl in ('python','rust'):
            monkeypatch.setenv(backup_native.SELECTOR,impl)
            assert not backend.backup(remote,STORAGE,now=time.time()+100).get('storageContacted')
        with backend.db() as db:
            assert db.execute('SELECT generation FROM backups').fetchone()[0] == 0
        os.kill(child_pid,signal.SIGCONT)
        assert select.select([pidfd],[],[],10)[0], 'native child did not exit'
        with backend.db() as db:
            after_old = db.execute('SELECT generation FROM backups').fetchone()[0]
        assert after_old >= 0
        with (backend.root/'managed-backup.lock').open('a') as locked:
            fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        monkeypatch.setenv(backup_native.SELECTOR,'rust')
        assert backend.backup(remote,STORAGE,now=time.time()+400)['backup'] == 'complete'
        with backend.db() as db:
            final = db.execute('SELECT generation FROM backups').fetchone()[0]
            assert final == int(backend.meta(db,'generation')) and final >= after_old
    finally:
        if broker.is_alive():
            broker.kill()
        broker.join(5)
        if pidfd is not None:
            if not select.select([pidfd],[],[],0)[0]:
                os.kill(child_pid,signal.SIGKILL)
                assert select.select([pidfd],[],[],5)[0]
            os.close(pidfd)
        receive.close()
