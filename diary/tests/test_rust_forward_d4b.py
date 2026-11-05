"""D4b: the read routes the Rust Diary server adds (GET /api/files, POST /api/file,
GET /api/storage-status, GET /api/workspace-export, and the tenant refusals of GET /api/health).

Same method as test_rust_forward.py: every request runs twice on the same synthetic state, once
with Python alone and once with the forwarder and the Rust process (each a freshly signed request,
verified once by Python), and the replies are compared byte for byte (status, every header, body).
Each case also says whether the Rust side must serve it natively or decline it to Python.
POST /api/file has a body and is never forwarded (D4 part 1), so it stays on Python. Needs the real binaries (DIARY_RUST_SERVER_BIN,
the tenant-assertion leaf, and DIARY_WORKSPACE_EXPORT_BIN for the export cases).
"""
import hashlib
import json
import os
import secrets
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

import agent.app as appmod
from agent import rust_forward, workspace_export_native
from agent import tenant_assertion as ta
from agent.managed_storage import ManagedCorpusBackend
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import (BIN, DAILY, KEY, LONG, MONTHLY, TOKEN, Modes, Recording, b64,  # noqa: F401
                                     install_forwarder, needs_server, seed, start_server, world)

EXPORT_BIN = os.environ.get('DIARY_WORKSPACE_EXPORT_BIN', '')
needs_exporter = pytest.mark.skipif(not (EXPORT_BIN and os.path.isfile(EXPORT_BIN))
                                    and os.environ.get('DIARY_RUST_SERVER_REQUIRED') != '1',
                                    reason='diary-workspace-export not built (set DIARY_WORKSPACE_EXPORT_BIN)')


def sign(user, method, path, body=b'', extra=None, query=b'', content_type=None):
    headers = {'Authorization': f'Bearer {TOKEN}', 'X-Cowork-User-ID': user, **(extra or {})}
    if content_type is not None:
        headers['Content-Type'] = content_type
    body_hash = hashlib.sha256(body).hexdigest() if ta.body_is_hashed(content_type or '') else ta.STREAM
    headers[ta.HEADER] = ta.sign(KEY, user, method, path, headers, int(time.time()), secrets.token_hex(16), query,
                                 body_hash=body_hash)
    return headers


def db(state_dir, uid, name='managed-diary.db'):
    return sqlite3.connect(state_dir / 'users' / uid / name)


def quiesce(state_dir, *uids):
    """A settled tenant: nothing left for Python's recovery (dirty documents, index outbox)."""
    for uid in uids:
        with db(state_dir, uid, 'managed-index.db') as idx:
            idx.execute('DELETE FROM dirty_documents')
        with db(state_dir, uid) as conn:
            assert conn.execute('SELECT COUNT(*) FROM workspace_index_outbox').fetchone()[0] == 0


def seed_files(state_dir):
    managed = ManagedCorpusBackend(state_dir / 'users' / DAILY, DAILY)
    for path, data in (('memory.md', b'# Synthetic memory\n\nPrefers tea.\n'), ('.hidden.md', b'hidden'),
                       ('README.MD', b'upper'), ('Notes/note.md', 'Synthetic “note” ü\n'.encode()),
                       ('big.md', b'x' * (512 * 1024 + 1)), ('bad-utf8.md', b'ok \xff\xfe \xed\xa0\x80 end'),
                       ('Tie/a.md', b'a'), ('Tie/A.md', b'A'), ('Notes/.draft.md', b'd'), ('Notes/image.png', b'png')):
        assert managed.put(path, data)[0], path
    for i in range(501):
        assert managed.put(f'Bulk/{i:03d}.md', b'b')[0]
    managed.create_directory('Projects')
    managed.close()


@pytest.fixture
def pair(world, monkeypatch):
    saved = install_forwarder()
    client = TestClient(appmod.app)
    modes = Modes(monkeypatch, None)
    modes.python()
    seed(world, client)
    seed_files(world)
    if EXPORT_BIN:
        monkeypatch.setattr(workspace_export_native, 'BINARY', EXPORT_BIN)
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'rust')
    # Python recovers each tenant's state on its first full read; only then may Rust serve it.
    for uid in (DAILY, MONTHLY):
        assert client.get('/api/months', headers=sign(uid, 'GET', '/api/months')).status_code == 200
    started = []

    def start():
        started.append(start_server())
        modes.rust(started[-1])
        modes.python()
        return modes

    yield client, start, world
    for server in started:
        server.stop()
    appmod.app.user_middleware[:] = saved
    appmod.app.middleware_stack = None


def call(client, case):
    user, method, path, query, body, extra, ctype = case
    target = path + ('?' + query.decode() if query else '')
    headers = sign(user, method, path, body, extra, query, ctype)
    return client.request(method, target, headers=headers, content=body if body else None)


def wire(resp):
    return resp.status_code, [(k, v) for k, v in resp.headers.raw if k.lower() not in (b'date', b'server')], resp.content


def check(client, modes, cases):
    rows = []
    for name, case, status, native in cases:
        modes.python()
        p = call(client, case)
        modes.rust()
        Recording.served = []
        r = call(client, case)
        rows.append(f'{name:48} python={p.status_code} rust={r.status_code} native={Recording.served}')
        assert wire(p) == wire(r), name
        assert p.status_code == status, (name, p.status_code, p.content[:200])
        # native None: never forwarded (a request with a body stays on Python).
        assert Recording.served == ([] if native is None else [native]), (name, Recording.served)
    print('\n'.join(rows))


def get(user, path, query=b'', extra=None):
    return (user, 'GET', path, query, b'', extra or {}, None)


def post_file(user, body, ctype='application/json'):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return (user, 'POST', '/api/file', b'', raw, {}, ctype)


def webdav(**fields):
    return b64({'kind': 'webdav', 'baseUrl': 'https://dav.invalid/remote.php', 'username': 'synthetic',
                'corpusRoot': 'Diary', **fields})


@needs_binary
@needs_server
def test_files_parity_and_file_reads_stay_with_python(pair):
    client, start, state = pair
    modes = start()
    # Not settled yet: Python still owes this tenant a reindex, so Rust declines.
    with db(state, DAILY, 'managed-index.db') as idx:
        idx.execute("INSERT OR IGNORE INTO dirty_documents VALUES ('memory.md')")
    check(client, modes, [('files with dirty documents', get(DAILY, '/api/files'), 200, False)])
    quiesce(state, DAILY, MONTHLY)
    with db(state, DAILY) as conn:
        conn.execute("INSERT INTO workspace_index_outbox VALUES ('memory.md')")
    check(client, modes, [('files with an index outbox', get(DAILY, '/api/files'), 200, False)])
    with db(state, DAILY) as conn:
        conn.execute('DELETE FROM workspace_index_outbox')
    cases = [
        ('files root', get(DAILY, '/api/files'), 200, True),
        ('files folder', get(DAILY, '/api/files', b'path=Notes'), 200, True),
        ('files inferred folders keep the last lastmod', get(DAILY, '/api/files', b'path=Entries'), 200, True),
        ('files nested', get(DAILY, '/api/files', b'path=Entries%2F2030'), 200, True),
        ('files explicit empty folder', get(DAILY, '/api/files', b'path=Projects'), 200, True),
        ('files missing folder', get(DAILY, '/api/files', b'path=Nope'), 200, True),
        ('files last path wins', get(DAILY, '/api/files', b'path=Nope&path=Notes'), 200, True),
        ('files monthly tenant', get(MONTHLY, '/api/files'), 200, True),
        ('files too many', get(DAILY, '/api/files', b'path=Bulk'), 413, True),
        ('files hidden folder', get(DAILY, '/api/files', b'path=.git'), 400, True),
        ('files empty segment', get(DAILY, '/api/files', b'path=a//b'), 400, True),
        ('files backslash', get(DAILY, '/api/files', b'path=a%5Cb'), 400, True),
        ('files too long', get(DAILY, '/api/files', b'path=' + b'a' * 501), 400, True),
        ('files names differing only in case', get(DAILY, '/api/files', b'path=Tie'), 200, False),
        ('files non-uuid tenant', get('not-a-uuid', '/api/files'), 400, True),
        ('files storage header too large', get(DAILY, '/api/files', extra={'X-Cowork-Storage': LONG}), 413, True),
        ('file read', post_file(DAILY, {'path': 'memory.md'}), 200, None),
        ('file read nested unicode', post_file(DAILY, {'path': 'Notes/note.md'}), 200, None),
        ('file read day file', post_file(DAILY, {'path': 'Entries/2030/January/January 6, 2030.md'}), 200, None),
        ('file read missing', post_file(DAILY, {'path': 'missing.md'}), 200, None),
        ('file read upper-case extension', post_file(DAILY, {'path': 'README.MD'}), 200, None),
        ('file read invalid utf-8 (replaced)', post_file(DAILY, {'path': 'bad-utf8.md'}), 200, None),
        ('file read charset parameter', post_file(DAILY, {'path': 'memory.md'}, 'application/json; charset=utf-8'), 200, None),
        ('file read too large', post_file(DAILY, {'path': 'big.md'}), 413, None),
        ('file read not markdown', post_file(DAILY, {'path': 'Notes/image.png'}), 400, None),
        ('file read hidden', post_file(DAILY, {'path': '.hidden.md'}), 400, None),
        ('file read parent', post_file(DAILY, {'path': '../x.md'}), 400, None),
        ('file read backslash', post_file(DAILY, {'path': 'a\\b.md'}), 400, None),
        ('file read non-string path', post_file(DAILY, {'path': 5}), 400, None),
        ('file read no path', post_file(DAILY, {}), 400, None),
        ('file read not an object (422)', post_file(DAILY, [1]), 422, None),
        ('file read not JSON media (422)', post_file(DAILY, {'path': 'memory.md'}, 'text/plain'), 422, None),
        ('file read NaN body', post_file(DAILY, b'{"path":"memory.md","x":NaN}'), 200, None),
    ]
    check(client, modes, cases)
    for mode in (modes.python, modes.rust):
        mode()
        Recording.served = []
        headers = sign(DAILY, 'GET', '/api/files')
        headers['Authorization'] = 'Bearer wrong'
        resp = client.get('/api/files', headers=headers)
        assert (resp.status_code, resp.json()) == (401, {'detail': 'unauthorized'})
        assert Recording.served == ([True] if mode == modes.rust else [])


@needs_binary
@needs_server
def test_storage_status_parity(pair):
    client, start, state = pair
    quiesce(state, DAILY, MONTHLY)
    modes = start()
    complete = {'kind': 'webdav', 'baseUrl': 'https://dav.invalid/remote.php', 'username': 'synthetic', 'corpusRoot': 'Diary'}
    unicode = {'kind': 'nextcloud', 'baseUrl': 'https://dav.invalid/ü', 'username': 'synthétique', 'corpusRoot': 'Tagebuch 📓'}
    failed = {**complete, 'username': 'failing'}
    stale = {**complete, 'username': 'stale'}
    with db(state, DAILY) as conn:
        generation = int(conn.execute("SELECT value FROM meta WHERE key='generation'").fetchone()[0])
        for storage, gen, last, retry, error in ((complete, generation, 1762280000.25, 0, None),
                                                 (unicode, generation, 1762280123.5, 0, None),
                                                 (failed, generation - 1, None, 1762280300.75, 'Synthetic failure'),
                                                 (stale, generation - 1, 1762270000.0, 0, '')):
            conn.execute('INSERT INTO backups VALUES (?,?,?,?,?,?)',
                         (ManagedCorpusBackend.destination(storage), gen, last, retry, 1 if error else 0, error))
    cases = [
        ('status no descriptor', get(DAILY, '/api/storage-status'), 200, True),
        ('status local', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': b64({'kind': 'local'})}), 200, True),
        ('status not an object', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': 'WzFd'}), 200, True),
        ('status webdav, never backed up', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': webdav(corpusRoot='New')}), 200, True),
        ('status webdav complete', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': b64(complete)}), 200, True),
        ('status nextcloud non-ASCII destination', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': b64(unicode)}), 200, True),
        ('status failed', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': b64(failed)}), 200, True),
        ('status stale, empty error', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': b64(stale)}), 200, True),
        ('status blocked', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': b64({'kind': 'blocked'})}), 200, True),
        ('status monthly tenant', get(MONTHLY, '/api/storage-status'), 200, True),
        ('status non-string username', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': webdav(username=7)}), 200, False),
        ('status padded descriptor (lenient decoder)', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': b64(complete) + '=='}), 200, False),
        ('status storage header too large', get(DAILY, '/api/storage-status', extra={'X-Cowork-Storage': LONG}), 413, True),
        ('status non-uuid tenant', get('not-a-uuid', '/api/storage-status'), 400, True),
        ('health storage header too large (no detail)', get(DAILY, '/api/health', extra={'X-Cowork-Storage': LONG}), 200, True),
        ('health with tenant detail', get(DAILY, '/api/health', extra={'X-Cowork-Storage': b64({'kind': 'local'})}), 200, False),
    ]
    check(client, modes, cases)


@needs_binary
@needs_server
@needs_exporter
def test_workspace_export_parity(pair, monkeypatch):
    client, start, state = pair
    quiesce(state, DAILY, MONTHLY)
    modes = start()
    cases = [
        ('export daily', get(DAILY, '/api/workspace-export'), 200, True),
        ('export monthly', get(MONTHLY, '/api/workspace-export'), 200, True),
    ]
    check(client, modes, cases)
    modes.python()
    zip_reply = call(client, cases[0][1])
    assert zip_reply.headers['content-type'] == 'application/zip' and zip_reply.content[:2] == b'PK'
    # Over the file limit: the same 409 from the same read transaction.
    with db(state, MONTHLY) as conn:
        conn.executemany('INSERT INTO files VALUES (?,?,?,?)',
                         ((f'Bulk/{i}.md', b'b', hashlib.sha256(b'b').hexdigest(), 1762280000.0) for i in range(5001)))
    check(client, modes, [('export over 5000 files', get(MONTHLY, '/api/workspace-export'), 409, True)])


@needs_binary
@needs_server
def test_export_with_the_python_exporter_stays_with_python(pair, monkeypatch):
    client, start, state = pair
    quiesce(state, DAILY)
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'python')
    modes = start()
    modes.rust()
    Recording.served = []
    resp = call(client, get(DAILY, '/api/workspace-export'))
    assert resp.status_code == 200 and resp.content[:2] == b'PK'
    assert Recording.served == [False]


def test_d4b_routes_are_supported_but_off_by_default(monkeypatch):
    monkeypatch.delenv('DIARY_RUST_ROUTES', raising=False)
    assert rust_forward.configured_routes() == {}
    routes = rust_forward.configured_routes('/api/files,/api/file,/api/storage-status,/api/workspace-export')
    # POST /api/file carries a body, which is never forwarded: not a Rust route (yet).
    assert routes == {'/api/files': 'GET', '/api/storage-status': 'GET', '/api/workspace-export': 'GET'}


def test_server_config_carries_the_export_selection(world, monkeypatch):
    monkeypatch.setenv('WORKSPACE_EXPORT_IMPL', 'rust')
    cfg = rust_forward.server_config(appmod._base_cfg, appmod.get_state(), 8011, 't', 'f')
    assert cfg['workspace_export']['impl'] == 'rust'
    assert cfg['workspace_export']['binary'] == workspace_export_native.BINARY
    assert cfg['workspace_export']['settings']['entry_layout'] == 'daily'
    assert cfg['workspace_export']['settings']['index_enabled'] is False
