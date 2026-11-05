"""D6: GET /api/external-sources, served natively by the Rust Diary server.

Same method as test_rust_forward_d4b.py: every request runs twice on the same synthetic state (once
with Python alone, once through the forwarder and the Rust process, each a freshly signed request
verified once by Python) and the replies are compared byte for byte. Each case also says whether
the Rust side must serve it natively or decline it to Python. The folders are synthetic trees under
the test's tmp directory; nothing outside it is read. Needs the real binaries (DIARY_RUST_SERVER_BIN
and the tenant-assertion leaf).
"""
import base64
import hashlib
import json
import os
import time

from agent import rust_forward
from agent.external_sources import external_source_paths
import agent.app as appmod
from tests.tenant_binary import needs_binary
from tests.test_rust_forward import DAILY, MONTHLY, needs_server, world  # noqa: F401
from tests.test_rust_forward_d4b import LONG, check, db, get, pair, quiesce, sign  # noqa: F401

DAY = 24 * 3600


def touch(path, text='x', days_ago=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    if days_ago is not None:
        # Noon, so no timezone offset moves the date.
        stamp = time.mktime(time.strptime('2026-03-15 12:00:00', '%Y-%m-%d %H:%M:%S')) - days_ago * DAY
        os.utime(path, (stamp, stamp))


def build_tree(root):
    touch(root / '2026-09-06 trip.md', days_ago=1)
    touch(root / 'notes_20260907.txt', 'abc', days_ago=2)
    touch(root / 'undated.markdown', 'u' * 40, days_ago=3)
    touch(root / 'UPPER.MD', days_ago=4)
    touch(root / '.hidden.md', days_ago=5)
    touch(root / '.md', days_ago=6)  # a name that is only an extension has no suffix
    touch(root / 'trailing.md.', days_ago=6)
    touch(root / 'image.png', days_ago=6)
    touch(root / 'sub' / 'dir' / '2026-02-30 bad.md', days_ago=7)  # not a calendar date: mtime
    touch(root / 'sub' / 'dir' / 'ünï 2026-1-2 x.txt', 'uni', days_ago=8)
    touch(root / 'sub' / 'x2026-09-061.md', days_ago=9)  # a digit after the day: no match
    touch(root / 'sub' / 'a' / 'a.md', days_ago=10)
    touch(root / 'sub' / 'a' / 'b.md', days_ago=10)  # the same date: ordered by rel_path
    touch(root / 'sub' / '0000-01-01.md', days_ago=11)  # year 0 is not a date
    touch(root / 'sub' / '20261301 14.md', days_ago=12)
    touch(root / 'tab\there.md', days_ago=13)  # a control character in the name
    touch(root / 'quote"back\\slash.md', days_ago=13)
    os.symlink(root / 'sub', root / 'link-to-dir')  # not listed, not entered
    os.symlink(root / 'UPPER.MD', root / 'link-to-file.md')  # listed (stat follows it)
    os.symlink(root / 'does-not-exist.md', root / 'broken.md')  # skipped


def sources(monkeypatch, *paths):
    monkeypatch.setenv('DIARY_EXTERNAL_SOURCES', ','.join(str(p) for p in paths))


@needs_binary
@needs_server
def test_external_sources_parity(pair, monkeypatch, tmp_path):
    client, start, state = pair
    tree, empty, plain = tmp_path / 'src-tree', tmp_path / 'src-empty', tmp_path / 'src-file.txt'
    build_tree(tree)
    empty.mkdir()
    plain.write_text('not a directory')
    sources(monkeypatch, tree)
    modes = start()
    quiesce(state, DAILY, MONTHLY)
    ext = '/api/external-sources'
    check(client, modes, [
        ('one tree', get(DAILY, ext), 200, True),
        ('one tree monthly tenant', get(MONTHLY, ext), 200, True),
        ('non-uuid tenant', get('not-a-uuid', ext), 400, True),
        ('storage header too large', get(DAILY, ext, extra={'X-Cowork-Storage': LONG}), 413, True),
    ])
    for mode in (modes.python, modes.rust):
        mode()
        headers = sign(DAILY, 'GET', ext)
        headers['Authorization'] = 'Bearer wrong'
        resp = client.get(ext, headers=headers)
        assert (resp.status_code, resp.json()) == (401, {'error': 'unauthorized'})


@needs_binary
@needs_server
def test_missing_file_and_empty_sources(pair, monkeypatch, tmp_path):
    client, start, state = pair
    tree, empty, plain = tmp_path / 'src-tree', tmp_path / 'src-empty', tmp_path / 'src-file.txt'
    build_tree(tree)
    empty.mkdir()
    plain.write_text('not a directory')
    sources(monkeypatch, tree, tmp_path / 'not-mounted', plain, empty, tree)  # the repeat is dropped
    modes = start()
    quiesce(state, DAILY, MONTHLY)
    check(client, modes, [('several sources', get(DAILY, '/api/external-sources'), 200, True)])


@needs_binary
@needs_server
def test_nothing_configured(pair, monkeypatch):
    client, start, state = pair
    monkeypatch.setenv('DIARY_EXTERNAL_SOURCES', ' , ')
    modes = start()
    quiesce(state, DAILY, MONTHLY)
    check(client, modes, [('not configured', get(DAILY, '/api/external-sources'), 200, True)])


@needs_binary
@needs_server
def test_listings_rust_cannot_reproduce_stay_with_python(pair, monkeypatch, tmp_path):
    client, start, state = pair
    full, wide, many = tmp_path / 'src-exact', tmp_path / 'src-wide', tmp_path / 'src-many'
    for i in range(500):  # exactly at the cap: not truncated, served natively
        touch(full / f'2026-01-{i % 28 + 1:02d} e{i:03d}.txt', days_ago=1)
    touch(wide / '２０２６-０１-０１ fullwidth.md', days_ago=1)  # Unicode digits match \d in Python
    for i in range(501):  # over the cap: which 500 survive depends on directory order
        touch(many / f'm{i:03d}.txt', days_ago=1)
    modes = start_with(start, monkeypatch, state, full)
    check(client, modes, [('exactly 500 files', get(DAILY, '/api/external-sources'), 200, True)])
    modes = start_with(start, monkeypatch, state, wide)
    check(client, modes, [('non-ASCII digit in a name', get(DAILY, '/api/external-sources'), 200, False)])
    modes = start_with(start, monkeypatch, state, many)
    check(client, modes, [('more than 500 files', get(DAILY, '/api/external-sources'), 200, False)])


def start_with(start, monkeypatch, state, *paths):
    sources(monkeypatch, *paths)
    modes = start()
    quiesce(state, DAILY, MONTHLY)
    return modes


def test_external_sources_route_is_supported_but_off_by_default(monkeypatch):
    monkeypatch.delenv('DIARY_RUST_ROUTES', raising=False)
    assert rust_forward.configured_routes() == {}
    assert rust_forward.configured_routes('/api/external-sources,/api/external-sources/import,/api/workspace-trash') == {
        '/api/external-sources': 'GET', '/api/workspace-trash': 'GET'}


def test_server_config_carries_the_resolved_source_paths(world, monkeypatch, tmp_path):
    monkeypatch.setenv('DIARY_EXTERNAL_SOURCES', f' {tmp_path}/a ,{tmp_path}/a/, ,{tmp_path}/b,~/x')
    cfg = rust_forward.server_config(appmod._base_cfg, appmod.get_state(), 8011, 't', 'f')
    assert cfg['external_sources'] == external_source_paths()
    assert cfg['external_sources'][:2] == [f'{tmp_path}/a', f'{tmp_path}/b']
    monkeypatch.delenv('DIARY_EXTERNAL_SOURCES')
    cfg = rust_forward.server_config(appmod._base_cfg, appmod.get_state(), 8011, 't', 'f')
    assert cfg['external_sources'] == []



# ---------------------------------------------------------------- GET /api/workspace-trash

PREFIX = '.noevia-trash/'


def op(n):
    return '00000000-0000-4000-8000-%012d' % n


def capsule(operation, path='Notes/a.md', data=b'hello', state='trashed', **over):
    record = {'format': 'noevia-trash-v1', 'id': operation, 'path': path, 'version': hashlib.sha256(data).hexdigest(),
              'data': base64.b64encode(data).decode(), 'state': state, 'trashedAt': 1730000000.5, **over}
    return json.dumps(record, sort_keys=True, ensure_ascii=False).encode()


def put(state_dir, uid, rows):
    with db(state_dir, uid) as conn:
        for path, data in rows:
            conn.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?)',
                         (path, data, hashlib.sha256(data).hexdigest(), 1.0))


def drop(state_dir, uid, like):
    with db(state_dir, uid) as conn:
        conn.execute('DELETE FROM files WHERE path LIKE ?', (like,))


TRASH = '/api/workspace-trash'


@needs_binary
@needs_server
def test_workspace_trash_list_parity(pair):
    client, start, state = pair
    modes = start()
    quiesce(state, DAILY, MONTHLY)
    check(client, modes, [('trash empty', get(DAILY, TRASH), 200, True)])
    good = [(PREFIX + op(n) + '.json', capsule(op(n), path=f'Notes/n{n}.md', data=b'x' * n)) for n in range(1, 6)]
    put(state, DAILY, good + [
        (PREFIX + op(6) + '.json', capsule(op(6), state='restored', restoredAt=1730000001.25)),
        (PREFIX + op(7) + '.json', b'{not json'),
        (PREFIX + op(8) + '.json', capsule(op(8), version='0' * 64)),
        (PREFIX + op(9) + '.json', capsule(op(9), format='other')),
        (PREFIX + op(10) + '.json', capsule(op(10), id=op(99))),
        (PREFIX + op(11) + '.json', capsule(op(11), path='.hidden/x.md')),
        (PREFIX + op(12) + '.json', capsule(op(12), data='€'.encode()).replace(b'4oKs', b'4oK*')),
        (PREFIX + op(13) + '.json', capsule(op(13), state='deleted')),
        (PREFIX + op(14) + '.json', b'[1, 2]'),
        (PREFIX + op(15) + '.json', capsule(op(15), path='Ünï/ü.md', data=b'\x00\xff\xfe')),
        (PREFIX + op(16) + '.json', capsule(op(16), trashedAt=1730000000)),
        ('Notes/other.md', b'not a capsule')])
    cases = [
        ('trash list', get(DAILY, TRASH), 200, True),
        ('trash list monthly tenant (none)', get(MONTHLY, TRASH), 200, True),
        ('after a valid id', get(DAILY, TRASH, b'after=' + op(3).encode()), 200, True),
        ('after in upper case', get(DAILY, TRASH, b'after=' + op(3).upper().encode()), 200, True),
        ('after without hyphens', get(DAILY, TRASH, b'after=' + op(3).replace('-', '').encode()), 200, True),
        ('after repeated (last wins)', get(DAILY, TRASH, b'after=bad&after=' + op(9).encode()), 200, True),
        ('after past the end', get(DAILY, TRASH, b'after=' + op(99).encode()), 200, True),
        ('after is not an id', get(DAILY, TRASH, b'after=abc'), 409, True),
        ('after too long to be an id', get(DAILY, TRASH, b'after=' + b'a' * 40), 409, True),
        ('after in forms only Python reads', get(DAILY, TRASH, b'after=urn:uuid:' + op(3).encode()), 200, False),
        ('after with braces', get(DAILY, TRASH, b'after=%7B' + op(3).encode() + b'%7D'), 200, False),
        ('after not hex', get(DAILY, TRASH, b'after=zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz'), 409, False),
        ('non-uuid tenant', get('not-a-uuid', TRASH), 400, True),
        ('storage header too large', get(DAILY, TRASH, extra={'X-Cowork-Storage': LONG}), 413, True),
    ]
    check(client, modes, cases)
    for mode in (modes.python, modes.rust):
        mode()
        headers = sign(DAILY, 'GET', TRASH)
        headers['Authorization'] = 'Bearer wrong'
        resp = client.get(TRASH, headers=headers)
        assert (resp.status_code, resp.json()) == (401, {'detail': 'unauthorized'})
    # A capsule name that is not an id fails the whole listing (409) - or is Python's to judge.
    put(state, DAILY, [(PREFIX + 'abc.json', capsule(op(1)))])
    check(client, modes, [('capsule name is a short hex string', get(DAILY, TRASH), 409, True)])
    drop(state, DAILY, PREFIX + 'abc.json')
    put(state, DAILY, [(PREFIX + 'zzz.json', capsule(op(1)))])
    check(client, modes, [('capsule name is not hex', get(DAILY, TRASH), 409, False)])
    drop(state, DAILY, PREFIX + 'zzz.json')
    put(state, DAILY, [(PREFIX + 'no-extension', capsule(op(1)))])
    check(client, modes, [('stray file in the namespace', get(DAILY, TRASH), 409, False)])
    drop(state, DAILY, PREFIX + 'no-extension')


@needs_binary
@needs_server
def test_workspace_trash_pages(pair):
    client, start, state = pair
    modes = start()
    put(state, MONTHLY, [(PREFIX + op(n) + '.json', capsule(op(n), path=f'Notes/n{n}.md')) for n in range(1, 106)])
    quiesce(state, DAILY, MONTHLY)
    page = client.get(TRASH, headers=sign(MONTHLY, 'GET', TRASH)).json()
    assert len(page['records']) == 100 and page['next'] == op(100)
    check(client, modes, [
        ('first page', get(MONTHLY, TRASH), 200, True),
        ('second page', get(MONTHLY, TRASH, b'after=' + op(100).encode()), 200, True),
    ])
