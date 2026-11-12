"""Standalone real helper admission is not transaction publication permission."""
import json
import os
import struct
import subprocess

import pytest
from agent.managed_storage import ManagedCorpusBackend
from agent.workspace_export import archive
from agent.workspace_import import fingerprint
from agent.workspace_import_native import MAGIC

HELPER = os.environ.get('WORKSPACE_IMPORT_APPLY_TEST_BIN', '')
pytestmark = pytest.mark.skipif(not HELPER, reason='actual import helper required')


def test_standalone_helper_rechecks_late_intent_before_commit(tmp_path):
    tenant = '12345678-1234-4234-8234-123456789abc'
    target = ManagedCorpusBackend(tmp_path / 'target', tenant)
    target.activate({'keep.md': b'synthetic existing'}, {})
    source = ManagedCorpusBackend(tmp_path / 'source', '00000000-0000-0000-0000-000000000001')
    source.activate({'new.md': b'synthetic import'}, {})
    body = archive(source, '', {})
    metadata = json.dumps({'version':1, 'dbPath':str(target.db_path), 'expectedTenant':tenant,
                         'destination':'Imports/Copy', 'reviewedFingerprint':fingerprint(body, 'Imports/Copy'),
                         'debounce':3.0}).encode()
    frame = MAGIC + struct.pack('<I', len(metadata)) + metadata + struct.pack('<Q', len(body)) + body
    child = subprocess.Popen([HELPER, 'apply-v1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        child.stdin.write(frame)
        child.stdin.flush()  # Real helper remains paused waiting for required EOF.
        with target.db() as db:
            db.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic-late-put')")
        child.stdin.close()
        child.stdin = None
        output, errors = child.communicate(timeout=6)
        assert child.returncode == 3, (child.returncode, output)
        assert output == b'unavailable: backup_pending\n'
        assert errors == b''
        with target.db() as db:
            assert db.execute('SELECT count(*) FROM files').fetchone()[0] == 1
            assert db.execute('SELECT count(*) FROM workspace_imports').fetchone()[0] == 0
            assert db.execute('SELECT count(*) FROM workspace_index_outbox').fetchone()[0] == 0
            assert target.meta(db, 'generation') == '1'
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
