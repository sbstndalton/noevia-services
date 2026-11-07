"""W1 fixture ownership regressions; no real Diary server is started by these cases.

The ordered witness calls the existing backup [False] case, including its real
ForwardMiddleware/scripted child, then drives the existing parity fixture only
through its seed. Runtime qualification still requires the tenant-assertion leaf.
"""
from collections import OrderedDict
import sqlite3
import sys
from types import SimpleNamespace

import pytest

import agent.app as appmod
from agent import rust_forward, tenant_writer
from agent.managed_storage import ManagedCorpusBackend
from tests import test_rust_backup_forward as backup
from tests import test_rust_forward as reads
from tests import test_rust_forward_d4b as files


class SeedComplete(RuntimeError):
    pass


@pytest.mark.parametrize('incoming', [frozenset(), frozenset({reads.OTHER}),
                                    frozenset({reads.DAILY, reads.MONTHLY})],
                         ids=['empty', 'unrelated', 'registered-seed-tenants'])
def test_backup_false_restores_incoming_before_parity_seed(tmp_path, incoming, record_property):
    saved, stack = list(appmod.app.user_middleware), appmod.app.middleware_stack
    sentinel = object()
    incoming_tenants = OrderedDict([('synthetic-existing-state', sentinel)])
    with pytest.MonkeyPatch.context() as outer:
        outer.setattr(tenant_writer, '_writers', incoming)
        outer.setattr(appmod, '_tenant_states', incoming_tenants)
        with pytest.MonkeyPatch.context() as backup_patch:
            backup.test_backup_does_not_kill_settling_child(backup_patch, False)
        # This checkpoint must precede world/seed. E3 leaks DAILY here; do not
        # conceal it by resetting the registry in world or before the assertion.
        record_property('registry_restored_before_seed', tenant_writer._writers is incoming)
        assert tenant_writer._writers is incoming, 'backup teardown leaked its configured writer registry'
        alias = sys.modules.get('diary.tests.test_rust_forward')
        record_property('reads_is_canonical_module', reads is sys.modules['tests.test_rust_forward'])
        record_property('diary_alias_present', alias is not None)
        record_property('diary_alias_is_same_object', alias is reads)
        checkpoints = []
        stop = SeedComplete('synthetic seed reached native-start boundary')
        with pytest.MonkeyPatch.context() as setup:
            world = reads.world.__wrapped__(tmp_path, setup)
            state = next(world)
            owned_state = appmod._state
            owned_tenants = appmod._tenant_states

            def before_native_start():
                assert (state / 'users' / reads.DAILY / 'managed-diary.active').is_file()
                assert (state / 'users' / reads.MONTHLY / 'managed-diary.active').is_file()
                assert tenant_writer._writers is incoming
                checkpoints.append('seed-complete-before-native-start')
                raise stop

            setup.setattr(reads, 'start_server', before_native_start)
            pair = reads.both.__wrapped__(state, setup)
            try:
                with pytest.raises(SeedComplete) as caught:
                    next(pair)
                assert caught.value is stop
                assert checkpoints == ['seed-complete-before-native-start']
                assert appmod.app.user_middleware == saved
                assert appmod.app.middleware_stack is stack
                owned = [owned_state, *owned_tenants.values()]
            finally:
                pair.close()
                # Pytest resumes a yield fixture normally during teardown.
                # close() would inject GeneratorExit and hide a cleanup error.
                with pytest.raises(StopIteration):
                    next(world)
            assert all(state.store.closed for state in owned)
            assert owned_tenants == {}
        assert appmod._tenant_states is incoming_tenants
        assert list(incoming_tenants.items()) == [('synthetic-existing-state', sentinel)]
        assert tenant_writer._writers is incoming
    assert appmod.app.user_middleware == saved and appmod.app.middleware_stack is stack


def test_registered_seed_tenant_keeps_writer_and_backup_guards(tmp_path, monkeypatch):
    registry = frozenset({reads.DAILY, reads.OTHER})
    monkeypatch.setattr(tenant_writer, '_writers', registry)
    monkeypatch.setattr(tenant_writer, '_data_root', tmp_path)
    monkeypatch.setattr(tenant_writer, 'denials', [])
    backend = ManagedCorpusBackend(tmp_path / 'users' / reads.DAILY, reads.DAILY)
    try:
        # Valid setup first: negative injections apply only to an active store.
        with tenant_writer.hold(reads.DAILY):
            backend.activate({'keep.md': b'synthetic'}, {})
        with pytest.raises(sqlite3.DatabaseError, match='not authorized'):
            backend.put('unheld.md', b'synthetic')
        assert tenant_writer.denials[-1][0] == reads.DAILY
        with tenant_writer.hold(reads.OTHER):
            assert not tenant_writer.held(reads.DAILY)
            with pytest.raises(sqlite3.DatabaseError, match='not authorized'):
                backend.put('wrong-tenant-lock.md', b'synthetic')
        with tenant_writer.hold(reads.DAILY):
            assert backend.put('held.md', b'synthetic')[0]
            with sqlite3.connect(backend.db_path) as db:
                db.execute("INSERT INTO meta VALUES ('backup_effect_pending','synthetic-late-intent')")
            assert tenant_writer.held(reads.DAILY)
            with pytest.raises(sqlite3.DatabaseError, match='not authorized'):
                backend.put('intent-blocked.md', b'synthetic')
        with pytest.raises(tenant_writer.WriterBusy):
            tenant_writer.acquire(reads.DAILY, deadline_s=.01)
        assert all(backend.get(path)[0] is None for path in
                   ('unheld.md', 'wrong-tenant-lock.md', 'intent-blocked.md'))
        assert backend.get('held.md')[0] == b'synthetic'
        assert tenant_writer._writers is registry
    finally:
        with sqlite3.connect(backend.db_path) as db:
            db.execute("DELETE FROM meta WHERE key='backup_effect_pending'")
        backend.close()


@pytest.mark.parametrize('module,fixture,failed_stage',
                         [(reads, 'both', 'seed'), (files, 'pair', 'seed'),
                          (files, 'pair', 'seed_files')],
                         ids=['read-seed', 'files-seed', 'files-extra-seed'])
@pytest.mark.parametrize('cleanup_fails', [False, True], ids=['close-ok', 'close-also-fails'])
def test_failed_seed_restores_only_owned_fixture_resources(tmp_path, module, fixture,
                                                          failed_stage, cleanup_fails, record_property):
    saved, stack = list(appmod.app.user_middleware), appmod.app.middleware_stack
    sentinel = object()
    incoming_tenants = OrderedDict([('synthetic-existing-state', sentinel)])
    # Keep baseline setup valid too: these cases diagnose cleanup after the
    # injected failure, independently of the registered-seed locking regression.
    incoming_registry = frozenset({reads.OTHER})
    failure = RuntimeError('synthetic failure after valid seed')
    clients_closed, states_closed, native_starts = [], [], []
    real_client, real_close_state = module.TestClient, appmod._close_state
    real_stage = getattr(module, failed_stage)

    def client_factory(*args, **kwargs):
        client = real_client(*args, **kwargs)
        close = client.close

        def tracked_close():
            close()
            clients_closed.append(client)
            if cleanup_fails:
                raise RuntimeError('synthetic client cleanup failure')
        client.close = tracked_close
        return client

    def close_state(state):
        assert state is not sentinel
        states_closed.append(state)
        real_close_state(state)

    def fail_after_valid_seed(*args):
        real_stage(*args)
        raise failure

    def forbidden_start():
        native_starts.append(True)
        raise AssertionError('failed seed must precede native startup')

    with pytest.MonkeyPatch.context() as outer:
        outer.setattr(tenant_writer, '_writers', incoming_registry)
        outer.setattr(appmod, '_tenant_states', incoming_tenants)
        outer.setattr(appmod, '_state', sentinel)
        with pytest.MonkeyPatch.context() as setup:
            world = reads.world.__wrapped__(tmp_path, setup)
            state = next(world)
            owned_state, owned_tenants = appmod._state, appmod._tenant_states
            setup.setattr(module, 'TestClient', client_factory)
            setup.setattr(module, failed_stage, fail_after_valid_seed)
            setup.setattr(module, 'start_server', forbidden_start)
            setup.setattr(appmod, '_close_state', close_state)
            pair = getattr(module, fixture).__wrapped__(state, setup)
            try:
                with pytest.raises(RuntimeError) as caught:
                    next(pair)
                assert caught.value is failure
                assert not native_starts
                assert len(clients_closed) == 1
                assert appmod.app.user_middleware == saved
                assert appmod.app.middleware_stack is stack
                owned = {id(item): item for item in (owned_state, *owned_tenants.values())}
                record_property('original_seed_exception_preserved', caught.value is failure)
            finally:
                pair.close()
                world.close()
            assert {id(item) for item in states_closed} == set(owned)
            assert all(item.store.closed for item in owned.values())
            assert owned_tenants == {}
        assert appmod._state is sentinel
        assert appmod._tenant_states is incoming_tenants
        assert list(incoming_tenants.items()) == [('synthetic-existing-state', sentinel)]
        assert tenant_writer._writers is incoming_registry


@pytest.mark.parametrize('cleanup_fails', [False, True])
@pytest.mark.parametrize('failed_stage', ['start', 'readiness'])
def test_start_server_reaps_its_partial_start_and_preserves_exception(monkeypatch, cleanup_fails, failed_stage):
    failure = RuntimeError('synthetic partial server startup failure')
    events = []
    ticks = iter((0, 16))
    monkeypatch.setattr(reads, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))

    class Child:
        ready = False

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append('owned-start-attempt')
            if failed_stage == 'start':
                raise failure

        def stop(self):
            events.append('owned-stop')
            if cleanup_fails:
                raise RuntimeError('synthetic child cleanup failure')

    monkeypatch.setattr(rust_forward, 'RustServer', Child)
    expected = RuntimeError if failed_stage == 'start' else AssertionError
    with pytest.raises(expected) as caught:
        reads.start_server()
    if failed_stage == 'start':
        assert caught.value is failure
    else:
        assert str(caught.value) == 'noevia-diary did not become ready'
    assert events == ['owned-start-attempt', 'owned-stop']


@pytest.mark.parametrize('fixture_module,fixture_name', [(reads, 'both'), (files, 'pair')],
                         ids=['read-pair', 'files-pair'])
def test_body_failure_closes_all_owned_servers_even_if_first_stop_fails(tmp_path, monkeypatch,
                                                                      fixture_module, fixture_name):
    saved, stack = list(appmod.app.user_middleware), appmod.app.middleware_stack
    failure = RuntimeError('synthetic fixture body failure')
    stopped, created, client_closed = [], [], []
    real_client = fixture_module.TestClient

    class Child:
        def stop(self):
            stopped.append(self)
            if self is created[0]:
                raise RuntimeError('synthetic first child cleanup failure')

    def start():
        child = Child()
        created.append(child)
        return child

    def client_factory(*args, **kwargs):
        client = real_client(*args, **kwargs)
        close = client.close

        def tracked_close():
            close()
            client_closed.append(True)
        client.close = tracked_close
        return client

    # Isolate the body/cleanup witness from the separate incoming-writer seed bug.
    monkeypatch.setattr(tenant_writer, '_writers', frozenset({reads.OTHER}))
    world = reads.world.__wrapped__(tmp_path, monkeypatch)
    state = next(world)
    monkeypatch.setattr(fixture_module, 'start_server', start)
    monkeypatch.setattr(fixture_module, 'TestClient', client_factory)
    pair = getattr(fixture_module, fixture_name).__wrapped__(state, monkeypatch)
    try:
        result = next(pair)  # All real synthetic seeding/recovery completes first.
        if fixture_name == 'both':
            result[1].rust(start())  # Same restart path as the existing replay test.
        else:
            result[1]()
            result[1]()
        assert len(created) == 2
        with pytest.raises(RuntimeError) as caught:
            pair.throw(failure)
        assert caught.value is failure
        assert stopped == created
        assert client_closed == [True]
        assert appmod.app.user_middleware == saved
        assert appmod.app.middleware_stack is stack
    finally:
        pair.close()
        world.close()
