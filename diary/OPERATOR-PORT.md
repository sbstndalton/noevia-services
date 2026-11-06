# Native Diary operator commands

The image runs `diary-backup-restore` for `backup_restore`, `backup-restore`,
`diary-backup-restore`, and the historical `python[3] -m agent.backup_restore`
command spelling. The Python module's `__main__` also execs that binary for callers
that bypass the image entrypoint. Arguments and the native process exit status pass
through without a shell wrapper. `config` and `diary-config` select `diary-config`.
The ordinary uvicorn service command is unchanged until the standalone server move.
The Rust source pin remains an integration-owner change.

The image-build smoke checks both installed native commands without starting a
server. `scripts/test_operator_dispatch.py` checks shell dispatch and the historical
Python command with a disposable executable outside the checkout. Shell dispatch
is checked by intercepting only `exec`; the real compatibility module also executes
an actual stub binary to prove argument and exit-code propagation.

## Public diagnostic output

`config` is a load-status check, not a configuration export. It emits only
`{"configured":true}` on success and a fixed error on failure. It never prints values,
field names, loader warnings or parser source excerpts. This includes malformed
YAML and invalid Unicode arguments/environment values. Full library projections
exist only in the Rust test example, which is not installed in the image.
The installed image smoke uses synthetic credential canaries on both success and
failure and exercises the real native and historical Python command spellings.

## Retained Python and importers

`agent/config.py` stays because the live Python service still imports `load_config`
and `Config`. Its three runtime importers are `agent/app.py`, `agent/corpus_store.py`
and `agent/storage.py`; migrating their HTTP/state ownership belongs to the later
standalone-server task. The gate and frozen tests also import it. Full importer list:

- `agent/app.py`
- `agent/corpus_store.py`
- `agent/storage.py`
- `gate/replay_precision.py`
- `tests/test_config.py`
- `tests/test_corpus_adaptation.py`
- `tests/test_corpus_parse_native.py`
- `tests/test_correction_citation_167.py`
- `tests/test_dedicated_storage.py`
- `tests/test_deleted_near_name.py`
- `tests/test_diary_edit.py`
- `tests/test_direct_context.py`
- `tests/test_embedding_long_chunks_1759.py`
- `tests/test_embedding_migration_1636.py`
- `tests/test_entity_focus.py`
- `tests/test_exchange_history.py`
- `tests/test_follow_up_173.py`
- `tests/test_generation_cap_127.py`
- `tests/test_journal_store.py`
- `tests/test_journal_tenant_fixes.py`
- `tests/test_near_name_followups_177_180.py`
- `tests/test_pipeline.py`
- `tests/test_reasoning_allowance_164.py`
- `tests/test_retrieval_context.py`
- `tests/test_retrieval_trace.py`
- `tests/test_rust_forward.py`
- `tests/test_s3_backend.py`
- `tests/test_s3_moto_e2e.py`
- `tests/test_s3_tenant.py`
- `tests/test_scaffold.py`
- `tests/test_score_gap.py`
- `tests/test_storage_backends.py`
- `tests/test_temporal_native.py`

`agent/backup_restore.py` keeps only its **imported restore API**, for the image smoke
and frozen tests. Its Python argument parser/CLI implementation has been removed;
executing the module transfers process ownership to Rust. Imported-API consumers:

- `scripts/smoke_diary_leaves.py`
- `tests/test_managed_storage.py`
- `tests/test_workspace_trash.py`

Cross-repository reference scan: `noevia-core` has no references to either module;
`noevia-rs` has provenance/test-oracle references; `noevia` has documentation and
preflight comments; `noevia-web/qa/restore-http.cjs` invokes the historical module
spelling, which now dispatches to the installed Rust CLI. No additional unreferenced
Python module exists in this task's two-module scope. Developer/QA environments
using the historical CLI spelling must include the native binary on PATH, as the
image does. The original restore function remains available for imported test APIs.
