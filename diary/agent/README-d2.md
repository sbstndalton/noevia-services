The pure JSON leaves use a bounded `json-v1` stdin/stdout protocol. Adapters default to Python,
set an explicit ten-second total subprocess deadline, cap both request and reply, reject stderr
and nonzero exits, and always reap children. Native failure never invokes the Python algorithm.

Selectors (all default `python`): `DIARY_ENTITY_FOCUS_IMPL`, `DIARY_INDEX_TRANSFORM_IMPL`,
`DIARY_STORAGE_BACKOFF_IMPL`, `WORKSPACE_RULES_IMPL`, `DIARY_BACKUP_FORMAT_IMPL`.

The entity adapter covers asked entities, classification and identified people. Candidate policy
still runs in Python using those helpers. Storage gate state, credential caching and locking stay
in Python; the native parser receives the caller's wall clock and local-date resolution.
Workspace rules cover relative path normalization and configured protected paths. Mutating SQL
operations remain Python-owned. Backup format covers canonical Python JSON encoding only;
the backup scheduler and remote transport remain Python-owned. The optional backup
worker below owns backup snapshot and retry-state transitions.

These leaves require a paired Rust revision with their json-v1 executables. Changing the image
pin, enabling selectors, adopting the transaction engine and enabling backup orchestration are
separate integration actions. No receipt schema or operation namespace is provisioned here.

`DIARY_BACKUP_IMPL` defaults to `python`. Its Rust worker reads the unchanged managed database,
uses the format crate for manifest bytes, and updates the existing backups row. Python retains
flock through reap and passes the same locked descriptor to the child, which retains it
until exit even if the broker dies. Python brokers immutable create-and-verify through
the existing credential-gated backend;
no secret enters Rust. A total 300-second child deadline kills/reaps on protocol failure. The
caller never runs the Python worker after failure. Cadence and account concurrency remain
owned by the unchanged caller. No workspace receipt schema is used.

Native leaf unavailability is an OSError: journal replay applies its transient cooldown
and retains intent past the usual quarantine-attempt threshold. No Python fallback occurs.
