# Native model manager

The production image runs `noevia-model-manager` as its sole process owner on port 8090.
All public `/api/v1` routes, background jobs and state writers are implemented in
`noevia-rs/bins/noevia-model-manager`. It does not start Python or a private child server.
The inference engine remains a separate service.

Build the default Docker target for production. It contains the native manager and inference
gate, CA certificates and runtime libraries. Healthchecks use `noevia-model-manager --healthcheck`.
The existing source/image pins and release assembly conventions apply.

State remains compatible: `DATA_DIR/model_loader.db`, `models.ini` and its backups,
`models.roles.json`, download partial/validator files, and embedding service state/presets.
No state migration or route contract change accompanies Python source retirement.

## Historical qualification

`qualification/python-oracle.json` pins the retired Python application's immutable source
archive and every source file hash. `restore_python_oracle.py ARCHIVE DESTINATION` verifies
both layers before creating `DESTINATION/app`; existing app directories are refused.
CI restores that source only in its disposable assembled workspace. The explicit
`legacy-runtime`/`qualification` Docker targets restore it for historical differential probes.
Production has no dependency on these targets or the Python interpreter.

Run `python qualification/check_python_retirement.py` before restoring the oracle. Historical
`tests/` continue to import the restored `app` package; committed Rust oracle fixtures and
native process/image tests remain the authority for the shipping service.

The older documents under `docs/` describe historical behavior and math. Their Python launch,
HTML UI and `python -m app.migrate_layout` instructions do not describe the current production
image. The offline layout migration utility has no native replacement and is preserved only
in the historical oracle; it is not a shipping command.
