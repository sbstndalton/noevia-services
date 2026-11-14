#!/bin/sh
# Offline operator commands share the image; the HTTP runtime remains unchanged.
set -eu
case "${1:-}" in
  backup_restore|backup-restore|diary-backup-restore)
    shift
    exec /usr/local/bin/diary-backup-restore "$@"
    ;;
  config|diary-config)
    shift
    exec /usr/local/bin/diary-config "$@"
    ;;
esac
# Historical image command spelling also transfers execution directly to Rust.
case "${1:-}" in
  python|python3)
    if [ "${2:-}" = "-m" ] && [ "${3:-}" = "agent.backup_restore" ]; then
      shift 3
      exec /usr/local/bin/diary-backup-restore "$@"
    fi
    ;;
esac
# R9 main mode (DIARY_MAIN=rust; unset = the line below, unchanged): the Rust Diary process becomes
# the main process on the public port and runs the image's HTTP command as its child on loopback.
if [ "${DIARY_MAIN:-}" = "rust" ] && [ "${1:-}" = "uvicorn" ] && [ "${2:-}" = "agent.app:app" ]; then
  exec /usr/local/bin/noevia-diary --main -- uvicorn agent.app:app --host 127.0.0.1 --port 8012
fi
exec "$@"
