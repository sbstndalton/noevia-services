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
exec "$@"
