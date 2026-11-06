#!/bin/sh
# Offline operator commands share the image; the HTTP runtime remains unchanged.
set -eu
case "${1:-}" in
  backup_restore|backup-restore)
    shift
    exec /usr/local/bin/diary-backup-restore "$@"
    ;;
  config)
    shift
    exec /usr/local/bin/diary-config "$@"
    ;;
esac
exec "$@"
