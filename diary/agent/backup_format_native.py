"""Opt-in adapter for diary-backup-format."""
from .leaf_native import call, selected
SELECTOR = 'DIARY_BACKUP_FORMAT_IMPL'
DEFAULT_BIN = '/usr/local/bin/diary-backup-format'

def enabled():
    return selected(SELECTOR)

def invoke(**payload):
    return call(DEFAULT_BIN, payload)
