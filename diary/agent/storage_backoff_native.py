"""Opt-in adapter for diary-storage-backoff."""
from .leaf_native import call, selected
SELECTOR = 'DIARY_STORAGE_BACKOFF_IMPL'
DEFAULT_BIN = '/usr/local/bin/diary-storage-backoff'

def enabled():
    return selected(SELECTOR)

def invoke(**payload):
    return call(DEFAULT_BIN, payload)
