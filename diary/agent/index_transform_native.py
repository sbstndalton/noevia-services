"""Opt-in adapter for diary-index-transform."""
from .leaf_native import call, selected
SELECTOR = 'DIARY_INDEX_TRANSFORM_IMPL'
DEFAULT_BIN = '/usr/local/bin/diary-index-transform'

def enabled():
    return selected(SELECTOR)

def invoke(**payload):
    return call(DEFAULT_BIN, payload)
