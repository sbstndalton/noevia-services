"""Opt-in adapter for diary-entity-focus."""
from .leaf_native import call, selected
SELECTOR = 'DIARY_ENTITY_FOCUS_IMPL'
DEFAULT_BIN = '/usr/local/bin/diary-entity-focus'

def enabled():
    return selected(SELECTOR)

def invoke(**payload):
    return call(DEFAULT_BIN, payload)
