"""Opt-in adapter for diary-workspace-rules."""
from .leaf_native import call, selected
SELECTOR = 'WORKSPACE_RULES_IMPL'
DEFAULT_BIN = '/usr/local/bin/diary-workspace-rules'

def enabled():
    return selected(SELECTOR)

def invoke(**payload):
    return call(DEFAULT_BIN, payload)
