"""Tests that run the real `tenant-assertion` leaf (since #1071 it is the only signature check) are skipped when no
binary is usable (TENANT_ASSERTION_BIN or the image path), unless TENANT_ASSERTION_REQUIRED=1, which CI sets after
building it: then a missing binary fails them instead."""
import os

import pytest

from agent import tenant_assertion as ta

needs_binary = pytest.mark.skipif(
    os.environ.get('TENANT_ASSERTION_REQUIRED') != '1' and ta._rust_binary() is None,
    reason='tenant-assertion binary not built (set TENANT_ASSERTION_BIN; CI builds it)')
