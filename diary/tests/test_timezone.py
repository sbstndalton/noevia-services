"""Pin fallback clocks to TZ, independently of the developer/CI host zone."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('zone,instant,day,header', [
    ('America/New_York', '2026-09-10T01:20:00+00:00', '2026-09-09', '### 21:20'),
    ('America/New_York', '2026-01-10T01:20:00+00:00', '2026-01-09', '### 20:20'),
    ('Asia/Tokyo', '2026-09-10T01:20:00+00:00', '2026-09-10', '### 10:20'),
    ('UTC', '2026-09-10T01:20:00+00:00', '2026-09-10', '### 01:20'),
])
def test_fallbacks_use_process_timezone(zone, instant, day, header):
    # A fresh interpreter receives TZ exactly as the container does. Freeze only
    # the instant: fromtimestamp still delegates zone conversion to the OS.
    # No prompts, real corpus, or write path are involved.
    script = '''
import json
import sys
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch
import agent.app as appmod
from agent import corpus

instant = datetime.fromisoformat(sys.argv[1]).timestamp()
class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromtimestamp(instant, tz)

store = Mock()
store.get_day_text.return_value = ''
store.get_standing_sections_text.return_value = ''
with patch.object(appmod, 'datetime', FrozenDatetime), \
     patch.object(corpus, 'datetime', FrozenDatetime), \
     patch.object(appmod, 'check_auth', return_value=True), \
     patch.object(appmod, '_tenant_state', return_value=SimpleNamespace(store=store)):
    response = json.loads(appmod.api_day(None).body)
    store.get_day_text.assert_called_once_with(
        FrozenDatetime.now().date(), max_chars=12000, include_markers=True)
    print(json.dumps([response['day'], corpus.render_subsection_header()]))
'''
    result = subprocess.run(
        [sys.executable, '-c', script, instant],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, 'TZ': zone},
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout) == [day, header]
