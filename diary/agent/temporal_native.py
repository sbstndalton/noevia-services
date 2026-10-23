"""Opt-in, bounded native temporal anchor and candidate cutoff transport.

Both calls use the same validated anchor. A native fault raises before any context
or model call can use unfiltered candidates as a completed cutoff.
"""
from __future__ import annotations

from datetime import date
import json
import logging
import os
import re
import selectors
import struct
import subprocess
import time

_LOG = logging.getLogger(__name__)
DEFAULT_BIN = '/usr/local/bin/diary-temporal-cli'
MAGIC_IN = b'DTA1'
MAGIC_OUT = b'DTR1'
END = b'END1'
MAX_QUERY = 1024 * 1024
MAX_CANDIDATES = 10_000
MAX_DAY = 64
MAX_REQUEST = 2 * 1024 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_STDERR = 256
TIMEOUT_S = 5
UNAVAILABLE_TEXT = 'Diary date filtering is unavailable. Nothing was saved; try again later.'
_ISO = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\Z')


class TemporalUnavailable(RuntimeError):
    """Native temporal focus is unavailable; do not send context to a model."""


class TemporalCapacity(TemporalUnavailable):
    """The explicit native temporal request exceeded a documented limit."""


def _fault(code, capacity=False):
    _LOG.warning('Diary temporal focus %s', code)  # Fixed reason code only.
    cls = TemporalCapacity if capacity else TemporalUnavailable
    return cls('Diary temporal focus capacity exceeded.' if capacity
               else 'Diary temporal focus is unavailable. Try again later.')


def choice():
    value = (os.environ.get('DIARY_TEMPORAL_FOCUS_IMPL') or 'python').strip().lower()
    return 'rust' if value == 'rust' else 'python'


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError('duplicate key')
        out[key] = value
    return out


def _anchor_from_json(value):
    if value is None:
        return None
    if type(value) is not dict or set(value) != {'day', 'inclusive'}:
        raise ValueError('anchor shape')
    day_text = value['day']
    if type(day_text) is not str or not _ISO.fullmatch(day_text) or type(value['inclusive']) is not bool:
        raise ValueError('anchor fields')
    day = date.fromisoformat(day_text)
    if day.isoformat() != day_text:
        raise ValueError('anchor canonical date')
    return day, value['inclusive']


def _anchor_to_json(value):
    if value is None:
        return None
    if (type(value) is not tuple or len(value) != 2 or type(value[0]) is not date
            or type(value[1]) is not bool):
        raise _fault('anchor_input')
    return {'day': value[0].isoformat(), 'inclusive': value[1]}


def _frame(payload):
    try:
        if not 16 <= len(payload) <= MAX_RESPONSE + 16 or payload[:4] != MAGIC_OUT or payload[-4:] != END:
            raise ValueError('frame')
        size = struct.unpack_from('>Q', payload, 4)[0]
        if size > MAX_RESPONSE or size != len(payload) - 16:
            raise ValueError('length')
        reply = json.loads(bytes(memoryview(payload)[12:-4]).decode('utf-8'), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
        if type(reply) is not dict or type(reply.get('version')) is not int or reply['version'] != 1:
            raise ValueError('version')
        return reply
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError, struct.error):
        raise _fault('protocol') from None


def _call(request):
    try:
        encoded = json.dumps(request, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, OverflowError):
        raise _fault('request_encoding') from None
    if len(encoded) > MAX_REQUEST - 16:
        raise _fault('request_capacity', capacity=True)
    body = MAGIC_IN + struct.pack('>Q', len(encoded)) + encoded + END
    binary = DEFAULT_BIN
    if not (os.path.isabs(binary) and os.path.isfile(binary) and os.access(binary, os.X_OK)):
        raise _fault('missing_binary')
    try:
        selector = selectors.DefaultSelector()
        proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0, close_fds=True,
                                env={'PATH': '/usr/local/bin:/usr/bin:/bin'})
    except OSError:
        if 'selector' in locals():
            selector.close()
        raise _fault('spawn') from None
    output = bytearray()
    stderr_bytes = 0
    offset = 0
    fed = False
    fault = None
    deadline = time.monotonic() + TIMEOUT_S

    def close(stream):
        try:
            selector.unregister(stream)
        except (KeyError, ValueError):
            pass
        stream.close()

    try:
        for stream, event, kind in ((proc.stdin, selectors.EVENT_WRITE, 'input'),
                                    (proc.stdout, selectors.EVENT_READ, 'output'),
                                    (proc.stderr, selectors.EVENT_READ, 'error')):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, event, kind)
        while selector.get_map() or proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                fault = 'timeout'
                break
            for key, _ in selector.select(timeout=min(.05, remaining)):
                stream, kind = key.fileobj, key.data
                try:
                    if kind == 'input':
                        if offset == len(body):
                            close(stream)
                            fed = True
                            continue
                        count = os.write(stream.fileno(), memoryview(body)[offset:offset + 65536])
                        if count == 0:
                            raise OSError('short write')
                        offset += count
                    else:
                        cap = MAX_RESPONSE + 16 if kind == 'output' else MAX_STDERR
                        used = len(output) if kind == 'output' else stderr_bytes
                        chunk = os.read(stream.fileno(), min(65536, cap + 1 - used))
                        if not chunk:
                            close(stream)
                            continue
                        if kind == 'output':
                            output.extend(chunk)
                        else:
                            stderr_bytes += len(chunk)
                        if used + len(chunk) > cap:
                            fault = kind + '_overflow'
                            break
                except BlockingIOError:
                    continue
                except OSError:
                    fault = kind + '_io'
                    break
            if fault:
                break
    except (OSError, ValueError):
        fault = 'io'
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        code = proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if not stream.closed:
                stream.close()
        selector.close()
    if fault or not fed or time.monotonic() >= deadline:
        raise _fault(fault or 'transport', capacity=fault == 'output_overflow')
    if code != 0:
        raise _fault('child_exit', capacity=code == 3)
    if stderr_bytes:
        raise _fault('stderr')
    return _frame(output)


def anchor(query):
    if type(query) is not str:
        raise _fault('query_type')
    try:
        if len(query.encode('utf-8')) > MAX_QUERY:
            raise _fault('query_capacity', capacity=True)
    except UnicodeError:
        raise _fault('query_utf8') from None
    reply = _call({'version': 1, 'op': 'anchor', 'query': query})
    try:
        if set(reply) != {'version', 'op', 'anchor'} or reply['op'] != 'anchor':
            raise ValueError('anchor reply')
        return _anchor_from_json(reply['anchor'])
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
        raise _fault('anchor_protocol') from None


def cutoff(anchor_value, rows):
    if len(rows) > MAX_CANDIDATES:
        raise _fault('candidate_capacity', capacity=True)
    anchor_json = _anchor_to_json(anchor_value)
    candidates = []
    try:
        for row in rows:
            day = str(row.get('day'))
            if len(day.encode('utf-8')) > MAX_DAY:
                raise _fault('day_capacity', capacity=True)
            candidates.append({'day': day})
    except (AttributeError, TypeError, UnicodeError):
        raise _fault('candidate_input') from None
    reply = _call({'version': 1, 'op': 'cutoff', 'anchor': anchor_json, 'candidates': candidates})
    try:
        if set(reply) != {'version', 'op', 'anchor', 'kept_ranks', 'removed'} or reply['op'] != 'cutoff':
            raise ValueError('cutoff reply')
        if _anchor_from_json(reply['anchor']) != anchor_value:
            raise ValueError('wrong anchor')
        kept = reply['kept_ranks']
        removed = reply['removed']
        if type(kept) is not list or type(removed) is not list:
            raise ValueError('partition arrays')
        dropped = []
        for item in removed:
            if type(item) is not dict or set(item) != {'rank', 'reason'} or item['reason'] != 'after_time_anchor':
                raise ValueError('removal')
            dropped.append(item['rank'])
        n = len(rows)
        for ranks in (kept, dropped):
            if any(type(rank) is not int or rank < 1 or rank > n for rank in ranks):
                raise ValueError('rank range')
            if ranks != sorted(set(ranks)):
                raise ValueError('rank order')
        if sorted(kept + dropped) != list(range(1, n + 1)):
            raise ValueError('partition')
        if anchor_value is None and dropped:
            raise ValueError('no-anchor removal')
        return kept, dropped
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
        raise _fault('cutoff_protocol') from None
