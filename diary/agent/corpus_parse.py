"""Opt-in, bounded transport for the native Diary parser.

The complete response is staged and checked before any caller sees days or chunks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import date

from . import corpus as fmt

_LOG = logging.getLogger(__name__)
DEFAULT_BIN = '/usr/local/bin/diary-corpus-parse'
MAX_INPUT = 256 * 1024 * 1024
MAX_OUTPUT = 512 * 1024 * 1024
MAX_NODES = 200_000
TIMEOUT_S = 120
MAGIC_IN = b'DCP1'
MAGIC_OUT = b'DCR1'
END = b'END1'
HEX = re.compile(r'[0-9a-f]{64}\Z')
ISO = re.compile(r'\d{4}-\d{2}-\d{2}\Z')


class ParseUnavailable(ValueError):
    """Native parser, transport, or response unavailable; never an empty parse."""


class ParseCapacity(ParseUnavailable):
    """Native parser capacity boundary reached; caller may reduce input."""


@dataclass(frozen=True)
class Parsed:
    days: list[fmt.DaySection]
    chunks: list[tuple[str, str, str, str]]


def _failure(reason, capacity=False):
    _LOG.warning('Diary corpus parse %s', reason)  # Fixed codes only.
    cls = ParseCapacity if capacity else ParseUnavailable
    return cls('Diary corpus parsing capacity exceeded.' if capacity
               else 'Diary corpus parsing is unavailable. Try again later.')


def _choice():
    value = (os.environ.get('DIARY_CORPUS_PARSE_IMPL') or 'python').strip().lower()
    return 'rust' if value == 'rust' else 'python'


def _chunks(days):
    rows = []
    for day in days:
        if day.date is None:
            continue
        day_iso = day.date.isoformat()
        for sub in day.subsections:
            body = '\n'.join(
                (f'**Me:** {ex.me}\n\n**Assistant:** {ex.claude}' if ex.claude else f'**Me:** {ex.me}')
                for ex in sub.exchanges
            ).strip()
            if not body:
                continue
            text = f'[{day_iso}] {day.header}\n### {sub.header}\n{body}'
            rows.append((day_iso, sub.header, text, hashlib.sha256(text.encode('utf-8')).hexdigest()))
    return rows


def parse(text: str | None) -> Parsed:
    text = text or ''
    if _choice() == 'rust':
        return _native(text)
    days = fmt.parse_diary(text)
    return Parsed(days, _chunks(days))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def _frame(raw):
    try:
        if len(raw) < 16 or len(raw) > MAX_OUTPUT or raw[:4] != MAGIC_OUT or raw[-4:] != END:
            raise ValueError('frame')
        length = struct.unpack_from('>Q', raw, 4)[0]
        if length > MAX_OUTPUT - 16 or length != len(raw) - 16:
            raise ValueError('length')
        obj = json.loads(bytes(memoryview(raw)[12:-4]).decode('utf-8'), object_pairs_hook=_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError('constant')))
        if type(obj) is not dict or set(obj) != {'version', 'days', 'chunks'} or type(obj['version']) is not int or obj['version'] != 1:
            raise ValueError('schema')
        if type(obj['days']) is not list or type(obj['chunks']) is not list:
            raise ValueError('arrays')
        days = []
        nodes = 0
        for d in obj['days']:
            if type(d) is not dict or set(d) != {'header', 'date', 'subsections'} or type(d['header']) is not str or type(d['subsections']) is not list:
                raise ValueError('day')
            iso = d['date']
            if iso is not None:
                if type(iso) is not str or not ISO.fullmatch(iso):
                    raise ValueError('date')
                parsed_date = date.fromisoformat(iso)
                if parsed_date.isoformat() != iso:
                    raise ValueError('date')
            else:
                parsed_date = None
            subs = []
            nodes += 1
            if nodes > MAX_NODES:
                raise ValueError('nodes')
            for s in d['subsections']:
                if type(s) is not dict or set(s) != {'header', 'time', 'topic', 'exchanges'} or type(s['header']) is not str or type(s['exchanges']) is not list:
                    raise ValueError('subsection')
                exchanges = []
                nodes += 1
                if nodes > MAX_NODES:
                    raise ValueError('nodes')
                for e in s['exchanges']:
                    if type(e) is not dict or set(e) != {'me', 'claude', 'xid'} or type(e['me']) is not str or type(e['claude']) is not str or (e['xid'] is not None and type(e['xid']) is not str):
                        raise ValueError('exchange')
                    exchanges.append(fmt.Exchange(e['me'], e['claude'], e['xid']))
                    nodes += 1
                    if nodes > MAX_NODES:
                        raise ValueError('nodes')
                sub = fmt.SubSection(s['header'], exchanges)
                if s['time'] != sub.time or s['topic'] != sub.topic:
                    raise ValueError('subsection projection')
                subs.append(sub)
            days.append(fmt.DaySection(d['header'], parsed_date, subs))
        expected = _chunks(days)
        if len(obj['chunks']) != len(expected):
            raise ValueError('chunk count')
        for row, want in zip(obj['chunks'], expected):
            if type(row) is not dict or set(row) != {'day', 'header', 'text', 'content_hash'} or any(type(row[k]) is not str for k in row):
                raise ValueError('chunk')
            if not HEX.fullmatch(row['content_hash']) or (row['day'], row['header'], row['text'], row['content_hash']) != want:
                raise ValueError('chunk projection')
        return Parsed(days, expected)
    except (ValueError, TypeError, KeyError, OverflowError, UnicodeError, RecursionError, struct.error):
        raise _failure('protocol') from None


def _native(text):
    try:
        body = text.encode('utf-8')
    except UnicodeError:
        raise _failure('input_utf8') from None
    if len(body) > MAX_INPUT:
        raise _failure('input_limit', capacity=True)
    binary = DEFAULT_BIN
    if not (os.path.isabs(binary) and os.path.isfile(binary) and os.access(binary, os.X_OK)):
        raise _failure('missing_binary')
    try:
        proc = subprocess.Popen([binary, 'parse-v1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, close_fds=True,
                                env={'PATH': '/usr/local/bin:/usr/bin:/bin'})
    except OSError:
        raise _failure('spawn') from None

    output = bytearray()
    errors = bytearray()
    fault = threading.Event()
    fed = threading.Event()
    def feed():
        try:
            for part in (MAGIC_IN, struct.pack('>Q', len(body)), body, END):
                view = memoryview(part)
                while view:
                    size = proc.stdin.write(view)
                    if not size:
                        raise OSError('short write')
                    view = view[size:]
            proc.stdin.close()
            fed.set()
        except (OSError, ValueError):
            fault.set()
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass

    def drain(pipe, target, cap):
        try:
            while True:
                chunk = pipe.read1(min(65536, cap + 1 - len(target)))
                if not chunk:
                    return
                target.extend(chunk)
                if len(target) > cap:
                    fault.set()
                    return
        except (OSError, ValueError):
            fault.set()

    workers = [threading.Thread(target=feed, daemon=True),
               threading.Thread(target=drain, args=(proc.stdout, output, MAX_OUTPUT), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, errors, 256), daemon=True)]
    for worker in workers:
        worker.start()
    deadline = time.monotonic() + TIMEOUT_S
    code = None
    while True:
        if fault.is_set():
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            code = proc.wait(timeout=min(0.05, remaining))
            break
        except subprocess.TimeoutExpired:
            pass
    if code is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()
    for worker in workers:
        worker.join(timeout=max(0, deadline - time.monotonic()))
    for pipe in (proc.stdout, proc.stderr):
        try:
            pipe.close()
        except OSError:
            pass
    if fault.is_set() or code is None or any(worker.is_alive() for worker in workers) or time.monotonic() >= deadline:
        raise _failure('transport', capacity=len(output) > MAX_OUTPUT)
    if code != 0 or not fed.is_set():
        raise _failure('child_exit', capacity=code == 3 and errors == b'capacity\n')
    if errors:
        raise _failure('stderr')
    return _frame(output)
