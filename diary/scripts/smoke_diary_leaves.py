"""Image/overlay smoke: each installed Diary Rust leaf answers its protocol and agrees with Python.

Every leaf named on the command line (or every one with --all) is run once through its real adapter with
its selector set to `rust` in this short-lived process only, on synthetic data, and the result is compared
with the Python path. Nothing is written outside a temporary directory, and no service, model or network is
touched. A leaf that is missing, unavailable or disagrees fails the smoke (exit 1).

The Diary Dockerfile runs it with --all after copying the /out/ bundle; the Diary overlay runs it in the
candidate for the leaves it built. Every selector stays `python` in the running service.
"""
import argparse
from contextlib import contextmanager
from datetime import date, datetime
import io
import os
from pathlib import Path
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import corpus as fmt  # noqa: E402
from agent import corpus_parse  # noqa: E402
from agent import corpus_write_native  # noqa: E402
from agent import entity_focus  # noqa: E402
from agent import score_gap  # noqa: E402
from agent import score_gap_native  # noqa: E402
from agent import temporal_native  # noqa: E402
from agent import workspace_export  # noqa: E402
from agent import workspace_export_native  # noqa: E402
from agent import workspace_import  # noqa: E402
from agent import workspace_import_native  # noqa: E402
from agent.managed_storage import ManagedCorpusBackend  # noqa: E402

TENANT = '11111111-1111-4111-8111-111111111111'
XID = '11111111-1111-4111-8111-111111111111'
SAMPLE = ('## Saturday, August 29, 2026\n\n### 09:15 — Morning\n\n'
          '**Me:** Synthetic note\n\n**Assistant:** Synthetic reply\n\n'
          f'<!-- xid:{XID} -->\n')


class SmokeFailure(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise SmokeFailure(message)


@contextmanager
def selected(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def corpus_write(_temp):
    now = datetime(2026, 8, 29, 9, 15)
    with selected(corpus_write_native.SELECTOR, 'rust'):
        prepared = corpus_write_native.transform(
            'prepare_append', time_token=now.strftime('%H:%M'), topic='Morning',
            me='Synthetic note', assistant='Synthetic reply', xid=XID)
        inspected = corpus_write_native.transform(
            'inspect_edit', current=SAMPLE, xid=XID, new_me='Edited note', new_assistant='Edited reply')
    header = fmt.render_subsection_header(now=now, topic='Morning')
    check(prepared['body'] == fmt.render_exchange('Synthetic note', 'Synthetic reply', XID), 'append body differs')
    check(prepared['sub_header'] == (header[4:] if header.startswith('### ') else header), 'append header differs')
    check(inspected['found'] is True and 'Edited reply' in inspected['edited_text'], 'edit target not found')


def corpus_parse_leaf(_temp):
    text = SAMPLE + SAMPLE.replace('Morning', 'Evening').replace(XID, XID[:-1] + '2')
    with selected('DIARY_CORPUS_PARSE_IMPL', 'python'):
        expected = corpus_parse.parse(text)
    with selected('DIARY_CORPUS_PARSE_IMPL', 'rust'):
        actual = corpus_parse.parse(text)
    check(actual.chunks == expected.chunks and len(actual.chunks) == 2, 'chunks differ')
    check(repr(actual.days) == repr(expected.days), 'days differ')


def score_gap_leaf(_temp):
    rows = [(1, {'score': 0.9}), (2, {'score': 0.2}), (3, {'score': 0.85}), (4, {'score': None})]
    with selected('DIARY_SCORE_GAP_IMPL', 'python'):
        expected = score_gap.cut(rows, 0.3)
    diagnostics = {}
    with selected('DIARY_SCORE_GAP_IMPL', 'rust'):
        actual = score_gap.cut(rows, 0.3, diagnostics=diagnostics)
    check(diagnostics.get('status') == 'ok', 'score-gap leaf unavailable')
    check(actual == expected and len(actual[1]) == 1, 'score-gap partition differs')


def temporal(_temp):
    rows = [{'day': '2030-01-20', 'header': 'a', 'text': 'a', 'score': .9},
            {'day': '2030-01-21', 'header': 'b', 'text': 'b', 'score': .8}]
    with selected('DIARY_TEMPORAL_FOCUS_IMPL', 'rust'):
        anchored = 0
        for query in ('As of January 20, 2030, what had I written?', 'Before 2030-02-10, what did I note?',
                      'No dated question.'):
            expected = entity_focus.time_anchor(query)
            check(temporal_native.anchor(query) == expected, 'anchor differs')
            anchored += expected is not None
        check(anchored == 2, 'synthetic queries no longer anchor')
        kept, removed = temporal_native.cutoff((date(2030, 1, 20), True), rows)
        check(temporal_native.cutoff(None, rows) == ([1, 2], []), 'unanchored cutoff differs')
    check(kept == [1] and removed == [2], 'anchored cutoff differs')


def _members(body):
    with zipfile.ZipFile(io.BytesIO(body)) as bundle:
        return {name: bundle.read(name) for name in bundle.namelist() if name != 'manifest.json'}


def _backend(root):
    backend = ManagedCorpusBackend(root, TENANT)
    backend.activate({'note.md': b'# Synthetic\r\n', 'assets/bin.dat': b'\x00\xff'},
                     {'entry_layout': 'daily'}, ['empty/nested'])
    return backend


def workspace_export_leaf(temp):
    backend = _backend(temp / 'export')
    with selected('WORKSPACE_EXPORT_IMPL', 'python'):
        expected = workspace_export.archive(backend, '', backend.settings())
    with selected('WORKSPACE_EXPORT_IMPL', 'rust'):
        check(workspace_export_native.selected(), 'export selector not honoured')
        actual = workspace_export.archive(backend, '', backend.settings())
    check(_members(actual) == _members(expected), 'export members differ')


def workspace_import_apply(temp):
    with selected('WORKSPACE_EXPORT_IMPL', 'python'):
        source = _backend(temp / 'import-source')
        body = workspace_export.archive(source, '', source.settings())
    reviewed = workspace_import.fingerprint(body, 'Imports/Copy')
    results, rows = {}, {}
    for impl in ('python', 'rust'):
        target = _backend(temp / f'import-{impl}')
        with selected('WORKSPACE_IMPORT_APPLY_IMPL', impl):
            results[impl] = workspace_import.apply(target, body, 'Copy', reviewed)
        with target.db() as db:
            rows[impl] = ([tuple(r) for r in db.execute('SELECT path,data,version FROM files ORDER BY path')],
                          [tuple(r) for r in db.execute('SELECT * FROM directories ORDER BY 1')],
                          [tuple(r) for r in db.execute('SELECT * FROM workspace_index_outbox ORDER BY 1')],
                          [tuple(r) for r in db.execute('SELECT * FROM workspace_imports ORDER BY 1')])
    check(results['rust'] == results['python'], 'import result differs')
    check(rows['rust'] == rows['python'], 'imported rows differ')


# Leaf name (the installed binary's name) -> adapter attribute holding its fixed path, and the check.
LEAVES = {
    'diary-corpus-write-cli': (corpus_write_native, 'BINARY', corpus_write),
    'diary-corpus-parse': (corpus_parse, 'DEFAULT_BIN', corpus_parse_leaf),
    'diary-score-gap': (score_gap_native, 'BINARY', score_gap_leaf),
    'diary-temporal-cli': (temporal_native, 'DEFAULT_BIN', temporal),
    'diary-workspace-export': (workspace_export_native, 'BINARY', workspace_export_leaf),
    'diary-workspace-import-apply': (workspace_import_native, 'DEFAULT_BIN', workspace_import_apply),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('leaves', nargs='*', help='binary names to check; see --all')
    parser.add_argument('--all', action='store_true', help='check every leaf')
    parser.add_argument('--bin-dir', type=Path, help='local validation only; images use the fixed installed paths')
    args = parser.parse_args(argv)
    names = list(LEAVES) if args.all else args.leaves
    unknown = [n for n in names if n not in LEAVES]
    if not names or unknown:
        parser.error(f'name leaves from {", ".join(LEAVES)} or pass --all; unknown: {", ".join(unknown)}')
    failed = []
    with tempfile.TemporaryDirectory(prefix='diary-leaf-smoke-') as directory:
        temp = Path(directory)
        for name in names:
            module, attribute, run = LEAVES[name]
            if args.bin_dir is not None:
                setattr(module, attribute, str((args.bin_dir / name).resolve()))
            path = getattr(module, attribute)
            try:
                check(os.path.isfile(path) and os.access(path, os.X_OK), f'{path} is not an executable file')
                run(temp)
            except Exception as exc:  # any failure, including an unavailable leaf, fails the smoke
                failed.append(name)
                print(f'{name}: FAILED ({type(exc).__name__}: {exc})', file=sys.stderr)
            else:
                print(f'{name}: ok')
    if failed:
        print(f'diary leaf smoke failed: {" ".join(failed)}', file=sys.stderr)
        return 1
    print('diary leaf smoke ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
