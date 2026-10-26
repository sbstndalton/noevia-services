"""Image/CI smoke: the installed decoder must speak decode-v1 end to end.

Synthetic bytes only. Selection changes are confined to this short-lived process.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import workspace_decode as decoder
from agent import workspace_restore as restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, help='Local validation only; images use the fixed installed path')
    args = parser.parse_args()
    if args.binary is not None:
        decoder.DEFAULT_BIN = str(args.binary.resolve())
    source = b'# Synthetic decode capability\r\n\x00\xff'
    settings = {'capability': 'decode-v1', 'unicode': '日記', 'nested': [None, False, 17]}
    manifest = {'format': 'noevia-workspace-v1', 'settings': settings,
                'directories': ['empty'], 'fileCount': 1, 'bytes': len(source),
                'files': [{'path': 'note.md', 'bytes': len(source),
                           'sha256': hashlib.sha256(source).hexdigest()}]}
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False).encode())
        bundle.writestr('workspace/', b'')
        bundle.writestr('workspace/empty/', b'')
        bundle.writestr('workspace/note.md', source)
    os.environ['WORKSPACE_RESTORE_VERIFY_IMPL'] = 'rust'

    def forbidden_python_parser(_):
        raise RuntimeError('Python archive parser must not run in native smoke')
    restore._verify = forbidden_python_parser
    files, directories, actual = restore.verify(out.getvalue())
    if files != {'note.md': source} or directories != {'empty'} or actual['settings'] != settings:
        raise RuntimeError('decode-v1 smoke content mismatch')
    try:
        restore.verify(b'synthetic invalid archive')
    except decoder.DecodeUnavailable:
        raise RuntimeError('decode-v1 unavailable') from None
    except ValueError as exc:
        if str(exc) != restore.INVALID:
            raise RuntimeError('decode-v1 refusal mismatch') from None
    else:
        raise RuntimeError('decode-v1 accepted invalid archive')
    print('workspace decode-v1 smoke ok')


if __name__ == '__main__':
    main()
