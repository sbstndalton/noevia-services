"""Preview a portable ZIP; restore only to a new directory, never a live corpus.

python -m agent.workspace_restore ARCHIVE.zip
python -m agent.workspace_restore ARCHIVE.zip --destination NEW --fingerprint REVIEWED_SHA256
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import zipfile
from .managed_storage import safe_key
from .workspace_export import MAX_ARCHIVE, MAX_FILE, MAX_TOTAL


def verify(body):
    try:
        return _verify(body)
    except (zipfile.BadZipFile, KeyError, TypeError, AttributeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid workspace archive or manifest") from exc


def _verify(body):
    if len(body) > MAX_ARCHIVE:
        raise ValueError('Archive exceeds 272 MiB')
    with zipfile.ZipFile(io.BytesIO(body)) as bundle:
        entries = bundle.infolist()
        if len(entries) > 10002 or len({e.filename for e in entries}) != len(entries):
            raise ValueError('Too many archive entries or duplicate paths')
        total = 0
        for entry in entries:
            safe_key(entry.filename.rstrip('/'))
            mode = entry.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
                raise ValueError('Links, special files and encrypted entries are unsupported')
            if entry.file_size > MAX_FILE or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError('Archive entry exceeds limits or uses unsupported compression')
            if entry.is_dir() and entry.file_size != 0:
                raise ValueError('Directory entries must be empty')
            if (stat.S_IFMT(mode) == stat.S_IFDIR and not entry.is_dir()) or (stat.S_IFMT(mode) == stat.S_IFREG and entry.is_dir()):
                raise ValueError('Archive entry type does not match its path')
            total += entry.file_size
        if total > MAX_TOTAL + 8 * 1024 * 1024:
            raise ValueError('Expanded archive exceeds limits')
        if bundle.getinfo('manifest.json').file_size > 8 * 1024 * 1024:
            raise ValueError('Manifest exceeds 8 MiB')
        manifest = json.loads(bundle.read('manifest.json'))
        if not isinstance(manifest, dict):
            raise ValueError('Invalid manifest object')
        if manifest.get('format') != 'noevia-workspace-v1':
            raise ValueError('Unsupported workspace format')
        rows, directories = manifest.get('files'), manifest.get('directories')
        if not isinstance(rows, list) or len(rows) > 5000 or not isinstance(directories, list) or len(directories) > 5000:
            raise ValueError('Invalid manifest counts')
        dirs = set()
        for directory in directories:
            safe_key(directory)
            if directory in dirs:
                raise ValueError('Duplicate directory')
            dirs.add(directory)
        files = {}
        for row in rows:
            if not isinstance(row, dict) or type(row.get('bytes')) is not int or row['bytes'] < 0 or not isinstance(row.get('sha256'), str):
                raise ValueError('Invalid manifest file record')
            path = safe_key(row['path'])
            if path in files or path in dirs:
                raise ValueError('Duplicate or overlapping path')
            data = bundle.read('workspace/' + path)
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Source checksum or size mismatch')
            files[path] = data
        for path in [*files, *dirs]:
            if any(str(parent) in files for parent in Path(path).parents if str(parent) != '.'):
                raise ValueError('File overlaps a parent directory')
        expected = {'manifest.json', 'workspace/'} | {'workspace/' + p for p in files} | {'workspace/' + p + '/' for p in dirs}
        if {e.filename for e in entries} != expected:
            raise ValueError('Archive contains missing or unlisted entries')
        total = sum(map(len, files.values()))
        if type(manifest.get('fileCount')) is not int or type(manifest.get('bytes')) is not int:
            raise ValueError('Invalid manifest totals')
        if total > MAX_TOTAL or manifest.get('fileCount') != len(files) or manifest.get('bytes') != total:
            raise ValueError('Manifest totals mismatch')
    return files, dirs, manifest


def preview(body):
    files, dirs, manifest = verify(body)
    return {'fingerprint': hashlib.sha256(body).hexdigest(), 'fileCount': len(files),
            'bytes': manifest['bytes'], 'files': sorted(files), 'directories': sorted(dirs),
            'mode': 'new-directory-only', 'settings': manifest.get('settings', {})}


def restore(body, destination, fingerprint):
    if hashlib.sha256(body).hexdigest() != fingerprint:
        raise ValueError('Archive changed. Review a fresh preview.')
    files, dirs, manifest = verify(body)
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Restore requires a new directory; existing data will not be overwritten')
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    try:
        for directory in sorted(dirs):
            (destination / directory).mkdir(mode=0o700, parents=True, exist_ok=True)
        for path, data in files.items():
            target = destination / path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as output:
                os.chmod(target, 0o600)
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
    except BaseException:
        shutil.rmtree(destination)
        raise
    return {'fileCount': len(files), 'bytes': manifest['bytes'], 'settings': manifest.get('settings', {})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--destination')
    parser.add_argument('--fingerprint')
    args = parser.parse_args()
    if bool(args.destination) != bool(args.fingerprint):
        parser.error('--destination requires the reviewed --fingerprint')
    with open(args.archive, 'rb') as source:
        body = source.read(272 * 1024 * 1024 + 1)
    print(json.dumps(restore(body, args.destination, args.fingerprint) if args.destination else preview(body), indent=2))
