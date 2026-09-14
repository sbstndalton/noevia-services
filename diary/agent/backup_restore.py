"""Verify a downloaded immutable manifest and restore into a NEW directory only.

Run: python -m agent.backup_restore BACKUP_ROOT MANIFEST_JSON NEW_DIRECTORY
BACKUP_ROOT contains the downloaded WebDAV tree (including corpusRoot).
Never points at a live corpus, chooses a latest manifest, or overwrites files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid
from .managed_storage import safe_key


def restore(root, manifest_path, destination):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError('Restore requires a new directory; existing data will not be overwritten')
    body = manifest_path.read_bytes()
    if len(body) > 8 * 1024 * 1024 or manifest_path.stem != hashlib.sha256(body).hexdigest():
        raise ValueError('Manifest checksum does not match its immutable filename')
    manifest = json.loads(body)
    if manifest.get('format') != 'noevia-diary-backup-v1':
        raise ValueError('Unsupported backup manifest')
    tenant = str(uuid.UUID(manifest['tenant']))
    files = manifest['files']
    if not isinstance(files, list) or len(files) > 50000:
        raise ValueError('Invalid backup file list')
    verified = {}
    total = 0
    for row in files:
        relative = safe_key(row['path'])
        object_key = safe_key(row['object'])
        if f'/noevia-backups/{tenant}/objects/' not in '/' + object_key:
            raise ValueError('Backup object has the wrong tenant')
        source = (root / object_key).resolve()
        if root not in source.parents or relative in verified:
            raise ValueError('Backup escaped its root or duplicated a path')
        data = source.read_bytes()
        total += len(data)
        if total > 1024 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Backup object checksum mismatch or restore exceeds 1 GiB')
        verified[relative] = data
    directories = [safe_key(path) for path in manifest.get('directories', [])]
    # Validate the namespace before creating anything at the destination.
    all_paths = [*verified, *directories]
    for path in all_paths:
        if any(str(parent) in verified for parent in Path(path).parents if str(parent) != '.'):
            raise ValueError('Backup has overlapping file and directory paths')
    if any(path in verified for path in directories):
        raise ValueError('Backup has overlapping file and directory paths')
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    for directory in directories:
        (destination / directory).mkdir(mode=0o700, parents=True, exist_ok=True)
    for relative, data in verified.items():
        target = destination / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    return {'files': len(verified), 'bytes': total, 'tenant': tenant, 'settings': manifest.get('settings', {})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backup_root')
    parser.add_argument('manifest')
    parser.add_argument('new_directory')
    args = parser.parse_args()
    print(json.dumps(restore(args.backup_root, args.manifest, args.new_directory), indent=2))
