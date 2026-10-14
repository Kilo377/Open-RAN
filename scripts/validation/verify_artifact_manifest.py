"""Create or verify a portable SHA256 inventory without changing result files."""
import argparse
import hashlib
import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(root, manifest_path, receipt, strict=False):
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text())
    expected = set()
    total = 0
    for entry in manifest['files']:
        relative = entry['path']
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or relative in expected:
            raise ValueError(f'Unsafe or duplicate manifest path: {relative}')
        expected.add(relative)
        assert file.is_file(), relative
        assert file.stat().st_size == entry['bytes'], relative
        assert sha256(file) == entry['sha256'], relative
        total += entry['bytes']
    if strict:
        actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if manifest_path.resolve().is_relative_to(root):
            expected.add(str(manifest_path.resolve().relative_to(root)))
        assert actual == expected, dict(unlisted=sorted(actual - expected), missing=sorted(expected - actual))
    result = dict(verified=True, count=len(manifest['files']), bytes=total,
                  source_sha256=manifest.get('source_sha256'), manifest_sha256=sha256(manifest_path),
                  root=str(root), host=socket.gethostname(), system=platform.system(), machine=platform.machine(),
                  verified_at=datetime.now(timezone.utc).isoformat(), strict_inventory=strict)
    if receipt:
        assert not receipt.resolve().is_relative_to(root), 'Receipt must stay outside the immutable result inventory'
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--source-sha')
    parser.add_argument('--strict-inventory', action='store_true')
    args = parser.parse_args()
    if args.create:
        assert not args.manifest.exists(), 'Refusing to overwrite an existing manifest'
        entries = [dict(path=str(p.relative_to(args.root)), bytes=p.stat().st_size, sha256=sha256(p))
                   for p in sorted(args.root.rglob('*')) if p.is_file()]
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(dict(source_sha256=args.source_sha, files=entries), indent=2) + '\n')
    verify(args.root, args.manifest, args.receipt, args.strict_inventory)
