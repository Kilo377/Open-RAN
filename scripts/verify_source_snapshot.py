"""Read-only check of the original files recorded at copy time."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    source=Path(manifest['source'])
    if not source.is_dir():
        sys.exit('Original source unavailable on this host; retained manifest is still portable')
    changed=[]
    for entry in manifest['files']:
        path=source/entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            changed.append(entry['path'])
    print(json.dumps({'checked_files':len(manifest['files']),'changed_files':changed},indent=2))
    sys.exit(bool(changed))
