"""Wait for a frozen validation run, audit it, and publish immutable evidence.

This does not start/restart experiments or mark an incomplete matrix as accepted.
The published manifest is also usable by a disconnected client after reconnecting.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f'/proc/{pid}/stat')
    return not (stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--publish-root', type=Path, required=True)
    parser.add_argument('--timeout-hours', type=float, default=24)
    args = parser.parse_args()
    source = source_fingerprint(ROOT)
    run = ROOT / 'outputs/rt_research_readiness_20261013/final_ran'
    destination_root = args.publish_root.resolve()
    destination = destination_root / 'outputs/rt_research_readiness_20261013/final_ran'
    if destination.exists():
        raise FileExistsError(f'Refusing to replace saved evidence: {destination}')
    if source_fingerprint(destination_root) != source:
        raise ValueError('Published checkout differs from the frozen executing source')
    deadline = time.monotonic() + args.timeout_hours * 3600
    while process_alive(args.wait_pid) and time.monotonic() < deadline:
        time.sleep(30)
    if process_alive(args.wait_pid):
        raise TimeoutError('Run still active; keep live outputs untouched, do not publish a final snapshot')
    if source_fingerprint(ROOT) != source:
        raise ValueError('Executing source changed during validation')
    with (run / 'finalization_audit.log').open('w') as log:
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/validation/summarize_research_readiness.py')],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status = dict(passed=result.returncode == 0, audit_returncode=result.returncode,
                  source_sha256=source, executing_root=str(ROOT), published_root=str(destination_root),
                  completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  scope='Final acceptance requires passed=true; failed/incomplete evidence is preserved explicitly')
    (run / 'publication_status.json').write_text(json.dumps(status, indent=2))
    files = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and path.name != 'data_manifest.json':
            files.append(dict(path=str(path.relative_to(run)), bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(source_sha256=source, passed=status['passed'], files=files,
                    file_count=len(files), total_bytes=sum(p['bytes'] for p in files))
    (run / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name('final_ran_publishing')
    if staging.exists():
        raise FileExistsError(f'Unfinished publication exists: {staging}')
    shutil.copytree(run, staging)
    for entry in files:
        path = staging / entry['path']
        assert path.stat().st_size == entry['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    staging.rename(destination)
    print(json.dumps(dict(**status, file_count=len(files), total_bytes=manifest['total_bytes'])), flush=True)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
