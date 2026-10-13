"""Wait for atomic Spark publication, copy it to Mac, and verify every SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--host-key-alias', required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--timeout-hours', type=float, default=24)
    args = parser.parse_args()
    relative = 'outputs/rt_research_readiness_20261013/final_ran'
    remote_path = args.remote_root.rstrip('/') + '/' + relative
    target = ROOT / relative
    receipt = ROOT / 'outputs/rt_research_readiness_20261013/final_ran_receive_status.json'
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
           '-o', f'HostKeyAlias={args.host_key_alias}', '-o', 'StrictHostKeyChecking=yes']
    deadline = time.monotonic() + args.timeout_hours * 3600
    receipt.parent.mkdir(parents=True, exist_ok=True)
    while time.monotonic() < deadline:
        try:
            check = subprocess.run(ssh + [args.remote, 'test -f ' + shlex.quote(remote_path + '/data_manifest.json')],
                                   capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            check = subprocess.CompletedProcess(ssh, 124, '', 'SSH status check timed out; retrying')
        if check.returncode == 0:
            break
        receipt.write_text(json.dumps(dict(stage='waiting_for_spark_publication',
            checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            last_ssh_returncode=check.returncode, detail=check.stderr[-1000:]), indent=2))
        time.sleep(30)
    else:
        raise TimeoutError('Spark publication not available; reconnect and rerun this receiver')
    # The remote source is immutable after publication; interrupted rsync is resumable.
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-az', '-e', shlex.join(ssh),
                    args.remote + ':' + remote_path + '/', str(target) + '/'], check=True)
    manifest = json.loads((target / 'data_manifest.json').read_text())
    for entry in manifest['files']:
        path = (target / entry['path']).resolve()
        if not path.is_relative_to(target.resolve()):
            raise ValueError('Manifest path escapes evidence directory')
        assert path.stat().st_size == entry['bytes'], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
    status = dict(stage='saved_and_sha256_verified', passed=manifest['passed'],
                  file_count=manifest['file_count'], total_bytes=manifest['total_bytes'],
                  source_sha256=manifest['source_sha256'],
                  completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    receipt.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
