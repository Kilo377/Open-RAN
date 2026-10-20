#!/usr/bin/env python3
"""Start the prepared lab: existing 5GC, link monitor and pinned B210 gNB."""
import argparse
import json
import os
import pathlib
import pwd
import subprocess
import sys
import time

import labctl

CORE = ('mysql', 'ims', 'oai-nrf', 'oai-udr', 'oai-udm', 'oai-ausf',
        'oai-amf', 'oai-smf', 'oai-upf', 'oai-ext-dn')


def core_ready():
    result = labctl.run(['docker', 'inspect', *CORE], check=False)
    if result.returncode:
        return False
    return all(item['State']['Running'] and
               item['State'].get('Health', {}).get('Status', 'healthy') == 'healthy'
               for item in json.loads(result.stdout))


def preflight(state, core):
    for name in ('legacy.json', 'prepared.json', 'compose.env', 'gnb.conf'):
        if not (state / name).is_file():
            raise RuntimeError(f'Missing {state / name}; complete labctl.py prepare first.')
    if not (state / 'uhd-images').is_dir():
        raise RuntimeError('Prepared UHD image directory is missing.')
    files = [core / name for name in ('compose.yaml', 'compose.yml',
                                      'docker-compose.yaml', 'docker-compose.yml')]
    compose_file = next((path for path in files if path.is_file()), None)
    if compose_file is None:
        raise RuntimeError(f'Existing 5GC Compose file not found in {core}.')
    prepared = json.loads((state / 'prepared.json').read_text())
    if prepared['image'] != labctl.LOCK['image']:
        raise RuntimeError('Prepared image does not match the repository lockfile.')
    labctl.run(['docker', 'image', 'inspect', labctl.LOCK['image']])
    labctl.compose(state, 'config', '--quiet')
    if labctl.run(['systemctl', 'show', 'oai-link-monitor.service', '-p', 'LoadState',
                   '--value']).stdout.strip() != 'loaded':
        raise RuntimeError('This startup requires the existing oai-link-monitor.service.')
    info = labctl.container_info()
    if info and info['State']['Running']:
        mounts = {item['Destination']: item['Source'] for item in info['Mounts']}
        if (info['Config']['Image'] != labctl.LOCK['image'] or
                mounts.get('/opt/oai-gnb/etc/gnb.conf') != str(state / 'gnb.conf')):
            raise RuntimeError('Running gNB does not match the pinned image and state directory.')
        legacy = json.loads((state / 'legacy.json').read_text())
        if labctl.active(legacy['unit']):
            raise RuntimeError('Both old and new gNB are active; resolve the SDR ownership first.')
    return compose_file


def main():
    user = os.environ.get('SUDO_USER')
    home = pathlib.Path(pwd.getpwnam(user).pw_dir) if user else pathlib.Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=pathlib.Path, default=home / 'oai-gnb-next')
    parser.add_argument('--core-dir', type=pathlib.Path, default=home / 'oai-cn5g')
    parser.add_argument('--monitor-url', default='http://127.0.0.1:18765')
    parser.add_argument('--check', action='store_true', help='Read-only prerequisite check.')
    args = parser.parse_args()
    state, core = args.state_dir.expanduser().resolve(), args.core_dir.expanduser().resolve()
    if not args.check:
        labctl.require_root()
        labctl.run(['systemctl', 'start', 'docker.service'])
    compose_file = preflight(state, core)
    if args.check:
        print('Prerequisites OK. Existing services and radio were not changed.')
        return
    print('Starting the existing 5GC and waiting for healthy containers...', flush=True)
    # Retain the locally installed core images and existing container definitions.
    labctl.run(['docker', 'compose', '-f', compose_file, 'up', '-d', '--no-recreate',
                '--pull', 'never'], timeout=120)
    deadline = time.monotonic() + 120
    while not core_ready():
        if time.monotonic() >= deadline:
            raise RuntimeError('5GC is not healthy; inspect docker compose ps and logs in --core-dir.')
        time.sleep(2)
    labctl.run(['systemctl', 'start', 'oai-link-monitor.service'])
    print('Waiting for the AGX monitor agent; start oai-link-agent.service on AGX if needed.', flush=True)
    deadline = time.monotonic() + 60
    while True:
        try:
            status = labctl.monitor_request(args.monitor_url, '/api/state')
            if status['agent']['online'] and status['agent']['identity'].get('module_ready'):
                if (status.get('run') or {}).get('running'):
                    raise RuntimeError('A UI test is already active; stop it before startup validation.')
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('AGX/520NGL not ready. Check the AGX agent and UE registration, then rerun startup.')
        time.sleep(2)
    info = labctl.container_info()
    if not info or not info['State']['Running']:
        labctl.activate(state)
    else:
        # A retained container may have been restarted independently after a reboot.
        labctl.save(state / 'current.json', {'mode': 'startup-validation', 'container_id': info['Id'],
                     'image': labctl.LOCK['image'], 'config_sha256': labctl.sha(state / 'gnb.conf'),
                     'epoch': time.time()})
    labctl.smoke_test(state, args.monitor_url, 35)
    labctl.accept(state)
    print('Startup verified: OAI 2026.w42, 40 MHz, normal gain, module-bound payload valid.')
    print('Open the UI on the host desktop: http://127.0.0.1:18765/')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print('ERROR:', str(exc), file=sys.stderr)
        sys.exit(1)
