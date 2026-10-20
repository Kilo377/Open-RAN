#!/usr/bin/env python3
"""Isolated OAI gNB lifecycle. The existing 5GC and subscriber DB are untouched."""
import argparse
import hashlib
import json
import os
import pathlib
import pwd
import re
import subprocess
import sys
import time
import urllib.request

LAB = pathlib.Path(__file__).resolve().parents[1]
LOCK = json.loads((LAB / 'upstream.lock.json').read_text())
CONTAINER = 'oai-gnb-next'
RECOVERY = 'oai-gnb-next-recovery'


def run(args, check=True, timeout=90):
    p = subprocess.run([str(x) for x in args], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f'{args[0]} failed ({p.returncode}): {p.stderr.strip()[:600]}')
    return p


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_root():
    if os.geteuid():
        raise RuntimeError('Use sudo for activate, accept or restore; these commands switch the SDR owner.')


def active(unit):
    return run(['systemctl', 'is-active', unit], check=False).stdout.strip() == 'active'


def container_info():
    p = run(['docker', 'inspect', CONTAINER], check=False)
    return json.loads(p.stdout)[0] if p.returncode == 0 else None


def compose(state, *args):
    return run(['docker', 'compose', '--project-name', CONTAINER, '--env-file', state / 'compose.env',
                '-f', LAB / 'compose.yaml', *args], timeout=120)


def gnb_logs():
    p = run(['docker', 'logs', '--tail', '3000', CONTAINER], check=False)
    return p.stdout + '\n' + p.stderr


def gates(text):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    gain = re.findall(r'USRP TX_GAIN:\s*([\d.]+).*?tx_gain:\s*([\d.]+)', text)
    return {
        'ng_setup': 'Received NGSetupResponse from AMF' in text,
        'rf_ready': 'rf device ready' in text,
        '40MHz': bool(re.search(r'Actual TX bandwidth:\s*40\.0+M', text)),
        '106_PRB': bool(re.search(r'N_RB_DL\s*[= ]\s*106\b', text)),
        'normal_TX_gain': bool(gain and abs(float(gain[-1][0])-77.75) < .02
                               and abs(float(gain[-1][1])-12) < .02),
    }


def prepare(state, legacy_unit):
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+\.service', legacy_unit or ''):
        raise RuntimeError('Specify the existing gNB unit with --legacy-unit NAME.service.')
    if not active(legacy_unit):
        raise RuntimeError('The legacy gNB must be active during prepare so its exact launch can be saved.')
    state.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'src', 'uhd-images'):
        (state / name).mkdir(exist_ok=True)
    pid = int(run(['systemctl', 'show', legacy_unit, '-p', 'MainPID', '--value']).stdout)
    proc = pathlib.Path('/proc') / str(pid)
    argv = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
    if not argv or pathlib.Path(argv[0]).name != 'nr-softmodem':
        raise RuntimeError('Legacy unit does not directly launch nr-softmodem; inspect its startup first.')
    cwd = run(['systemctl', 'show', legacy_unit, '-p', 'WorkingDirectory', '--value']).stdout.strip()
    legacy = {'unit': legacy_unit, 'argv': argv, 'cwd': cwd or str(pathlib.Path(argv[0]).parent),
              'saved_epoch': time.time()}
    if not (state / 'legacy.json').exists():
        save(state / 'legacy.json', legacy)
    elif json.loads((state / 'legacy.json').read_text())['argv'] != argv:
        raise RuntimeError('Existing legacy snapshot differs; use a new state directory.')
    source = state / 'src/openairinterface5g'
    if not source.exists():
        run(['git', 'clone', '--depth', '1', '--branch', LOCK['tag'], LOCK['repository'], source], timeout=600)
    commit = run(['git', '-C', source, 'rev-parse', 'HEAD']).stdout.strip()
    if commit != LOCK['commit']:
        raise RuntimeError('Upstream source does not match upstream.lock.json.')
    run(['docker', 'pull', LOCK['image']], timeout=600)
    config = state / 'gnb.conf'
    if not config.exists():
        config.write_bytes((LAB / LOCK['configuration']).read_bytes())
    # Download the images belonging to the container UHD, without opening the SDR.
    run(['docker', 'run', '--rm', '--network', 'bridge',
         '--mount', f'type=bind,src={state / "uhd-images"},dst=/uhd-images',
         '--entrypoint', '/opt/oai-gnb/bin/uhd_images_downloader.py', LOCK['image'],
         '-t', 'b2xx', '-i', '/uhd-images'], timeout=300)
    (state / 'compose.env').write_text('GNB_CONFIG=' + json.dumps(str(config)) + '\nUHD_IMAGES='
                                       + json.dumps(str(state / 'uhd-images')) + '\n')
    compose(state, 'config', '--quiet')
    # OAI's --help exits with 255; usable help output is the expected result.
    help_run = run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint',
                    '/opt/oai-gnb/bin/nr-softmodem', LOCK['image'], '--help'], check=False)
    if '--continuous-tx' not in help_run.stdout or '-E:' not in help_run.stdout:
        raise RuntimeError('Container softmodem help did not contain the expected B210 options.')
    save(state / 'prepared.json', {'source_commit': commit, 'image': LOCK['image'],
                                 'config_sha256': sha(config), 'epoch': time.time()})
    print('Prepared isolated 2026.w42 environment. Legacy gNB is still running.', flush=True)


def restore(state):
    require_root()
    legacy = json.loads((state / 'legacy.json').read_text())
    compose(state, 'down', '--timeout', '20')
    unit = legacy['unit']
    if not active(unit):
        # A transient legacy unit may have disappeared when it was stopped.
        started = run(['systemctl', 'start', unit], check=False)
        if started.returncode:
            run(['systemctl', 'reset-failed', unit], check=False)
            run(['systemd-run', '--unit=' + unit.removesuffix('.service'), '--property=Type=exec',
                 '--property=KillSignal=SIGINT', '--property=TimeoutStopSec=20',
                 '--property=WorkingDirectory=' + legacy['cwd'],
                 '--property=StandardOutput=append:' + str(state / 'logs/legacy-restored.log'),
                 '--property=StandardError=append:' + str(state / 'logs/legacy-restored.log'), *legacy['argv']])
    time.sleep(2)
    if not active(unit):
        raise RuntimeError('Legacy gNB did not stay active; inspect its service log.')
    run(['systemctl', 'stop', RECOVERY + '.timer'], check=False)
    save(state / 'current.json', {'mode': 'legacy', 'epoch': time.time()})
    print('Legacy gNB restored; 5GC and SIM were not changed.', flush=True)


def activate(state):
    require_root()
    prepared = json.loads((state / 'prepared.json').read_text())
    if prepared['image'] != LOCK['image']:
        raise RuntimeError('Prepare this lockfile before switching the SDR.')
    if active(RECOVERY + '.timer'):
        raise RuntimeError('A trial is already pending. Run smoke-test and accept, or restore.')
    info = container_info()
    if info and info['State']['Running']:
        raise RuntimeError('The new gNB is already running.')
    legacy = json.loads((state / 'legacy.json').read_text())
    compose(state, 'config', '--quiet')
    run(['systemd-run', '--unit=' + RECOVERY, '--on-active=10min', '--timer-property=AccuracySec=1s',
         sys.executable, pathlib.Path(__file__).resolve(), '--state-dir', state, 'restore'])
    if not active(RECOVERY + '.timer'):
        raise RuntimeError('Independent recovery timer did not arm.')
    try:
        run(['systemctl', 'stop', legacy['unit']])
        if run(['pgrep', '-x', 'nr-softmodem'], check=False).stdout.strip():
            raise RuntimeError('Another nr-softmodem is still running; refusing to share the B210.')
        compose(state, 'up', '--detach')
        for _ in range(75):
            info = container_info()
            if not info or not info['State']['Running']:
                raise RuntimeError('New gNB exited during startup.')
            text = gnb_logs()
            if all(gates(text).values()):
                save(state / 'current.json', {'mode': 'trial', 'container_id': info['Id'],
                     'image': LOCK['image'], 'config_sha256': sha(state / 'gnb.conf'), 'epoch': time.time()})
                print('New gNB ready at 40 MHz / normal gain. Recovery is armed for 10 minutes.', flush=True)
                return
            time.sleep(1)
        raise RuntimeError('New gNB startup checks timed out.')
    except BaseException:
        (state / 'logs/new-gnb-startup.log').write_text(gnb_logs())
        restore(state)
        raise


def monitor_request(base, path, body=None, csrf=None):
    headers = {'Content-Type': 'application/json'}
    if csrf: headers['X-CSRF-Token'] = csrf
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base.rstrip('/') + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.load(response)


def smoke_test(state, base, duration):
    current = json.loads((state / 'current.json').read_text())
    info = container_info()
    if not info or not info['State']['Running'] or current.get('container_id') != info['Id']:
        raise RuntimeError('Activate the new gNB before testing.')
    initial = monitor_request(base, '/api/state')
    identity = initial['agent']['identity']
    if not initial['agent']['online'] or identity['product'] != 'RM520N-GL':
        raise RuntimeError('The existing AGX monitor agent must be online with RM520N-GL.')
    if initial.get('run') and initial['run']['running']:
        monitor_request(base, '/api/stop', {}, initial['csrf']);time.sleep(1)
    result = monitor_request(base, '/api/start', {'duration_s': duration}, initial['csrf'])
    rid = result['run']['id']
    deadline = time.monotonic() + duration + 20
    while time.monotonic() < deadline:
        result = monitor_request(base, '/api/state')
        if result.get('run', {}).get('id') != rid:
            raise RuntimeError('Monitor run changed during validation.')
        if not result['run']['running']: break
        time.sleep(2)
    else:
        monitor_request(base, '/api/stop', {}, result['csrf'])
        raise RuntimeError('Monitor did not finish within its deadline.')
    points = result['run'].get('points', [])
    measured = [p for p in points if p['t'] >= 5]
    receive_windows = [sum(p['delta_bytes'] for p in measured if start <= p['t'] < start+5)
                       for start in range(5, duration-1, 5)]
    checks = gates(gnb_logs())
    checks.update({'module_identity': identity['product'] == 'RM520N-GL',
                   'module_socket_binding': bool(points) and all(p['bound_device'] == identity['interface']
                                                               for p in points if p['delta_bytes'] > 0),
                   'payload_verified': bool(points) and points[-1]['verified_frames'] > 0
                                        and all(p['invalid_frames'] == 0 for p in points),
                   'sustained_receive': bool(measured) and points[-1]['t'] >= duration-1
                                        and bool(receive_windows) and all(n > 0 for n in receive_windows)})
    if not all(checks.values()):
        raise RuntimeError('Smoke test failed: ' + json.dumps(checks))
    throughput = sum(p['delta_bytes'] for p in measured)*8/sum(p['dt'] for p in measured)/1e6
    validation = {'container_id': info['Id'], 'image': LOCK['image'], 'source_commit': LOCK['commit'],
                  'config_sha256': sha(state / 'gnb.conf'), 'checks': checks, 'run_id': rid,
                  'duration_s': points[-1]['t'], 'verified_frames': points[-1]['verified_frames'],
                  'verified_bytes': points[-1]['total_bytes'], 'mean_after_warmup_Mbps': throughput,
                  'five_second_window_bytes': receive_windows,
                  'samples': points, 'epoch': time.time()}
    save(state / 'smoke-test.json', validation)
    print(json.dumps({k:v for k,v in validation.items() if k not in ('samples', 'container_id')}, indent=2), flush=True)


def accept(state):
    require_root()
    v = json.loads((state / 'smoke-test.json').read_text())
    info = container_info()
    if not info or not info['State']['Running'] or v['container_id'] != info['Id']:
        raise RuntimeError('The tested container is not running.')
    if not all(v['checks'].values()) or v['config_sha256'] != sha(state / 'gnb.conf'):
        raise RuntimeError('A passing smoke-test for this configuration is required.')
    run(['systemctl', 'stop', RECOVERY + '.timer'])
    if active(RECOVERY + '.timer'):
        raise RuntimeError('Recovery timer is still active.')
    save(state / 'current.json', {'mode': 'new', 'container_id': info['Id'], 'image': LOCK['image'],
                                'config_sha256': sha(state / 'gnb.conf'), 'epoch': time.time()})
    print('Validated new gNB remains running. Legacy snapshot is available for restore.', flush=True)


def status(state):
    info = container_info()
    legacy_path = state / 'legacy.json'
    legacy = json.loads(legacy_path.read_text()) if legacy_path.exists() else None
    print(json.dumps({'pinned_tag': LOCK['tag'], 'pinned_commit': LOCK['commit'],
                      'new_gnb_running': bool(info and info['State']['Running']),
                      'legacy_running': active(legacy['unit']) if legacy else None,
                      'recovery_timer_active': active(RECOVERY + '.timer'),
                      'state_directory': str(state)}, indent=2))


def main():
    username = os.environ.get('SUDO_USER')
    home = pathlib.Path(pwd.getpwnam(username).pw_dir) if username else pathlib.Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=pathlib.Path, default=home / 'oai-gnb-next')
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('prepare');setup.add_argument('--legacy-unit', required=True)
    for name in ('activate', 'accept', 'restore', 'status'):sub.add_parser(name)
    test = sub.add_parser('smoke-test');test.add_argument('--monitor-url', default='http://127.0.0.1:18765')
    test.add_argument('--duration', type=int, default=35)
    args = parser.parse_args();state = args.state_dir.expanduser().resolve()
    if args.command == 'prepare':prepare(state, args.legacy_unit)
    elif args.command == 'smoke-test':smoke_test(state, args.monitor_url, args.duration)
    else:globals()[args.command](state)


if __name__ == '__main__':
    try:main()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        print('ERROR:', str(exc), file=sys.stderr);sys.exit(1)
