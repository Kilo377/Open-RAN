"""Matched native MU on/off diagnostics, without tuning the baseline algorithm.

Uses real RT and NR feedback. A low BLER or higher throughput is not an
acceptance condition: the invariant is matched exogenous channels/configuration
with no xApp active, plus successful execution and explicit denominators.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Validation outputs must stay under project outputs/')
    if args.slots <= 100:
        raise ValueError('Leave an evaluation interval after the 50 ms warmup')
    output.mkdir(parents=True, exist_ok=False)
    base = json.loads((ROOT / 'configs/rt_sys/default.json').read_text())
    cases = []
    for seed in args.seeds:
        matched = []
        for enabled in (True, False):
            mode = 'mu_on' if enabled else 'mu_off'
            config = deepcopy(base)
            config['simulation']['scheduler_enable_spatial_multiplexing'] = enabled
            config_path = output / f'{mode}.json'
            config_path.write_text(json.dumps(config, indent=2))
            target = output / f'{mode}_seed{seed}'
            command = [sys.executable, '-X', 'faulthandler', str(ROOT / 'experiments/run_rt_sys.py'),
                       '--platform', args.platform, '--scene', 'munich', '--cells', '3', '--ues', '12',
                       '--slots', str(args.slots), '--seed', str(seed), '--ue-speed-mps', '1.5',
                       '--path-update-ms', '0.5', '--control-mode', 'baseline', '--live-xapp-test',
                       '--xapp-test-start', 'off', '--warmup-ms', '50', '--recording-mode', 'experiment',
                       '--config', str(config_path), '--output', str(target)]
            with (output / f'{mode}_seed{seed}.log').open('wb') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            status = json.loads((target / 'run_status.json').read_text())
            snapshot = json.loads((target / 'config_snapshot.json').read_text())
            summary = json.loads((target / 'final_summary.json').read_text())
            assert status['completed']
            assert all(not c.get('active_cell_ids') for c in summary['xapp_test']['catalog'])
            matched.append((status, snapshot))
            case = dict(seed=seed, mode=mode, output=str(target.relative_to(ROOT)),
                        cache_key=status['cache_key'],
                        steady_state=summary['steady_state']['per_ue'],
                        link_statistics=summary['link_statistics'])
            cases.append(case)
            (output / 'progress.json').write_text(json.dumps(dict(cases=cases), indent=2))
        on, off = matched
        assert on[0]['cache_key'] == off[0]['cache_key'], 'Channel realizations differ'
        assert on[1]['manifest'] == off[1]['manifest']
        on_config, off_config = deepcopy(on[1]['configs']), deepcopy(off[1]['configs'])
        on_config['sim'].pop('scheduler_enable_spatial_multiplexing')
        off_config['sim'].pop('scheduler_enable_spatial_multiplexing')
        assert on_config == off_config, 'An unrelated network parameter changed'
    result = dict(completed=True, matched_channel_checks_passed=True, cases=cases,
                  interpretation='Native algorithm diagnostic, not a capacity or target-BLER certificate; '
                  'MU remains enabled by default and no algorithm parameter was tuned.')
    (output / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(completed=True, cases=len(cases), output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['dgx', 'agx'], required=True)
    parser.add_argument('--slots', type=int, default=400)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
