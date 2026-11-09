"""Matched six-UE, 400-slot OFF/energy/DRX+MEC cost and accounting check.

Run from the accepted AGX GPU environment. Channel caches are warmed once;
order is reversed in repeat two. Every measured case exports its normal HTML.
"""
import argparse
import gzip
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--slots', type=int, default=400)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'outputs') or output.exists():
        raise ValueError('Choose a fresh output directory inside this project')
    output.mkdir(parents=True)
    cache = ROOT / 'channel_cache' / output.name
    receipt = dict(slots=args.slots, seed=1, completed=False, cases=[])

    def write():
        temporary = output / 'acceptance.tmp'
        temporary.write_text(json.dumps(receipt, indent=2))
        temporary.replace(output / 'acceptance.json')

    def run(name, mode, extra=()):
        out = output / name
        command = ['bash', 'scripts/orin/run-six-ue-cuda-nr.sh', '--model', 'fast',
                   '--slots', str(args.slots), '--seed', '1', '--ue-services', mode,
                   '--record-compression', 'gzip', '--channel-cache-root', str(cache),
                   '--output', str(out), *extra]
        before = time.perf_counter()
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            receipt['failure'] = dict(case=name, returncode=result.returncode)
            write()
            raise RuntimeError('Run failed: ' + name)
        return out, time.perf_counter() - before

    write()
    run('warmup', 'off', ['--generate-only'])
    for repeat, order in enumerate([['off', 'energy', 'drx-mec'], ['drx-mec', 'energy', 'off']], 1):
        for mode in order:
            receipt['running'] = f'{mode}_{repeat}'
            write()
            out, wall = run(receipt['running'], mode)
            status = load(out / 'run_status.json')
            perf = load(out / 'performance.json')
            summary = load(out / 'final_summary.json')
            assert status['completed'] and status['replay']['audits_passed']
            services = summary.get('ue_services', {})
            if mode != 'off':
                assert services['enabled']
                for uid, report in services['per_ue'].items():
                    e = services['energy_components'][uid]
                    assert abs(e['total_j'] - sum(e.get(k, 0) for k in ['radio_j', 'compute_j', 'transition_j'])) < 1e-9
                    task = report['ue_task_metrics']
                    if task:
                        assert task['generated'] == task['completed'] + task['failed'] + task['pending']
            receipt['cases'].append(dict(mode=mode, repeat=repeat, output=str(out),
                source_sha256=status['source_sha256'], simulation_wall_s=perf['simulation_s'],
                internal_wall_s=status['elapsed_s'], external_wall_s=wall,
                module_wall_s=services.get('module_wall_s', 0),
                peak_rss_bytes=status['peak_rss_bytes'], peak_cuda_bytes=status['peak_cuda_bytes'],
                output_bytes=status['output_bytes'], actual_decodes=perf['actual_decodes'],
                html_audits_passed=status['replay']['audits_passed']))
            write()
    assert len({c['source_sha256'] for c in receipt['cases']}) == 1
    for repeat in [1, 2]:
        off = load(output / f'off_{repeat}' / 'final_summary.json')
        energy = load(output / f'energy_{repeat}' / 'final_summary.json')
        assert off['packet_metrics'] == energy['packet_metrics'], 'Energy-only changed packet outcomes'
        for file in ['packet_events.jsonl.gz', 'harq_events.jsonl.gz']:
            with gzip.open(output / f'off_{repeat}' / file, 'rt') as a, gzip.open(output / f'energy_{repeat}' / file, 'rt') as b:
                assert [json.loads(s) for s in a] == [json.loads(s) for s in b], 'Energy-only changed ' + file
    receipt['energy_only_packet_and_harq_equal'] = True
    receipt['means'] = {mode: {key: statistics.mean(c[key] for c in receipt['cases'] if c['mode'] == mode)
                               for key in ['simulation_wall_s', 'internal_wall_s', 'module_wall_s', 'peak_rss_bytes', 'output_bytes']}
                        for mode in ['off', 'energy', 'drx-mec']}
    base = receipt['means']['off']
    receipt['simulation_ratios'] = {mode: receipt['means'][mode]['simulation_wall_s'] / base['simulation_wall_s']
                                    for mode in ['energy', 'drx-mec']}
    receipt['completed'] = True
    receipt.pop('running', None)
    write()
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
