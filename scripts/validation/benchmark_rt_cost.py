"""Serial GPU cost comparison on identical RT deployments and traffic.

Short, prefilled fixtures measure cost only, not steady-state capacity or BLER.
CUDA fences attribute completed work to stages; inclusive timings are nested.
No physical decisions, sample budgets, or random streams are changed.
"""
import argparse
from collections import defaultdict
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    from phy.rt_sys.reference import NrReferenceAbstraction, StatisticalBankAbstraction
    from phy.rt_sys.direct import DirectNrAbstraction
    from phy.rt_sys.channel import RtChannelProvider
    from experiments.run_rt_sys import parse_args, run_case
    from phy.rt_sys.contracts import source_fingerprint
    from scripts.validation.validate_environment_repairs import audit
    stages = defaultdict(lambda: dict(calls=0, inclusive_s=0.))
    latest = {}

    def instrument(cls, method, label):
        stages[label]  # preserve zero-call stages as explicit evidence
        original = getattr(cls, method)

        @functools.wraps(original)
        def timed(self, *positional, **keywords):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                return original(self, *positional, **keywords)
            finally:
                torch.cuda.synchronize()
                stages[label]['calls'] += 1
                stages[label]['inclusive_s'] += time.perf_counter() - start
                if method == 'evaluate':
                    latest.update(evaluated_links=self.calls, actual_decode_trials=self.decode_trials,
                                  link_statistics=dict(self.statistics))
        setattr(cls, method, timed)

    active = {'nr_tb_reference': NrReferenceAbstraction, 'nr_bler_bank': StatisticalBankAbstraction,
              'nr_direct': DirectNrAbstraction}[args.mode]
    for method, label in (('evaluate', 'link_total'), ('waveform', 'waveform_lookup_or_build'),
                          ('_probe', 'amc_and_bler_probes'),
                          ('_transmitted_feedback', 'transmitted_tb_feedback')):
        instrument(active, method, label)
    instrument(RtChannelProvider, 'snapshot', 'channel_snapshot')
    sys.argv = ['run_rt_sys.py', '--scene', args.scene, '--cells', '3', '--ues', '24',
                '--slots', str(args.slots), '--seed', '512', '--initial-queue-bytes', '1000',
                '--ue-speed-mps', '0', '--traffic-load', '1', '--feedback-mode', args.mode,
                '--control-mode', 'baseline', '--output', str(args.output)]
    if args.config:
        sys.argv.extend(['--config', str(args.config)])
    start = time.perf_counter()
    run_case(parse_args())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    status = json.loads((args.output / 'run_status.json').read_text())
    assert status['completed'] and status['spark_verified']
    payload = audit(args.output)
    initial = json.loads((args.output / 'initial_conditions.json').read_text())
    summary = json.loads((args.output / 'final_summary.json').read_text())
    report = dict(completed=True, scene=args.scene, mode=args.mode, slots=args.slots,
                  total_s=elapsed, seconds_per_slot=elapsed/args.slots, stages=dict(stages), **latest,
                  environment_sha256=initial['environment_sha256'], cache_key=initial['cache_key'],
                  source_sha256=source_fingerprint(ROOT), payload_audit=payload,
                  device=status['runtime_environment'],
                  delivered_bytes=sum(m['delivered_bytes'] for m in summary['packet_metrics'].values()),
                  scope='Short cost fixture with CUDA fences; includes initialization, not long-run speed or QoS')
    (args.output / 'cost_profile.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main(args):
    if args.output.exists():
        raise FileExistsError(f'Refusing overwrite: {args.output}')
    args.output.mkdir(parents=True)
    reports = []
    for scene in ('simple_street_canyon', 'munich'):
        for mode in ('nr_tb_reference', 'nr_bler_bank'):
            folder = args.output / f'{scene}_{mode}'
            folder.mkdir()
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--scene', scene,
                   '--mode', mode, '--slots', str(args.slots), '--output', str(folder)]
            if args.config:
                cmd.extend(['--config', str(args.config)])
            with (folder / 'process.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            reports.append(json.loads((folder / 'cost_profile.json').read_text()))
            print('completed', scene, mode, reports[-1]['total_s'], flush=True)
        assert reports[-2]['environment_sha256'] == reports[-1]['environment_sha256']
        # Compare exogenous arrivals independently of controlled delivery outcomes.
        events = []
        for mode in ('nr_tb_reference', 'nr_bler_bank'):
            file = args.output / f'{scene}_{mode}' / 'packet_events.jsonl'
            events.append([json.loads(line) for line in file.read_text().splitlines()
                           if json.loads(line).get('kind') == 'arrival'])
        assert events[0] and events[0] == events[1]
    final = dict(completed=True, cases=reports, clocks_unchanged=True,
                 scope='Serial short fixture comparison; no new offline probability table, no throughput-equivalence claim')
    (args.output / 'comparison.json').write_text(json.dumps(final, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--scene', choices=['simple_street_canyon', 'munich'])
    parser.add_argument('--mode', choices=['nr_direct', 'nr_tb_reference', 'nr_bler_bank'])
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.config:
        args.config = args.config.resolve()
    if not args.output.is_relative_to(ROOT / 'outputs') or args.slots < 1:
        parser.error('Use a new outputs/ directory and positive slots')
    worker(args) if args.worker else main(args)
