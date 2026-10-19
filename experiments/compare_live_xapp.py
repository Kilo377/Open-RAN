"""Matched OFF/ON runs of the registered live xApp; no GUI or synthetic KPI."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.live_xapp_test import APP, ARGUMENTS, atomic_json


def compare(spec_path, output, result, cell='*', app=APP):
    from experiments.run_rt_sys import parse_args, run_case
    spec = json.loads(Path(spec_path).read_text())
    if set(spec) != {'arguments', 'slots'} or set(spec['arguments']) - set(ARGUMENTS):
        raise ValueError('Invalid comparison specification')
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Comparison outputs must remain under project outputs')
    output.mkdir(parents=True, exist_ok=True)
    arms = {}
    for arm in ('off', 'on'):
        atomic_json(result, dict(stage='running_' + arm, app_id=app, output=str(output), arms=arms,
            note='Both runs use identical initial conditions; live before/after segments are not a causal comparison.'))
        previous = sys.argv
        try:
            sys.argv = ['run_rt_sys.py']
            args = parse_args()
        finally:
            sys.argv = previous
        for key, value in spec['arguments'].items():
            setattr(args, key, Path(value) if key in ('config', 'channel_cache_root') else value)
        args.slots = int(spec['slots'])
        args.live_xapp_test, args.xapp_test_start, args.xapp_test_cell = True, arm, cell
        args.xapp_test_app = app
        args.output = output / arm
        args.live_state = args.output / 'live.json'
        run_case(args)
        status = json.loads((args.output / 'run_status.json').read_text())
        if status.get('stage') != 'completed':
            raise ValueError('Comparison arm did not complete: ' + arm)
        summary = json.loads((args.output / 'final_summary.json').read_text())
        initial = json.loads((args.output / 'initial_conditions.json').read_text())
        live = json.loads(args.live_state.read_text())
        segment = summary['xapp_test']['segments'][-1]
        if segment['slots'] < 1:
            raise ValueError('Run needs more slots after the 20 ms control warm-up')
        arms[arm] = dict(metrics=segment, environment_sha256=initial['environment_sha256'],
            output=str(args.output), per_ue_packets=summary['packet_metrics'],
            executed_schedulers={c['cell_id']: c['scheduler']['scheduler_mode'] for c in live['cell_metrics']},
            decisions=summary['xapp_test']['last_decisions'], source_sha256=status['source_sha256'])
    matched = all(arms['off'][key] == arms['on'][key] for key in ('environment_sha256', 'source_sha256'))
    if not matched:
        raise ValueError('Source or exogenous initial conditions changed between arms')
    keys = ('throughput_mbps', 'mean_queue_bytes', 'mean_max_hol_ms', 'mean_completion_delay_ms', 'nack_fraction')
    delta = {key: arms['on']['metrics'][key] - arms['off']['metrics'][key]
             if arms['on']['metrics'][key] is not None and arms['off']['metrics'][key] is not None else None
             for key in keys}
    report = dict(stage='completed', app_id=app, matched_initial_conditions=matched, arms=arms,
        delta_on_minus_off=delta, output=str(output),
        interpretation='One paired seed is a functional comparison, not proof of general improvement. Delay includes completed packets only; queue/HOL include pending work.')
    atomic_json(result, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--cell', default='*')
    parser.add_argument('--app', default=APP)
    options = parser.parse_args()
    try:
        compare(options.spec, options.output, options.result, options.cell, options.app)
    except Exception as error:
        atomic_json(options.result, dict(stage='failed', error=str(error), output=str(options.output)))
        raise
    finally:
        options.result.with_suffix('.lock').unlink(missing_ok=True)
