"""Finite fixed-baseline GPU checks; no training or algorithm selection."""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phy.rt_sys.contracts import source_fingerprint


def audit(folder):
    status = json.loads((folder/'run_status.json').read_text())
    config = json.loads((folder/'config_snapshot.json').read_text())
    final = json.loads((folder/'final_summary.json').read_text())
    assert status['completed'] and status['spark_verified']
    assert status['source_sha256'] == source_fingerprint(ROOT)
    assert config['control_mode'] == 'baseline'
    assert not config['configs']['ric']['rapps'] and not config['configs']['ric']['xapps']
    assert config['configs']['ric']['freeze_learning']
    counts = Counter()
    delivered = Counter()
    scheduled = Counter()
    for row in csv.DictReader((folder/'ue_metrics.csv').open()):
        uid = row['ue_id']
        before, ack = int(row['queue_bytes']), row['ack'] == 'True'
        tb, good = int(row['tb_bytes']), int(row['delivered_bytes'])
        assert 0 <= good <= tb <= before
        assert good == (tb if ack else 0)
        if row['scheduled'] != 'True':
            assert good == 0
        else:
            scheduled[uid] += 1
        if row['ack_source'] == 'no_path':
            assert tb == good == 0
        if row['reference_tb_ack'] == 'False':
            assert not ack
        assert row['measurement_sinr_kind'] == 'rt_full_band_reference'
        initial, offered = int(row['packet_initial_bytes']), int(row['packet_offered_bytes'])
        pending, lost = int(row['packet_pending_bytes']), int(row['packet_overflow_bytes'])
        cumulative = int(row['packet_delivered_bytes'])
        assert initial+offered == pending+lost+cumulative
        assert pending == before-good
        counts['zero_tb'] += tb == 0 and row['scheduled'] == 'True'
        counts['no_path'] += row['ack_source'] == 'no_path'
        counts['unsupported_code'] += row['ack_source'] == 'unsupported_nr_code_no_tb'
        counts['target_unmet'] += row['amc_selection_target_met'] == 'False'
        counts['low_rate_reference'] += row['coding_backend'] == 'sionna_bg2_rv0_circular_repetition'
        counts['capacity_guards'] += row['mac_capacity_guard'] == 'True'
        counts['rows'] += 1
        delivered[uid] += good
    assert len(delivered) == len(config['manifest']['ues'])
    assert counts['rows'] == config['manifest']['total_slots']*len(delivered)
    for uid, metrics in final['packet_metrics'].items():
        assert metrics['delivered_bytes'] == delivered[uid]
        assert metrics['initial_bytes']+metrics['offered_bytes'] == metrics['pending_bytes']+metrics['overflow_bytes']+metrics['delivered_bytes']
    power_error = 0.
    for row in csv.DictReader((folder/'cell_metrics.csv').open()):
        used, budget = float(row['average_used_w']), float(row['budget_w'])
        assert 0 <= int(row['occupied_re']) <= int(row['grid_re'])
        assert used <= budget + 1e-10
        power_error = max(power_error, abs(used-budget) if int(row['emitted_re_streams']) else abs(used))
    assert power_error < 1e-10
    payload_flushes = 0
    for line in (folder/'harq_events.jsonl').open():
        event = json.loads(line)
        if event['kind'] == 'payload_flush':
            assert event['pending_bytes'] > 0
            payload_flushes += 1
    assert payload_flushes == sum(u['payload_flushes'] for u in final['harq_lifecycle'].values())
    counts['packet_overflow_bytes'] = sum(u['overflow_bytes'] for u in final['packet_metrics'].values())
    counts['packet_deadline_misses'] = sum(u['deadline_missed_packets'] for u in final['packet_metrics'].values())
    return dict(case=folder.name, passed=True, elapsed_s=status['elapsed_s'], counters=dict(counts),
                zero_delivery_ues=[uid for uid,value in delivered.items() if value == 0],
                never_scheduled_ues=[uid for uid in delivered if not scheduled[uid]],
                delivered_bytes=sum(delivered.values()), max_power_error_w=power_error,
                source_sha256=status['source_sha256'], control_mode='baseline',
                mean_completion_latency_ms_by_ue={uid:m['mean_completion_latency_ms'] for uid,m in final['packet_metrics'].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/rt_environment_repairs_20261013/final_baseline')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, choices=(1,2,3,4), default=4)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'outputs'):
        raise ValueError('Validation output must remain in copy outputs/')
    out.mkdir(parents=True,exist_ok=True)
    plan = [(scene,ues,load,slots) for scene in ('simple_street_canyon','munich')
            for ues,load,slots in ((4,1.,40),(12,.1,120),(24,.1,120),(24,1.,120))]
    def run(spec):
        scene,ues,load,slots = spec
        folder = out/f'{scene}_u{ues}_load{load}_slots{slots}'
        if args.run:
            if folder.exists():
                raise ValueError(f'Refusing to overwrite saved validation: {folder}')
            folder.mkdir()
            cmd = [sys.executable,str(ROOT/'experiments/run_rt_sys.py'),'--scene',scene,
                   '--cells','3','--ues',str(ues),'--seed','402','--slots',str(slots),
                   '--traffic-load',str(load),'--control-mode','baseline','--output',str(folder)]
            with (folder/'process.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        return audit(folder)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(run,plan))
    report = dict(passed=True, cases=results, scope='Finite environment baseline integration; no QoS guarantee or algorithm comparison',
                  checks=['RT-only reporting','packet byte ledger each UE/slot','delivery/ACK/TB/queue consistency',
                          'power conservation','no-path zero delivery','positive-payload HARQ flush only',
                          'no rApps/xApps or learning'], source_sha256=source_fingerprint(ROOT))
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
