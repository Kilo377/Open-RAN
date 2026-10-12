"""Equal cell power/grid SU/MU check with real NR reference feedback.

Analytic channels isolate spatial separation. This is not a map throughput
benchmark and does not force grants in a natural scheduler run.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.phy_input import PhyInput, PhyUeInput
from phy.rt_sys.adapter import RtSysPhyAdapter
from phy.rt_sys.contracts import ChannelSnapshot, source_fingerprint


def grant(uid, scheduled, group):
    return PhyUeInput(uid, 100000, 0, 0, .1, 0, scheduled, 'embb', 'video', 100, 1,
        allocated_prbs=2, allocated_symbols=14, allocated_subcarrier_indices=list(range(24)),
        allocated_symbol_indices=list(range(14)), requested_bytes=100000,
        grant_bytes_budget=100000, target_mcs=28, mcs_cap=28, layer_count=1,
        spatial_group_size=group, spatial_group_id=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    adapter = RtSysPhyAdapter({'device': args.device, 'feedback_mode': 'nr_tb_reference', 'rng_seed': 401})
    records = []
    for label, rows in (
        ('orthogonal_su', [[1, 0], [0, 1]]),
        ('orthogonal_mu', [[1, 0], [0, 1]]),
        ('correlated_mu', [[1, 0], [1, 0]]),
    ):
        h = np.asarray(rows, complex)[:, None, None, :, None, None] * 1e-5
        h = np.broadcast_to(h, (2, 1, 1, 2, 14, 24)).copy()
        totals = {'delivered_bytes': 0, 'expected_bytes_from_independent_bler': 0., 'physical_capacity_bytes': 0}
        sinrs = []; correlations = []
        for slot in range(1, 9):
            group = 1 if label.endswith('_su') else 2
            inputs = [grant('u0', True, group), grant('u1', group == 2, group)]
            snapshot = ChannelSnapshot(slot, ('u0', 'u1'), ('c0',), h, np.zeros((2, 3)), 'analytic', {'normalized': False})
            result = adapter.run_cluster_slot([PhyInput(slot, 1, 'c0', 0, 0, 0, ue_inputs=inputs)], snapshot)['c0']
            cell = adapter.trace['cells']['c0']
            assert abs(cell['budget_w'] - .001) < 1e-12
            assert abs(cell['average_used_w'] - .001) < 1e-12
            assert cell['occupied_re'] == 336
            for row in adapter.trace['ues']:
                if row['bler_probe_samples']:
                    totals['physical_capacity_bytes'] += row['physical_tb_capacity_bytes']
                    totals['expected_bytes_from_independent_bler'] += row['physical_tb_capacity_bytes'] * (1 - row['bler'])
                    sinrs.append(row['sinr_db'])
            totals['delivered_bytes'] += sum(row.delivered_bytes for row in result.ue_results)
            correlations.extend(row['mean_pair_channel_correlation'] for row in adapter.trace['precoders']
                if row['mean_pair_channel_correlation'] is not None)
        records.append({'case': label, 'slots': 8, 'cell_power_w': .001, 'occupied_re_per_slot': 336,
            **totals, 'mean_sinr_db': float(np.mean(sinrs)),
            'mean_pair_channel_correlation': float(np.mean(correlations)) if correlations else None})
    su, orthogonal, correlated = records
    assert orthogonal['delivered_bytes'] > su['delivered_bytes']
    assert orthogonal['expected_bytes_from_independent_bler'] > su['expected_bytes_from_independent_bler']
    assert correlated['delivered_bytes'] < orthogonal['delivered_bytes']
    assert correlated['mean_sinr_db'] < orthogonal['mean_sinr_db'] - 5
    output = ROOT / 'outputs/rt_sys_targeted_repairs/reference_mu_validation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'passed': True, 'device': args.device,
        'source_sha256': source_fingerprint(ROOT), 'cases': records,
        'scope': 'Analytic equal power/grid spatial multiplexing with actual NR decoding; not natural map gains'}, indent=2) + '\n')
    print(output.read_text())
