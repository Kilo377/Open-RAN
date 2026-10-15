"""Real RT/NR handover lifecycle fixture with deliberately overlapping cells.

Two co-located, equal-power cells give both candidates a viable reference SINR.
This is a controlled protocol/consumer fixture, not a realistic deployment or a
capacity experiment. No channel tensors or decoded feedback are substituted.
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args):
    from experiments import run_rt_sys as runner
    original = runner.build_case
    def fixture(*values, **kwargs):
        manifest, configs, intent = original(*values, **kwargs)
        placement = deepcopy(manifest['cells'][1])
        manifest['cells'][0] = {**deepcopy(placement), 'cell_id': 'cell_0'}
        configs['cell']['neighbor_cell_positions'] = {
            c['cell_id']: dict(zip(('x_m', 'y_m', 'z_m'), c['position_m'])) for c in manifest['cells']}
        configs['sim']['measurement_sinr_noise_std_db'] = 0
        return manifest, configs, intent
    runner.build_case = fixture
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ('success', 'target_failure', 'management_cancel', 'success_interference_step'):
        timeline = [dict(slot=20, commands=[dict(command_type='request_handover', target_ue_id='ue_0', target_cell_id='cell_1')])]
        if mode.startswith('success'):
            timeline.append(dict(slot=120,commands=[dict(command_type='request_handover',target_ue_id='ue_0',target_cell_id='cell_0')]))
            if mode=='success_interference_step':
                timeline.append(dict(slot=110,config=dict(target_cell_id='cell_0',tx_power_dbm=24)))
        elif mode == 'target_failure':
            timeline.append(dict(slot=40, config=dict(target_cell_id='cell_1', tx_power_dbm=-100)))
        elif mode == 'management_cancel':
            timeline.append(dict(slot=40, config=dict(target_cell_id='cell_0', neighbor_cell_ids=[])))
        path = root / (mode + '_timeline.json')
        path.write_text(json.dumps(timeline, indent=2))
        argv = ['--config', str(ROOT / 'configs/rt_sys/paper_interfaces.json'), '--cells', '2', '--ues', '2' if mode=='success_interference_step' else '1', '--seed', '512',
                '--slots', '160', '--ue-speed-mps', '0', '--initial-queue-bytes', '0', '--control-mode', 'baseline',
                '--feedback-mode', 'nr_direct', '--control-timeline', str(path), '--output', str(root / mode)]
        saved = sys.argv
        try:
            sys.argv = [saved[0], *argv]
            runner.run_case(runner.parse_args())
        finally:
            sys.argv = saved
        final = json.loads((root / mode / 'final_summary.json').read_text())
        events = final['paper_interfaces']['events']
        kinds = [e['kind'] for e in events]
        assert kinds.count('handover_attempt') == 1, kinds
        expected = {'success': 'handover_success','success_interference_step':'handover_success', 'target_failure': 'handover_failure', 'management_cancel': 'handover_cancelled'}[mode]
        assert kinds.count(expected) == 1, kinds
        if mode != 'target_failure':
            assert 'handover_failure' not in kinds
        decoded_cells = set();dwell_rejected=False;interference=[];reference_before=[];reference_after=[]
        for line in (root / mode / 'rt_sys_trace.jsonl').open():
            trace = json.loads(line)
            dwell_rejected |= any(r.get('reason')=='minimum_dwell_time' for r in trace['paper_interfaces']['command_receipts'])
            for u in trace['ues']:
                if u['actual_decode'] and u['ue_id']=='ue_0':
                    decoded_cells.add(u['cell_id'])
                    if u['cell_id']=='cell_1':interference.append(dict(slot=trace['slot_id'],interference_dbm=u['inter_cell_interference_dbm']))
                if mode=='success_interference_step' and u['ue_id']=='ue_0' and u['cell_id']=='cell_1':
                    ref=trace['paper_interfaces']['reports']['ue']['ue_0']['reference_sinr_db']
                    if 102<=trace['slot_id']<110:reference_before.append(ref)
                    if trace['slot_id']>=140:reference_after.append(ref)
        if mode.startswith('success'):
            assert decoded_cells == {'cell_0', 'cell_1'}
            assert dwell_rejected
        stats = final['link_reference_statistics']
        assert stats['actual_tb_decodes'] > 0
        assert all(stats.get(k, 0) == 0 for k in ('amc_probe_trials', 'bler_probe_trials', 'calibration_decode_trials', 'incompatible_no_tb', 'unsupported_no_tb'))
        if mode=='success_interference_step':
            assert reference_before and reference_after and max(reference_after)>min(reference_before)
            assert any(r['interference_dbm']>-200 for r in interference),'No actual neighboring transmission interference'
        from scripts.validation.validate_efficiency import audit
        audit_result=audit(root/mode)
        results.append(dict(mode=mode, passed=True, events=events, decoded_cells=sorted(decoded_cells), actual_decodes=stats['actual_tb_decodes'],
            audit=audit_result,interference_samples=interference,reference_before=reference_before,reference_after=reference_after,
            interpretation='Executed power step and actual interference evidence; changing grants preclude a fixed-dB data-SINR causal claim.'))
    (root / 'validation.json').write_text(json.dumps(dict(passed=True, fixture='co-located overlap; real RT and NR; not a deployment performance result', results=results), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
