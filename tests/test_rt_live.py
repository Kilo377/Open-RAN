import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.rt_live import LivePublisher
from experiments.run_rt_sys import parse_args, run_case
from experiments.rt_live_state import configuration, plain
from models.action import ConfigAction, ControlMetadata


def test_live_telemetry_uses_actual_slot_results_and_counts_only_decodes(tmp_path):
    manifest = dict(scene='test', total_slots=4, slot_duration_ms=.5,
        cells=[dict(cell_id='cell_0', position_m=[0, 0, 10])],
        ues=[dict(ue_id='ue_0', trajectory=dict(times_s=[0, 1], positions_m=[[0, 0, 1], [2, 0, 1]]))])
    mesh = dict(vertex_positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), faces=np.array([0, 1, 2]))
    provider = NS(backend=NS(scene=NS(objects={'mesh': NS(mi_mesh=mesh)}), mi=NS(traverse=lambda x: x)))
    path = tmp_path/'live.json'
    publisher = LivePublisher(path, manifest, provider, dict(rt='cpu'), 'nr_direct')
    user = NS(ue_id='ue_0', queue_bytes=17)
    state = NS(slot_id=1, sim_time_ms=.5, ues=[user], connections={'ue_0': NS(serving_cell_id='cell_0', connection_state='connected')})
    simulator = NS(state=state)
    row = dict(ue_id='ue_0', delivered_bytes=100, actual_decode=True, scheduled=True,
        sinr_db=9.5, mcs=3, ack=True, throughput_mbps=1.6)
    adapter = NS(trace=dict(ues=[row]))
    publisher.publish(simulator, adapter)
    result = json.loads(path.read_text())
    assert result['delivered_bytes'] == 100
    assert result['throughput_mbps'] == 1.6
    assert result['decode_attempts'] == 1
    assert result['ues'][0]['position_m'] == pytest.approx([.001, 0, 1])
    assert result['ues'][0]['queue_bytes'] == 17
    assert user.queue_bytes == 17 and adapter.trace['ues'][0] == row
    state.slot_id, state.sim_time_ms = 2, 1.
    row.update(delivered_bytes=0, actual_decode=False, scheduled=False, ack=False)
    publisher.publish(simulator, adapter)
    publisher.finish('completed')
    result = json.loads(path.read_text())
    assert result['decode_attempts'] == 1 and result['nacks'] == 0
    assert result['delivered_bytes'] == 100
    assert result['throughput_mbps'] == .8
    assert result['ues'][0]['sinr_db'] is None
    assert result['stage'] == 'completed'
    with np.load(result['mesh_path']) as mesh:
        assert mesh['vertices'].shape == (3, 3)
        assert mesh['faces'].tolist() == [[0, 1, 2]]
    assert not path.with_suffix('.tmp').exists()


def test_operations_snapshot_distinguishes_effective_pending_and_rejected(tmp_path):
    from state.cell_state import CellState
    from state.scheduler_state import SchedulerState
    from state.resource_grid_state import ResourceGridState
    from state.control_state import ControlState
    manifest = dict(scene='test', total_slots=400, slot_duration_ms=.5,
        cells=[dict(cell_id='cell_0', position_m=[0, 0, 10])],
        ues=[dict(ue_id='ue_0', trajectory=dict(times_s=[0, 1], positions_m=[[0, 0, 1], [2, 0, 1]]))])
    mesh = dict(vertex_positions=np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]), faces=np.array([0, 1, 2]))
    provider = NS(backend=NS(scene=NS(objects={'mesh': NS(mi_mesh=mesh)}), mi=NS(traverse=lambda x: x)))
    request = ConfigAction(tx_power_dbm=40, target_cell_id='cell_0', metadata=ControlMetadata(source='test_xapp', issued_at_slot=0))
    cell = CellState('cell_0', tx_power_dbm=30)
    control = ControlState(active_config_actions=[request], actuation_outcomes=[dict(slot_id=1,
        status='rejected', source='test_xapp', reason='power envelope', target_cell_id='cell_0')])
    runtime = NS(cell=cell, scheduler=SchedulerState(), resource_grid=ResourceGridState(utilization=.25), radio_profile={})
    packet = dict(offered_bytes=120, delivered_bytes=100, pending_bytes=20, completed_packets=2,
        late_completed_packets=1, completion_latency_ms_sum=4.)
    user = NS(ue_id='ue_0', queue_bytes=20, packet_ledger=NS(summary=lambda: dict(packet)))
    state = NS(slot_id=1, sim_time_ms=.5, ues=[user], managed_cells={'cell_0':runtime}, control=control,
        connections={'ue_0': NS(serving_cell_id='cell_0', connection_state='connected')})
    simulator = NS(state=state, _pending_config_actuations=[(3, 1, request)], _pending_command_actuations=[])
    row = dict(ue_id='ue_0', delivered_bytes=100, actual_decode=True, scheduled=True,
        sinr_db=9., mcs=3, ack=True, throughput_mbps=1.6)
    path = tmp_path/'live.json'
    publisher = LivePublisher(path, manifest, provider, {}, 'nr_direct', control_mode='orchestrated')
    publisher.publish(simulator, NS(trace=dict(ues=[row], cells={})))
    d = json.loads(path.read_text())
    assert d['cell_metrics'][0]['effective']['tx_power_dbm'] == 30
    assert d['control']['pending'][0]['action']['tx_power_dbm'] == 40
    assert d['control']['pending'][0]['due_slot'] == 3
    assert d['control']['history'][0]['status'] == 'rejected'
    assert d['control']['history'][0]['reason'] == 'power envelope'
    assert d['control']['totals'] == {'rejected': 1}
    assert d['packet_totals']['mean_completion_latency_ms'] == 2
    assert d['packet_totals']['late_completed_fraction'] == .5
    # Observer never applies a pending action, consumes a receipt, or drains packets.
    assert cell.tx_power_dbm == 30 and len(simulator._pending_config_actuations) == 1
    assert len(control.actuation_outcomes) == 1 and user.queue_bytes == 20
    # Duplicate publications cannot double-count a decode or chart sample.
    publisher.publish(simulator, NS(trace=dict(ues=[row], cells={})))
    assert len(json.loads(path.read_text())['history']) == 1
    # Multi-window consumers receive all bounded samples, not only polled slots.
    control.actuation_outcomes.clear()
    row.update(actual_decode=False, scheduled=False, delivered_bytes=0)
    for slot in range(2, 306):
        state.slot_id, state.sim_time_ms = slot, slot*.5
        publisher.publish(simulator, NS(trace=dict(ues=[row], cells={})))
    d = json.loads(path.read_text())
    assert len(d['history']) == 300 and d['history'][0]['slot'] == 6
    assert d['decode_attempts'] == 1
    assert d['control']['totals'] == {'rejected': 1}


def test_configuration_is_allowlisted_and_baseline_is_explicit():
    manifest = dict(carrier_frequency_hz=3.5e9, slot_duration_ms=.5)
    configs = dict(ric=dict(xapps=[], rapps=[], api_key='do-not-export', xapp_interval_slots=5),
        sim=dict(config_actuation_delay_slots=1, password='do-not-export'))
    d = configuration(manifest, configs, 'baseline', {'requested':'agx'})
    assert d['control_mode'] == 'baseline'
    assert d['controllers']['xapps'] == []
    assert d['controllers']['xapp_interval_slots'] == 5
    assert 'do-not-export' not in json.dumps(d)
    assert plain(np.float32(np.inf)) is None


def test_live_state_rejects_path_outside_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['run_rt_sys', '--dry-run', '--live-state', str(tmp_path/'live.json')])
    with pytest.raises(ValueError, match='telemetry must stay'):
        run_case(parse_args())
