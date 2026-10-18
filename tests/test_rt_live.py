import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.rt_live import LivePublisher
from experiments.run_rt_sys import parse_args, run_case


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


def test_live_state_rejects_path_outside_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['run_rt_sys', '--dry-run', '--live-state', str(tmp_path/'live.json')])
    with pytest.raises(ValueError, match='telemetry must stay'):
        run_case(parse_args())
