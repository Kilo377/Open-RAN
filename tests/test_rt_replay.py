"""Offline replay audits, arbitrary populations, partial windows and time scales."""
from pathlib import Path
import json
import re
import numpy as np
import pytest
from experiments.replay import export_replay,build_data,replay_window_slots,duration_label


def fixture_run(folder,*,slots=17,slot_ms=.5,sampled=False):
    folder.mkdir(exist_ok=True)
    ids=['ue_7','ue_19'];cells=['cell_5','cell_9'];dt=slot_ms/1000
    manifest=dict(scene='fixture',total_slots=slots,slot_duration_ms=slot_ms,cells=[dict(cell_id=c,position_m=[i*20,0,10]) for i,c in enumerate(cells)],
        ues=[dict(ue_id=u,trajectory=dict(times_s=[0,slots*dt],positions_m=[[i*20,1,1.5],[i*20+1,1,1.5]])) for i,u in enumerate(ids)],
        carrier_frequency_hz=3.5e9,bandwidth_hz=1e7,subcarrier_spacing_hz=30000,tx_rows=2,tx_cols=2,num_rx_ant=2,path_update_ms=5,ray_solver=dict(max_depth=2))
    configs=dict(phy=dict(rt_sys=dict(feedback_mode='fast' if sampled else 'nr_direct')),sim=dict(scheduler_enable_spatial_multiplexing=False),cell={},ric=dict(xapps=[],xapp_pool='chongyu'),
        ue=dict(ues=[dict(ue_id=u,slice_id='__APP_CODE__</script><script>alert(1)</script>',latency_budget_ms=10,max_queue_bytes=10000) for u in ids]))
    (folder/'config_snapshot.json').write_text(json.dumps(dict(manifest=manifest,configs=configs)))
    (folder/'run_status.json').write_text(json.dumps(dict(completed=True,elapsed_s=2,source_sha256='fixture')))
    (folder/'initial_conditions.json').write_text(json.dumps(dict(ues=[dict(ue_id=u,serving_cell_id=c,queue_bytes=0) for u,c in zip(ids,cells)])))
    metrics={u:dict(initial_bytes=0,offered_bytes=slots*100,delivered_bytes=slots*100,pending_bytes=0,overflow_bytes=0,completed_packets=slots,
        late_completed_packets=0,deadline_missed_packets=0,completion_latency_ms_sum=0,mean_completion_latency_ms=0) for u in ids}
    with (folder/'rt_sys_trace.jsonl').open('w') as f:
        for slot in range(1,slots+1):
            users=[]
            for u,c in zip(ids,cells):
                row=dict(ue_id=u,cell_id=c,scheduled=True,actual_decode=not sampled,feedback_available=True,ack_source='fast_sys_bernoulli' if sampled else 'actual_nr_crc_and_bit_check',
                    ack=True,is_retx=False,delivered_bytes=100,sinr_db=10,mcs=11)
                row.update({'packet_'+k:(slot*100 if k in ('offered_bytes','delivered_bytes') else slot if k=='completed_packets' else v) for k,v in metrics[u].items()})
                row['packet_hol_delay_ms']=0;users.append(row)
            f.write(json.dumps(dict(slot_id=slot,ues=users,cells={c:dict(average_used_w=1,occupied_re=10,grid_re=20) for c in cells},connection_states={u:'connected' for u in ids}))+'\n')
    (folder/'final_summary.json').write_text(json.dumps(dict(total_slots=slots,n_prbs_total=24,packet_metrics=metrics,paper_interfaces=dict(rf_energy_j={c:slots*dt for c in cells}))))
    np.savez(folder/'replay.mesh.npz',vertices=np.array([[0,0,0],[20,0,0],[0,20,0]],dtype=np.float32),faces=np.array([[0,1,2]]))
    return folder


def test_partial_window_has_correct_rates_and_end_counters(tmp_path):
    run=fixture_run(tmp_path/'run',sampled=True);data=build_data(run,window_ms=5)
    assert [f['slot'] for f in data['frames']]==[0,10,17]
    assert data['frames'][-1]['window_ms']==3.5
    assert data['frames'][-1]['g']==pytest.approx(3.2)
    assert data['frames'][-1]['decodes']==34 and data['meta']['actual_decodes']==0 and data['meta']['sampled_attempts']==34
    assert sum(f['g']*f['window_ms']/1000*1e6/8 for f in data['frames'][1:])==pytest.approx(3400)
    assert data['frames'][-1]['rf_energy_j']==pytest.approx(.017)


@pytest.mark.parametrize('new_trace',[False,True])
def test_replay_resource_utilization_excludes_fft_guard_tones(tmp_path,new_trace):
    run=fixture_run(tmp_path/'run');p=run/'config_snapshot.json';config=json.loads(p.read_text())
    config['manifest'].update(num_symbols=14,fft_size=512);p.write_text(json.dumps(config))
    p=run/'rt_sys_trace.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()]
    for row in rows:
        for cell in row['cells'].values():
            cell.update(grid_re=14*512,occupied_re=14*24*12)
            if new_trace:cell['active_grid_re']=14*24*12
    p.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    data=build_data(run)
    assert all(cell[1]==1 for frame in data['frames'][1:] for cell in frame['cells'])


def test_single_file_keeps_data_safe_and_embeds_all_dependencies(tmp_path):
    run=fixture_run(tmp_path/'run');report=export_replay(run,max_frames=3)
    source=Path(report['html']).read_text()
    payload=re.search(r'<script id="replay-data" type="application/json">(.*?)</script>',source,re.S).group(1)
    data=json.loads(payload)
    assert data['users'][0]['slice']=='__APP_CODE__</script><script>alert(1)</script>'
    assert '</script>' not in payload and 'data:text/javascript;base64,' in source
    assert not re.search(r'<script[^>]+src=',source)
    assert '0.5 ms' in source and report['slots']==17 and report['frames']==3
    assert report['self_contained'] and report['audits_passed']


def test_replay_refuses_incomplete_or_inconsistent_source_and_preserves_it(tmp_path):
    run=fixture_run(tmp_path/'run');p=run/'run_status.json';p.write_text('{"completed":false}')
    with pytest.raises(ValueError,match='completed'):export_replay(run)
    assert p.read_text()=='{"completed":false}' and not (run/'network-replay.html').exists()
    p.write_text('{"completed":true}');trace=run/'rt_sys_trace.jsonl';source=trace.read_text();trace.write_text(source.replace('"slot_id": 2,','"slot_id": 3,',1))
    with pytest.raises(ValueError,match='missing/repeated'):export_replay(run)


def test_internal_export_does_not_publish_run_completion_before_cleanup(tmp_path):
    run=fixture_run(tmp_path/'run');p=run/'run_status.json'
    p.write_text(json.dumps(dict(completed=False,simulation_completed=True,stage='exporting_replay')))
    with pytest.raises(ValueError,match='completed'):export_replay(run)
    assert export_replay(run,allow_finalizing=True)['audits_passed']
    assert json.loads(p.read_text())['completed'] is False
    for stage in ['failed','cleanup_failed','cancelled']:
        p.write_text(json.dumps(dict(completed=False,simulation_completed=True,stage=stage)))
        with pytest.raises(ValueError,match='completed'):build_data(run,allow_finalizing=True)
        p.write_text(json.dumps(dict(completed=True,stage=stage)))
        with pytest.raises(ValueError,match='completed'):build_data(run,allow_finalizing=True)
    p.write_text(json.dumps(dict(completed=True,stage='exporting_replay')))
    with pytest.raises(ValueError,match='completed'):export_replay(run)


def test_resource_cleanup_attempts_all_closers_and_reports_failures():
    from experiments.run_rt_sys import close_run_resources
    from types import SimpleNamespace as N
    calls=[]
    def close_first():calls.append('phy');raise RuntimeError('injected PHY close failure')
    def close_second():calls.append('rt');raise KeyboardInterrupt('injected RT cancellation')
    def close_third():calls.append('writers')
    errors=close_run_resources([('phy',N(close=close_first)),('rt',N(close=close_second)),
        ('writers',N(close=close_third)),('absent',None)])
    assert calls==['phy','rt','writers']
    assert [e['resource'] for e in errors]==['phy','rt']
    assert 'RuntimeError' in errors[0]['error'] and 'KeyboardInterrupt' in errors[1]['error']


def test_adaptive_windows_bound_replay_size_from_ms_to_hours():
    for slots in [1,17,400,2000,20000,120000,7200000,172800000]:
        window=replay_window_slots(slots,.5,max_frames=500)
        assert 1<=window and (slots+window-1)//window<=500
    assert replay_window_slots(20000,.5)==40
    assert duration_label(.2)=='200 ms' and duration_label(10)=='10 s' and duration_label(3600)=='60 min'
    with pytest.raises(ValueError):replay_window_slots(100,.5,window_ms=float('nan'))


def test_replay_labels_historical_and_current_fast_sampling(tmp_path):
    run=fixture_run(tmp_path/'run',sampled=True)
    assert build_data(run)['meta']['fast_profile']=='reference'
    p=run/'config_snapshot.json';c=json.loads(p.read_text())
    c['configs']['phy']['rt_sys'].update(fast_profile='efficient',algebra_device='cpu')
    p.write_text(json.dumps(c))
    assert build_data(run)['meta']['fast_profile']=='efficient'
    report=export_replay(run);source=Path(report['html']).read_text()
    assert '4 SC / PRB' in source and '1 / slot / stream pattern' in source
