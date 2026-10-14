"""Paper interfaces: physical effects, transaction atomicity and event accounting."""
from types import SimpleNamespace as NS
from copy import deepcopy
import math
import numpy as np
import pytest
from models.action import ConfigAction,CommandAction
from phy.rt_sys.paper_controls import validate_transaction
from phy.rt_sys.paper_radio import ret_basis,quantify_cqi


def make_sim(tmp_path,cells=2,slots=100,events=False,config=None,timeline=None):
    from test_rt_sys import FakeTracer
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from core.simulator import Simulator
    from core.factory import build_controller_registry,build_scheduler
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    m,c,_=build_case(num_cells=cells,num_ues=4,total_slots=slots,seed=512,initial_queue_bytes=4000,ue_speed_mps=0)
    c['ric'].update(rapps=[],xapps=[]);c['sim']['measurement_sinr_noise_std_db']=0
    c['phy']['rt_sys']['paper_interfaces']={'enabled':True,'connection_events':events,**(config or {})}
    provider=RtChannelProvider(m,tmp_path/'cache',backend=FakeTracer(m))
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'cpu','feedback_mode':'nr_direct'},channel_provider=provider)
    runtime=build_runtime_scenario_config(scenario_name='paper_fixture',scenario_type='rt_scene',sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    registry=build_controller_registry(c['ric'])
    if timeline:
        from experiments.fixed_control_timeline import FixedControlTimeline
        registry.register_xapp(FixedControlTimeline(timeline))
    return Simulator(runtime,registry,build_scheduler(c['sim']),adapter,c['cell'],c['ue'],c['beamforming'],str(tmp_path/'run'))


def command(kind,uid='ue_0',target=None,rid='request',priority=1):
    c=CommandAction(kind,target_ue_id=uid,target_cell_id=target,control_request_id=rid)
    c.metadata.priority=priority;c.metadata.issued_at_slot=0
    return c


def test_ret_main_lobe_and_rzf_cannot_undo_analog_tilt():
    from phy.rt_sys.numerics import Algebra
    rows,cols=8,2;z=(rows-1)/2-np.arange(rows)
    def h(angle):return np.kron(np.ones(cols),np.exp(-1j*np.pi*z*np.sin(np.deg2rad(angle))))[None,None,:]
    down=ret_basis(rows,cols,10);flat=ret_basis(rows,cols,0)
    np.testing.assert_allclose(down.conj().T@down,np.eye(cols),atol=1e-12)
    scan=np.linspace(-15,25,161)
    response=[np.linalg.norm(h(a)@down)**2 for a in scan]
    assert scan[int(np.argmax(response))]==10
    algebra=Algebra();hd=h(10)
    wf=algebra.rzf(hd,flat,np.array([[[1e-4]]]))
    wd=algebra.rzf(hd,down,np.array([[[1e-4]]]))
    assert abs((hd@wd).item())**2>abs((hd@wf).item())**2*2


def test_cqi_quantization_boundaries_and_missing():
    q=[quantify_cqi(v) for v in np.linspace(-30,40,1000)]
    assert q==sorted(q) and q[0]==0 and q[-1]==15
    assert quantify_cqi(None) is None and quantify_cqi(float('nan')) is None
    with pytest.raises(ValueError):quantify_cqi(0,[1]*15)


def test_power_transaction_budget_and_empty_clear(tmp_path):
    sim=make_sim(tmp_path);s=sim.state;n=s.managed_cells['cell_0'].resource_grid.n_prbs_total
    before=s.managed_cells['cell_0'].cell.tx_power_dbm
    bad=ConfigAction(target_cell_id='cell_0',tx_power_dbm=29,prb_power_w=[1.]*n)
    sim._apply_rt_config_action(bad,0)
    assert s.control.actuation_outcomes[-1]['reason']=='prb_power_exceeds_cell_budget'
    assert s.managed_cells['cell_0'].cell.tx_power_dbm==before
    action=ConfigAction(target_cell_id='cell_0',prb_power_w=[.4/n]*n,ue_prb_limits={'ue_0':4,'ue_1':0})
    sim._apply_rt_config_action(action,0)
    assert s.control.actuation_outcomes[-1]['status']=='applied'
    for _ in range(8):sim.step()
    assert sim.phy_adapter.trace['cells']['cell_0']['average_used_w']<=.4+1e-7
    for u in sim.phy_adapter.trace['ues']:
        if u['ue_id']=='ue_0':assert len(u['resource_allocation']['prbs'])<=4
        if u['ue_id']=='ue_1':assert not u['actual_decode']
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_0',clear_prb_power=True,ue_prb_limits={}),s.slot_id)
    assert s.managed_cells['cell_0'].cell.prb_power_w is None and not s.managed_cells['cell_0'].cell.ue_prb_limits


def test_zero_power_stops_actual_decodes_and_rf_energy(tmp_path):
    sim=make_sim(tmp_path,cells=1);n=sim.state.resource_grid.n_prbs_total
    sim._apply_rt_config_action(ConfigAction(prb_power_w=[0.]*n),0)
    for _ in range(4):sim.step()
    assert sim.phy_adapter.abstraction.decode_trials==0
    assert sim.phy_adapter.trace['cells']['cell_0']['average_used_w']==0
    assert sim.state.paper.rf_energy['cell_0']==0
    assert sim.state.paper.published['cell']['cell_0']['rf_efficiency_bit_per_j'] is None


def test_manual_disconnect_idempotence_conflict_and_reconnect(tmp_path):
    sim=make_sim(tmp_path,cells=1,config={'connection_model':{'good_ms':1}});p=sim.state.paper
    p.consume_commands([command('disconnect',rid='a',priority=10),command('reconnect',rid='b',priority=1)])
    assert 'ue_0' in p.manual and p.receipts[-1]['reason']=='conflicting_ue_command'
    p.consume_commands([command('disconnect',rid='a')]);assert p.receipts[-1]['reason']=='duplicate_request'
    for _ in range(3):sim.step()
    assert not next(u for u in sim.phy_adapter.trace['ues'] if u['ue_id']=='ue_0')['actual_decode']
    assert not any(e['kind']=='connection_drop' for e in p.all_events)
    p.consume_commands([command('reconnect',rid='c')])
    for _ in range(5):sim.step()
    assert sim.state.connections['ue_0'].connection_state=='connected'
    assert sum(e['kind']=='connection_normal_end' for e in p.all_events)==1


def test_delivered_projection_isolated_and_cqi_not_derived_from_mcs(tmp_path):
    from ric.xapp_contract import project,XAppContract
    sim=make_sim(tmp_path,cells=1)
    for _ in range(8):obs=sim.step()
    obs=sim._stamp_observation_delivery(obs,delivered_at_slot=10)
    contract=XAppContract(('wideband_cqi','resource_allocation','rf_energy_j'),frozenset({'tx_power_dbm'}),max_age_slots=20)
    projected,errors=project(obs,contract,now_slot=10)
    assert not errors,errors
    a=projected.ue_observations[0];before=a.wideband_cqi
    assert before is not None and a.kpi_metadata['wideband_cqi']['produced_at_slot']<=obs.produced_at_slot-2
    sim.state.ues[0].mcs_index=0
    again,_=project(obs,contract,now_slot=10);assert again.ue_observations[0].wideband_cqi==before
    a.resource_allocation['prbs'].append(999)
    assert 999 not in sim.state.paper.published['ue']['ue_0']['resource_allocation']['prbs']


def _reference_fixture(sim,monkeypatch):
    from phy.rt_sys import paper_environment as module
    raw=deepcopy(sim.state.paper.raw)
    for cells in raw.values():
        for cid,r in cells.items():
            r.update(rsrp_dbm=-70 if cid=='cell_0' else -72,reference_sinr_db=10.,no_path=False,prb_sinr_db=[10.]*len(r['prb_sinr_db']))
    monkeypatch.setattr(module,'references',lambda *args,**kwargs:deepcopy(raw))
    sim.phy_adapter.config['mobility']['measurement_filter_ms']=0
    return raw


def _tick_model(sim,slots):
    for _ in range(slots):
        sim.state.slot_id+=1
        sim.state.paper.before_scheduling()


def test_ttt_cio_hysteresis_and_neighbor_removal_do_not_change_radio(tmp_path,monkeypatch):
    sim=make_sim(tmp_path,events=True);p=sim.state.paper;raw=_reference_fixture(sim,monkeypatch)
    _tick_model(sim,1)
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_0',cio_offsets_db={'cell_1':5},time_to_trigger_ms=2),sim.state.slot_id)
    _tick_model(sim,3);assert 'ue_0' not in p.pending
    _tick_model(sim,2);assert 'ue_0' in p.pending
    assert p.observed['ue_0']['cell_1']['rsrp_dbm']==-72
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_0',neighbor_cell_ids=[]),sim.state.slot_id)
    assert 'ue_0' not in p.pending
    assert any(e['kind']=='handover_cancelled' for e in p.all_events)
    assert not any(e['kind']=='handover_failure' for e in p.all_events)
    assert p.raw['ue_0']['cell_1']['reference_interference_w']==raw['ue_0']['cell_1']['reference_interference_w']
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_0',neighbor_cell_ids=['cell_1'],steering_hysteresis_db=10),sim.state.slot_id)
    _tick_model(sim,10);assert 'ue_0' not in p.pending


def test_handover_success_failure_timeout_and_nack_not_rlf(tmp_path,monkeypatch):
    sim=make_sim(tmp_path,events=True,config={'connection_model':{'good_ms':1,'handover_timeout_ms':10}})
    p=sim.state.paper;raw=_reference_fixture(sim,monkeypatch);_tick_model(sim,1)
    p.prepare_reference();p.consume_commands([command('request_handover',target='cell_1',rid='ho')])
    _tick_model(sim,4)
    assert sim.state.connections['ue_0'].serving_cell_id=='cell_1'
    assert any(e['kind']=='handover_success' for e in p.all_events)
    p.prepare_reference();p.consume_commands([command('request_handover',target='cell_0',rid='too_soon')])
    assert p.receipts[-1]['reason']=='minimum_dwell_time'
    assert sum(e['kind']=='handover_attempt' for e in p.all_events)==1
    # Independent failure/timeout fixtures do not spend wall time in a dwell.
    p.adapter.config['mobility']['min_dwell_ms']=0
    p.prepare_reference();p.consume_commands([command('request_handover',target='cell_0',rid='fail')])
    raw['ue_0']['cell_0']['no_path']=True
    _tick_model(sim,1)
    assert any(e.get('reason')=='target_unavailable' for e in p.all_events)
    raw['ue_0']['cell_0']['no_path']=False;p.model['good_ms']=100;p.model['handover_timeout_ms']=1
    p.prepare_reference();p.consume_commands([command('request_handover',target='cell_0',rid='timeout')]);_tick_model(sim,4)
    assert any(e.get('reason')=='timeout' for e in p.all_events)
    sim.state.connections['ue_0'].consecutive_nack_count=100
    _tick_model(sim,10)
    assert not any(e['kind']=='connection_drop' for e in p.all_events)


def test_reference_rlf_once_and_recovery_opens_new_session(tmp_path,monkeypatch):
    sim=make_sim(tmp_path,cells=1,events=True,config={'connection_model':{'bad_ms':2,'good_ms':1}})
    p=sim.state.paper;raw=_reference_fixture(sim,monkeypatch)
    for uid,cells in raw.items():cells['cell_0'].update(no_path=True,rsrp_dbm=-200,reference_sinr_db=-200)
    _tick_model(sim,3);assert 'ue_0' in p.sessions
    _tick_model(sim,4);assert 'ue_0' not in p.sessions
    _tick_model(sim,8)
    assert sum(e['kind']=='connection_drop' and e['ue_id']=='ue_0' for e in p.all_events)==1
    for uid,cells in raw.items():cells['cell_0'].update(no_path=False,rsrp_dbm=-70,reference_sinr_db=10)
    _tick_model(sim,4)
    assert 'ue_0' in p.sessions and sim.state.connections['ue_0'].connection_state=='connected'
    assert sum(e['kind']=='connection_started' and e['ue_id']=='ue_0' for e in p.all_events)==2


def test_ret_actual_codec_closed_loop_and_restore_ports(tmp_path):
    sim=make_sim(tmp_path,cells=1)
    sim._apply_rt_config_action(ConfigAction(ret_enabled=True,ret_tilt_deg=10),0)
    for _ in range(4):sim.step()
    assert sim.phy_adapter.abstraction.decode_trials>0
    assert all(not u.get('mac_capacity_guard') for u in sim.phy_adapter.trace['ues'])
    assert all(x['beam_subspace_dimension']<=2 for x in sim.phy_adapter.trace['precoders'])
    sim._apply_rt_config_action(ConfigAction(ret_codebook_id=1),sim.state.slot_id)
    sim.step();assert sim.state.scheduler.max_total_spatial_layers==1
    sim._apply_rt_config_action(ConfigAction(ret_enabled=False),sim.state.slot_id)
    sim.step();assert sim.state.scheduler.max_total_spatial_layers==2


def test_absolute_frequency_power_changes_only_matching_interference():
    from test_rt_sys import channel,phy,ue,FixtureAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    h=np.ones((4,2,1,1))*1e-5;snap=channel(h,('u0','u1','u2','u3'),('c0','c1'))
    adapter=RtSysPhyAdapter({'device':'numpy'},abstraction=FixtureAbstraction())
    cells={cid:NS(cell=NS(prb_power_w=[.001,.001],ret_enabled=False)) for cid in ('c0','c1')}
    adapter.paper=NS(state=NS(managed_cells=cells))
    plans=[phy('c0',[ue('u0'),ue('u2',start=1)],10),phy('c1',[ue('u1'),ue('u3',start=1)],10)]
    for p in plans:
        p.runtime_context=NS(carrier_frequency_hz=3.5e9,bandwidth_hz=10e6,subcarrier_spacing_hz=30000,n_prbs_total=2,num_ofdm_symbols_per_slot=2,cyclic_prefix_type='normal',slot_duration_ms=.5)
    before=adapter.run_cluster_slot(plans,snap)
    base={u.ue_id:u.sinr_db for r in before.values() for u in r.ue_results}
    cells['c0'].cell.prb_power_w=[.002,.001]
    after=adapter.run_cluster_slot(plans,snap)
    now={u.ue_id:u.sinr_db for r in after.values() for u in r.ue_results}
    assert now['u0']>base['u0']+2.9 and now['u1']<base['u1']-2.5
    assert now['u2']==pytest.approx(base['u2']) and now['u3']==pytest.approx(base['u3'])
    assert adapter.trace['cells']['c0']['average_used_w']==pytest.approx(.003)


def test_timeline_goes_through_ric_permissions_and_actuation_delay(tmp_path):
    import json
    path=tmp_path/'timeline.json';path.write_text(json.dumps([dict(slot=5,config=dict(target_cell_id='cell_0',ue_prb_limits={'ue_0':1}),commands=[dict(command_type='disconnect',target_ue_id='ue_1')])]))
    sim=make_sim(tmp_path,cells=1,timeline=path)
    for _ in range(8):sim.step()
    assert sim.state.cell.ue_prb_limits=={'ue_0':1}
    assert 'ue_1' in sim.state.paper.manual
    assert any(e['kind']=='connection_normal_end' and e['ue_id']=='ue_1' for e in sim.state.paper.all_events)
    app=sim.controller_manager.e2_interfaces[0].xapp
    assert app.inputs
    record=next(r for r in app.inputs[0]['kpis'] if r['name']=='wideband_cqi')
    assert record['produced_at_slot']<app.inputs[0]['slot']


def test_retx_waits_for_budget_then_uses_original_codeword(tmp_path,monkeypatch):
    from phy.rt_sys.nr_harq import NrHarqReceiver
    original=NrHarqReceiver.transmit;sent=[]
    def force_first(receiver,wave,values,seed,slot,uid,context):
        result=original(receiver,wave,values,seed,slot,uid,context)
        if uid=='ue_0' and not result['no_tb']:
            sent.append((slot,deepcopy(result)))
            if len(sent)==1:result.update(ack=False,errors=1)
        return result
    monkeypatch.setattr(NrHarqReceiver,'transmit',force_first)
    sim=make_sim(tmp_path,cells=1)
    sim.step();assert sent
    ue=sim.state.ues[0];process=next(p for p in ue.harq.processes if p.tx_contract is not None)
    contract=process.tx_contract;rounds=process.retransmission_round
    sim._apply_rt_config_action(ConfigAction(ue_prb_limits={'ue_0':0}),sim.state.slot_id)
    for _ in range(2):sim.step()
    assert len(sent)==1 and ue.harq.retx_deferred_resource>=2
    process=next(p for p in ue.harq.processes if p.tx_contract is not None and p.tx_contract.tb_id==contract.tb_id)
    assert process.tx_contract==contract and process.retransmission_round==rounds
    sim._apply_rt_config_action(ConfigAction(ue_prb_limits={}),sim.state.slot_id)
    for _ in range(3):sim.step()
    assert len(sent)>=2 and sent[0][1]['codeword_id']==sent[1][1]['codeword_id']


def test_energy_utility_and_missing_models_have_honest_validity(tmp_path):
    from ric.xapp_contract import project,XAppContract
    sim=make_sim(tmp_path,cells=1)
    for _ in range(8):obs=sim.step()
    p=sim.state.paper
    r=p.published['cell']['cell_0'];dt=p.dt/1000
    expected=sum(s['energy']['cell_0'] for s in p.samples)
    assert r['rf_energy_j']==pytest.approx(expected)
    assert r['rf_efficiency_bit_per_j']==pytest.approx(sum(s['bits']['cell_0'] for s in p.samples)/expected)
    obs=sim._stamp_observation_delivery(obs,delivered_at_slot=10)
    view,_=project(obs,XAppContract((),frozenset({'tx_power_dbm'})),now_slot=10)
    assert not hasattr(view,'paper_report')
    ue=view.ue_observations[0]
    assert ue.handover_failure_rate is None and ue.kpi_metadata['handover_failure_rate']['reason']=='model_disabled'
    assert ue.requested_dl_rate_bps is None and ue.kpi_metadata['requested_dl_rate_bps']['reason']=='service_requirement_not_configured'
    assert not view.cell_observations[0].kpi_metadata['whole_device_power_w']['valid']


def test_second_recovery_requires_fresh_continuous_good_interval(tmp_path,monkeypatch):
    sim=make_sim(tmp_path,cells=1,events=True,config={'connection_model':{'bad_ms':1,'good_ms':2}})
    p=sim.state.paper;raw=_reference_fixture(sim,monkeypatch)
    def bad():raw['ue_0']['cell_0'].update(no_path=True,rsrp_dbm=-200,reference_sinr_db=-200)
    def good():raw['ue_0']['cell_0'].update(no_path=False,rsrp_dbm=-70,reference_sinr_db=10)
    bad();_tick_model(sim,4);assert 'ue_0' not in p.sessions
    good();_tick_model(sim,6);assert 'ue_0' in p.sessions
    bad();_tick_model(sim,4);assert 'ue_0' not in p.sessions
    good();_tick_model(sim,2);assert 'ue_0' not in p.sessions
    _tick_model(sim,4);assert 'ue_0' in p.sessions
    p.consume_commands([command('disconnect',rid='manual_after_recovery')])
    _tick_model(sim,1)
    p.consume_commands([command('reconnect',rid='reconnect_after_recovery')])
    _tick_model(sim,2);assert 'ue_0' not in p.sessions
    _tick_model(sim,4);assert 'ue_0' in p.sessions


def test_group_power_expansion_and_group_cqi_mapping(tmp_path):
    sim=make_sim(tmp_path,cells=1);n=sim.state.resource_grid.n_prbs_total
    groups=math.ceil(n/4)
    sim._apply_rt_config_action(ConfigAction(power_group_size_prbs=4,prb_power_w=[.4/groups]*groups),0)
    assert sum(sim.state.cell.prb_power_w)==pytest.approx(.4)
    for _ in range(3):sim.step()
    u=sim.state.paper.published['ue']['ue_0']
    assert len(u['subband_cqi'])==groups
    assert sum(u['subband_prb_groups'],[])==list(range(n))


def test_power_profile_validates_post_sleep_budget_and_cap(tmp_path):
    from phy.rt_sys.paper_controls import proposed_power_budget
    sim=make_sim(tmp_path,cells=1);cell=sim.state.cell;n=sim.state.resource_grid.n_prbs_total
    sim._apply_rt_config_action(ConfigAction(prb_power_w=[.4/n]*n),0)
    before=deepcopy(cell)
    action=ConfigAction(cell_sleep_mode='light_sleep')
    assert proposed_power_budget(cell,action)<.4
    sim._apply_rt_config_action(action,0)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='prb_power_exceeds_cell_budget'
    assert cell.tx_power_dbm==before.tx_power_dbm and cell.cell_sleep_mode==before.cell_sleep_mode
    # Explicit clearing allows the original scalar sleep behavior.
    sim._apply_rt_config_action(ConfigAction(clear_prb_power=True,power_group_size_prbs=4,cell_sleep_mode='light_sleep'),0)
    assert cell.prb_power_w is None and cell.cell_sleep_mode=='light_sleep'
    sim._apply_rt_config_action(ConfigAction(power_cap_dbm=20,cell_sleep_mode='micro_sleep',prb_power_w=[.05/(n//4)]*(n//4)),0)
    assert cell.tx_power_dbm<=20
    for _ in range(3):sim.step()
    assert sim.phy_adapter.trace['cells']['cell_0']['average_used_w']<=.05+1e-9
    # A deep-sleep transaction cannot retain a nonzero absolute profile.
    sim._apply_rt_config_action(ConfigAction(cell_sleep_mode='deep_sleep'),sim.state.slot_id)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='prb_power_exceeds_cell_budget'
    sim._apply_rt_config_action(ConfigAction(clear_prb_power=True,cell_sleep_mode='deep_sleep'),sim.state.slot_id)
    decodes=sim.phy_adapter.abstraction.decode_trials
    sim.step()
    assert sim.phy_adapter.abstraction.decode_trials==decodes
    assert sim.phy_adapter.trace['cells']['cell_0']['average_used_w']==0
