"""AP power arithmetic and actual radio availability, with explicit test profiles."""
from copy import deepcopy
from types import SimpleNamespace as NS
import random

import numpy as np
import pytest

from core.simulator import Simulator
from models.action import ConfigAction
from phy.rt_sys.ap_services import ApServices, radio_available
from phy.rt_sys.control import runtime_control_values
from phy.rt_sys.paper_controls import validate_transaction, proposed_power_budget
from ran.local_executor import LocalExecutor
from ran.scheduler import SimpleScheduler
from tests.test_ue_scheduler_weights import state


def configuration():
    # Analytic fixture values, not claimed hardware calibration.
    return dict(enabled=True, power=dict(static_w=10., load_w=2., pa_efficiency=.5,
        waking_w=15., sleep_w=dict(micro_sleep=2., light_sleep=1., deep_sleep=.5),
        wake_transition_j=dict(micro_sleep=.001, light_sleep=.002, deep_sleep=.003)),
        profiles=dict(micro_sleep=dict(period_ms=20.,sleep_ms=1.,wakeup_ms=0.),
            light_sleep=dict(period_ms=100.,sleep_ms=10.,wakeup_ms=6.),
            deep_sleep=dict(wakeup_ms=50.)))


def model(dt=.5):
    s=state();s.slot_id=0
    s.ap_services=ApServices(s,configuration(),dt)
    return s,s.ap_services


def step(s,m,slot,rf=0.,load=0.):
    s.slot_id=slot;m.begin_slot(slot);m.before_scheduling()
    trace=dict(cells={cid:dict(average_used_w=rf if radio_available(r.cell) else 0.,
        occupied_re=load*100 if radio_available(r.cell) else 0,active_grid_re=100)
        for cid,r in s.managed_cells.items()},ues=[])
    m.after_phy(trace)
    return trace


def policy(s,mode,slot=0,owner='ap_rf',cid='cell_0'):
    s.slot_id=slot
    action=ConfigAction(target_cell_id=cid,cell_sleep_mode=mode,control_request_id='control-'+str(slot))
    action._library_provenances=[dict(instance_id=owner,cell_id=cid,fields=['cell_sleep_mode'])]
    assert validate_transaction(s,action,[cid]) is None
    LocalExecutor().apply_action(s,action)
    return action


def test_active_energy_is_static_plus_actual_load_and_pa_and_no_radio_changes():
    s,m=model()
    for slot in range(1,2001):trace=step(s,m,slot,3.,.5)
    for report in m.reports.values():
        assert report['ap_energy_total_j']==pytest.approx(17.)
        assert report['ap_energy_components_j']==pytest.approx(dict(static=10.,load=1.,pa=6.,sleep=0.,waking=0.,transition=0.))
        assert report['ap_available_fraction']==1. and report['ap_available_slots_total']==2000
    assert m.summary()['network_ap_energy_total_j']==pytest.approx(34.)
    assert all(r.cell.tx_power_dbm==30. for r in s.managed_cells.values())
    assert trace['ap_services']['cells']['cell_0']['ap_power_w']==pytest.approx(17.)


@pytest.mark.parametrize('mode,sleep,waking,active,wakes',[
    ('micro_sleep',100,0,1900,50),('light_sleep',200,120,1680,10)])
def test_approved_different_periods_integrate_exact_duty_wake_and_transition(mode,sleep,waking,active,wakes):
    s,m=model();policy(s,mode)
    for slot in range(1,2001):step(s,m,slot)
    r=m.reports['cell_0'];p=configuration()['power']
    assert (r['ap_sleep_slots_total'],r['ap_waking_slots_total'],r['ap_available_slots_total'])==(sleep,waking,active)
    assert r['ap_wake_transitions_total']==wakes
    assert r['ap_available_fraction']==active/2000
    expected=(sleep*p['sleep_w'][mode]+waking*p['waking_w']+active*p['static_w'])*.0005+wakes*p['wake_transition_j'][mode]
    assert r['ap_energy_total_j']==pytest.approx(expected)
    assert r['ap_energy_components_j']['transition']==pytest.approx(wakes*p['wake_transition_j'][mode])
    assert r['ap_cycle_period_slots']==200 and r['ap_cycle_phase_slot']==199
    assert s.cell.tx_power_dbm==30.  # no TX dimming during awake portions


def test_deep_sleep_consumes_explicit_standby_and_active_request_waits_real_wakeup():
    s,m=model();policy(s,'deep_sleep')
    for slot in range(1,11):step(s,m,slot)
    assert m.reports['cell_0']['ap_energy_total_j']==pytest.approx(.5*.005)
    policy(s,'active',11)
    assert not s.cell.ap_available and s.cell.ap_runtime_state=='waking'
    for slot in range(11,111):
        if slot%20==0:policy(s,'active',slot)
        step(s,m,slot)
        assert not s.cell.ap_available
    step(s,m,111)
    assert s.cell.ap_available
    r=m.reports['cell_0']
    assert r['ap_wake_transitions_total']==1 and r['ap_waking_slots_total']==100
    assert r['ap_policy_applied_slot']==11
    assert r['ap_energy_components_j']['transition']==pytest.approx(.003)


def test_actual_ownership_ack_same_value_transfer_and_reissue_preserve_cycle():
    s,m=model();assert m.reports['cell_0']['ap_policy_owner'] is None
    policy(s,'active',0,owner='first')
    first=runtime_control_values(s,['cell_0'])['cell_0']
    assert first['ap_policy_owner']=='first' and first['ap_policy_applied_slot']==0
    policy(s,'active',5,owner='first')
    repeated=m.policy_receipt('cell_0')
    assert repeated['ap_policy_generation']==first['ap_policy_generation']
    assert repeated['ap_policy_request_id']==first['ap_policy_request_id']
    assert repeated['ap_policy_applied_slot']==0
    policy(s,'active',6,owner='second')
    transferred=m.policy_receipt('cell_0')
    assert transferred['ap_policy_generation']==first['ap_policy_generation']+1
    assert transferred['ap_policy_applied_slot']==6 and transferred['ap_policy_owner']=='second'
    assert transferred['ap_cycle_start_slot']==1


def test_sleep_does_not_reset_queue_or_harq_and_scheduler_emits_no_demands():
    s,m=model();u=s.ues[0];u.harq.processes=[NS(marker='retained')]
    before_queue=u.queue_bytes
    policy(s,'light_sleep')
    demands,users,trace=SimpleScheduler()._build_scheduling_demands(s)
    assert not demands and trace.filtered_out_reasons=={'ue_0':'ap_unavailable','ue_1':'ap_unavailable'}
    assert u.queue_bytes==before_queue and u.harq.processes[0].marker=='retained'
    assert SimpleScheduler()._get_eligible_ues(s)==[]


def test_power_validation_uses_full_awake_budget_with_persistent_prb_vector():
    s,m=model();s.cell.prb_power_w=[.1]*8
    action=ConfigAction(target_cell_id='cell_0',cell_sleep_mode='light_sleep')
    assert proposed_power_budget(s.cell,action)==pytest.approx(1.)
    assert validate_transaction(s,action,['cell_0']) is None
    LocalExecutor().apply_action(s,action)
    assert s.cell.tx_power_dbm==30. and s.cell.prb_power_w==[.1]*8
    before=s.cell.tx_power_dbm
    bad=ConfigAction(target_cell_id='cell_0',cell_sleep_mode='not_a_mode',tx_power_dbm=20.)
    assert validate_transaction(s,bad,['cell_0'])=='invalid_ap_sleep_mode'
    assert s.cell.tx_power_dbm==before
    capped=ConfigAction(target_cell_id='cell_0',cell_sleep_mode='light_sleep',power_cap_dbm=40.,prb_power_w=[1.]*8)
    assert proposed_power_budget(s.cell,capped)==pytest.approx(10.)
    assert validate_transaction(s,capped,['cell_0']) is None
    LocalExecutor().apply_action(s,capped)
    assert s.cell.tx_power_dbm==40. and s.cell.nominal_tx_power_dbm==40.


def test_disabled_executor_retains_legacy_sleep_dimmer_exactly():
    s=state();executor=LocalExecutor()
    for mode,expected in [('micro_sleep',27.),('active',30.),('light_sleep',22.),('active',30.),('deep_sleep',0.)]:
        executor.apply_action(s,ConfigAction(target_cell_id='cell_0',cell_sleep_mode=mode))
        assert s.cell.tx_power_dbm==expected
    assert not hasattr(s.cell,'ap_available') and not hasattr(s,'ap_services')


def test_unavailable_radio_emission_and_duplicate_energy_accounting_fail_closed():
    s,m=model();policy(s,'deep_sleep');s.slot_id=1;m.begin_slot(1)
    trace=dict(cells={c:dict(average_used_w=1.,occupied_re=1,active_grid_re=100) for c in s.managed_cells},ues=[])
    with pytest.raises(ValueError,match='unavailable_ap_emitted'):m.after_phy(trace)
    step(s,m,1)
    with pytest.raises(ValueError,match='exactly_once'):m.after_phy(trace)


@pytest.mark.parametrize('change',[
    lambda c:c.update(enabled=False),lambda c:c.update(unrecognized=1),
    lambda c:c['power'].update(pa_efficiency=0),lambda c:c['power'].update(static_w=True),
    lambda c:c['profiles']['light_sleep'].update(wakeup_ms=float('nan')),
    lambda c:c['profiles']['light_sleep'].update(sleep_ms=100),
    lambda c:c['profiles']['micro_sleep'].update(period_ms=20.1)])
def test_invalid_explicit_model_assumptions_are_rejected(change):
    c=configuration();change(c)
    with pytest.raises(ValueError):ApServices(state(),c,.5)


def simulator(tmp_path, *, ap=True, abstraction=None, cells=1):
    """Actual pipeline, analytic channel/decoder fixtures; no NR/RT computation."""
    from core.factory import build_controller_registry,build_scheduler
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    from phy.rt_sys.scenario import build_case
    from phy.rt_sys.channel import RtChannelProvider
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from tests.test_rt_sys import FakeTracer,FixtureAbstraction
    manifest,configs,_=build_case(num_cells=cells,num_ues=1,total_slots=50,seed=512,initial_queue_bytes=10000,ue_speed_mps=0)
    configs['ric'].update(rapps=[],xapps=[])
    configs['sim']['measurement_sinr_noise_std_db']=0
    config=deepcopy(configs['phy']['rt_sys'])
    config['paper_interfaces']={'enabled':True,'connection_events':False,'rule_observations':True}
    if ap is not None:config['ap_services']=configuration() if ap else {'enabled':False}
    provider=RtChannelProvider(manifest,tmp_path/'cache',backend=FakeTracer(manifest))
    adapter=RtSysPhyAdapter({**config,'device':'numpy'},abstraction=abstraction or FixtureAbstraction(),channel_provider=provider)
    runtime=build_runtime_scenario_config(scenario_name='ap_fixture',scenario_type='rt_scene',sim_config=configs['sim'],cell_config=configs['cell'],ric_config=configs['ric'])
    return Simulator(runtime,build_controller_registry(configs['ric']),build_scheduler(configs['sim']),adapter,
        configs['cell'],configs['ue'],configs['beamforming'],str(tmp_path/'run'))


def test_actual_pipeline_sleep_wake_reference_and_pdsch_are_consistent(tmp_path):
    sim=simulator(tmp_path);s=sim.state
    valid_reference=deepcopy(s.paper.filtered['ue_0','cell_0'])
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_0',cell_sleep_mode='light_sleep'),0)
    seen=[]
    for _ in range(36):
        sim.step();trace=sim.phy_adapter.trace;ap=trace['ap_services']['cells']['cell_0']
        seen.append(ap['ap_state'])
        if not ap['ap_available']:
            assert trace['cells']['cell_0']['average_used_w']==0.
            assert not any(u['scheduled'] for u in trace['ues'])
            assert s.paper.raw['ue_0']['cell_0']['no_path']
            assert s.paper.filtered['ue_0','cell_0']==valid_reference
            assert s.paper.observed['ue_0']['cell_0']['ap_reference_unavailable']
            public=s.paper.published['ue']['ue_0']
            assert public['reference_measurement_available'] is False
            assert public['candidate_reference_available']=={'cell_0':False}
            assert public['candidate_sinr_db']=={} and public['candidate_rsrp_dbm']=={}
            assert not s.scheduler.scheduled_ue_ids
        else:
            assert not s.paper.raw['ue_0']['cell_0']['no_path']
            assert s.paper.filtered['ue_0','cell_0']==valid_reference
            assert not s.paper.observed['ue_0']['cell_0']['ap_reference_unavailable']
            public=s.paper.published['ue']['ue_0']
            assert public['reference_measurement_available'] is True
            assert public['candidate_reference_available']=={'cell_0':True}
            assert public['candidate_sinr_db']['cell_0']==valid_reference['reference_sinr_db']
            assert s.cell.tx_power_dbm==30.
    assert seen==['sleeping']*20+['waking']*12+['active']*4
    assert s.ues[0].packet_ledger.stats['delivered_bytes']>0
    stats=s.ues[0].packet_ledger.stats
    assert stats['initial_bytes']+stats['offered_bytes']==stats['delivered_bytes']+stats['pending_bytes']+stats['overflow_bytes']
    assert s.paper.published['cell']['cell_0']['ap_energy_total_j']==s.ap_services.reports['cell_0']['ap_energy_total_j']
    assert 'ap_services' in sim.slot_records[-1]['rt_sys']


def test_explicit_disabled_model_is_byte_identical_to_absent_model(tmp_path):
    absent=simulator(tmp_path/'absent',ap=None);disabled=simulator(tmp_path/'disabled',ap=False)
    for _ in range(5):
        absent.step();disabled.step()
        assert absent.phy_adapter.trace==disabled.phy_adapter.trace
        assert absent.state.paper.filtered==disabled.state.paper.filtered
        assert 'reference_measurement_available' not in disabled.state.paper.published['ue']['ue_0']
        assert not hasattr(disabled.state,'ap_services')
    assert absent.state.ues[0].packet_ledger.stats==disabled.state.ues[0].packet_ledger.stats


def test_enabled_all_active_model_preserves_original_physics_and_arrivals(tmp_path):
    native=simulator(tmp_path/'native',ap=None);active=simulator(tmp_path/'active')
    for _ in range(8):
        native.step();active.step()
        for key in ('cells','ues','precoders'):
            assert native.phy_adapter.trace[key]==active.phy_adapter.trace[key]
        assert native.state.ues[0].packet_ledger.stats==active.state.ues[0].packet_ledger.stats
    assert active.state.ap_services.summary()['network_ap_energy_total_j']>0


def test_pending_actual_harq_survives_short_sleep_then_retries_without_duplicate_payload(tmp_path):
    from tests.test_rt_sys import FixtureAbstraction
    class SmallTb(FixtureAbstraction):
        fail=True
        def evaluate(self,*args,**kwargs):
            result=super().evaluate(*args,**kwargs)
            result.update(capacity_bytes=7,ack=not self.fail,tbler=1. if self.fail else 0.)
            return result
    decoder=SmallTb();sim=simulator(tmp_path,abstraction=decoder)
    sim.step();u=sim.state.ues[0]
    assert u.harq.pending_retx_bytes==7 and u.harq.processes
    before=[(p.tb_id,p.pending_bytes) for p in u.harq.processes]
    action=ConfigAction(target_cell_id='cell_0',cell_sleep_mode='micro_sleep')
    sim._pending_config_actuations.append((2,0,action))
    sim.step()
    assert not sim.state.cell.ap_available and not sim.phy_adapter.trace['ues'][0]['scheduled']
    assert [(p.tb_id,p.pending_bytes) for p in u.harq.processes]==before
    decoder.fail=False;sim.step()
    assert sim.state.cell.ap_available and u.harq.pending_retx_bytes==0
    assert sim.phy_adapter.trace['ues'][0]['is_retx']
    assert sim.phy_adapter.trace['ues'][0]['delivered_bytes']==7
    stats=u.packet_ledger.stats
    assert stats['initial_bytes']+stats['offered_bytes']==stats['delivered_bytes']+stats['pending_bytes']+stats['overflow_bytes']


def test_missing_first_reference_does_not_create_an_ewma_sample_or_valid_beams(tmp_path):
    sim=simulator(tmp_path);s=sim.state
    s.paper.config['beam_reports']=True
    s.paper.filtered.clear()
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_0',cell_sleep_mode='micro_sleep'),0)
    sim.step()
    assert ('ue_0','cell_0') not in s.paper.filtered
    assert s.paper.observed['ue_0']['cell_0']['no_path']
    assert all(not x['valid'] and x['ap_reference_unavailable'] for x in s.paper.observed['ue_0']['cell_0']['beam_reference_reports'])
    sim.step();sim.step()
    first=s.paper.filtered['ue_0','cell_0']
    assert not first['no_path'] and not first['ap_reference_unavailable']
    assert first['reference_sinr_db']==s.paper.raw['ue_0']['cell_0']['reference_sinr_db']
    assert first['rsrp_dbm']==s.paper.raw['ue_0']['cell_0']['rsrp_dbm']


def test_model_off_keeps_legacy_sleep_reference_filter_semantics(tmp_path):
    sim=simulator(tmp_path,ap=None);s=sim.state
    previous=deepcopy(s.paper.filtered['ue_0','cell_0'])
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_0',cell_sleep_mode='deep_sleep'),0)
    sim.step()
    current=s.paper.filtered['ue_0','cell_0']
    alpha=1-np.exp(-sim.config.slot_duration_ms/20.)
    for key in ('rsrp_dbm','reference_sinr_db'):
        expected=previous[key]+alpha*(s.paper.raw['ue_0']['cell_0'][key]-previous[key])
        assert current[key]==pytest.approx(expected)
    assert 'ap_reference_unavailable' not in current
    assert 'reference_measurement_available' not in s.paper.published['ue']['ue_0']


def reference_ranking(sim,slot,monkeypatch):
    s=sim.state;p=s.paper;s.slot_id=slot;p.committed=-1
    monkeypatch.setattr(p,'prepare_reference',lambda:None)
    for cid,rsrp in [('cell_0',-30.),('cell_1',-50.),('cell_2',-80.)]:
        if cid not in s.managed_cells:continue
        sample=deepcopy(p.raw['ue_0'][cid]);sample.update(rsrp_dbm=rsrp,reference_sinr_db=10.,no_path=False)
        p.filtered['ue_0',cid]=deepcopy(sample)
        if cid=='cell_0':
            sample['no_path']=True
            if getattr(s,'ap_services',None) is not None:sample['ap_reference_unavailable']=True
        p.raw['ue_0'][cid]=sample
    return s,p


def test_unavailable_strongest_ap_does_not_shadow_available_initial_attachment(tmp_path,monkeypatch):
    sim=simulator(tmp_path,cells=2)
    s,p=reference_ranking(sim,0,monkeypatch)
    p.before_scheduling()
    assert p.observed['ue_0']['cell_0']['rsrp_dbm']>p.observed['ue_0']['cell_1']['rsrp_dbm']
    assert s.connections['ue_0'].serving_cell_id=='cell_1'


def test_unavailable_strongest_ap_does_not_shadow_available_handover_target(tmp_path,monkeypatch):
    sim=simulator(tmp_path,cells=3)
    s,p=reference_ranking(sim,1,monkeypatch)
    connection=s.connections['ue_0'];connection.serving_cell_id='cell_2';connection.last_handover_slot=None
    cell=s.managed_cells['cell_2'].cell
    cell.time_to_trigger_ms=0.;cell.steering_hysteresis_db=cell.handover_cost_db=cell.handover_margin_db=0.
    p.before_scheduling()
    assert p.pending['ue_0']['to_cell']=='cell_1'


def test_model_off_keeps_original_initial_ranking(tmp_path,monkeypatch):
    sim=simulator(tmp_path,ap=None,cells=2)
    s,p=reference_ranking(sim,0,monkeypatch)
    p.before_scheduling()
    assert s.connections['ue_0'].serving_cell_id=='cell_0'
