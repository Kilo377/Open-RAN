from pathlib import Path
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
from phy.rt_sys.scenario import build_case
from phy.rt_sys.contracts import digest


def test_shared_initial_conditions_and_clocks_unchanged():
    from core.config_loader import load_all_configs
    _,a,_=build_case(num_cells=3,num_ues=24,initial_queue_bytes=0)
    _,b,_=build_case(num_cells=3,num_ues=24,initial_queue_bytes=0)
    assert a['ue']==b['ue']
    assert a['phy']['rt_sys']['initial_association']=='strongest_rt_reference'
    assert all(u['queue_bytes']==0 and all(lc['queue_bytes']==0 for lc in u.get('logical_channels',[])) for u in a['ue']['ues'])
    assert a['ric']['xapp_interval_slots']==5 and a['ric']['rapp_interval_slots']==10
    _,c,_=build_case(initial_queue_bytes=123)
    assert all(u['queue_bytes']==123 for u in c['ue']['ues'])
    assert all(sum(l['queue_bytes'] for l in u.get('logical_channels',[]))==123 for u in c['ue']['ues'] if u.get('logical_channels'))


def test_static_geometry_is_exact_not_moving_approximation():
    from phy.rt_sys.channel import geometry_blocks
    m,_,_=build_case(total_slots=10000,ue_speed_mps=0)
    assert geometry_blocks(m)==([0],10001)
    moving,_,_=build_case(total_slots=10000,ue_speed_mps=1.5)
    assert len(geometry_blocks(moving)[0])==10001


class FastWave:
    table=1;mcs=3;data_re=24;coding_backend='test_fixture';q=2;rate=.5;bits=24;num_cb=1
    def __init__(self):self.calls=0
    def simulate_fast(self,snrs,values,n,seed,**kwargs):self.calls+=1;return np.array([0])


def test_probability_bank_independent_admission_domain_and_rejection(tmp_path):
    from phy.rt_sys.probability_bank import ProbabilityBank
    bank=ProbabilityBank();w=FastWave();v=np.ones(24)*100
    first=bank.probe(w,v);assert first['validation']['passed'] and w.calls==3
    assert bank.probe(w,v)['key']==first['key'] and w.calls==3
    changed=v.copy();changed[0]*=2
    assert bank.probe(w,changed)['key']!=first['key'] and w.calls==6
    bad=FastWave()
    def unstable(snrs,v,n,seed,**kwargs):
        bad.calls+=1;return np.array([0 if bad.calls==1 else n])
    bad.simulate_fast=unstable
    other=ProbabilityBank();assert other.probe(bad,v) is None
    assert other.probe(bad,v) is None and bad.calls==6


def test_real_same_codeword_chase_llr_and_resource_guard():
    from phy.rt_sys.calibration import NRWaveform
    from phy.rt_sys.nr_harq import NrHarqReceiver
    import torch
    w=NRWaveform(1,3,324,'cpu');h=NrHarqReceiver('chase')
    ctx=dict(process_id=1,ndi=1,is_retx=False,cell_id='c0')
    first=h.transmit(w,np.ones(324)*.5,12,1,'u',ctx)
    bits=h.pending[('u',1)]['bits'].clone();llr=h.pending[('u',1)]['llr'].clone()
    ctx['is_retx']=True
    second=h.transmit(w,np.ones(324)*100,12,2,'u',ctx)
    torch.testing.assert_close(h.pending[('u',1)]['bits'],bits)
    assert first['codeword_id']==second['codeword_id'] and second['ack']
    assert second['reason']=='chase_llr_combined' and not torch.equal(h.pending[('u',1)]['llr'],llr)
    other=NRWaveform(1,3,336,'cpu')
    mismatch=h.transmit(other,np.ones(336)*100,12,3,'u',ctx)
    assert mismatch['no_tb'] and h.pending[('u',1)]['rounds']==2
    h.acknowledge('u',1);assert not h.pending


def make_sim(tmp_path,scene='simple_street_canyon',mode='baseline'):
    from test_rt_sys import FakeTracer,FixtureAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    from phy.rt_sys.channel import RtChannelProvider
    from core.simulator import Simulator
    from core.factory import build_controller_registry,build_scheduler
    from experiments.scenario_scan_utils import build_runtime_scenario_config
    m,c,_=build_case(scene,num_cells=3,num_ues=4,total_slots=32,initial_queue_bytes=1000)
    c['ric'].update(rapps=[],xapps=[])
    adapter=RtSysPhyAdapter({**c['phy']['rt_sys'],'device':'numpy'},abstraction=FixtureAbstraction(),
        channel_provider=RtChannelProvider(m,tmp_path/'cache',backend=FakeTracer(m)))
    runtime=build_runtime_scenario_config(scenario_name='readiness',scenario_type='rt_scene',
        sim_config=c['sim'],cell_config=c['cell'],ric_config=c['ric'])
    return Simulator(runtime,build_controller_registry(c['ric']),build_scheduler(c['sim']),adapter,
        c['cell'],c['ue'],c['beamforming'],str(tmp_path))


def test_rt_action_future_scope_expiry_unknown_and_unsupported(tmp_path):
    from models.action import ConfigAction,ControlDecision,ControlMetadata
    sim=make_sim(tmp_path)
    actions=[ConfigAction(target_cell_id='cell_1',power_cap_dbm=20,effective_from_slot=5,
              metadata=ControlMetadata(source='future',issued_at_slot=1)),
             ConfigAction(target_cell_id='missing',power_cap_dbm=10,metadata=ControlMetadata(source='bad',issued_at_slot=1)),
             ConfigAction(target_cell_id='cell_0',muting_ratio=.5,metadata=ControlMetadata(source='unsupported',issued_at_slot=1)),
             ConfigAction(target_cell_id='cell_0',power_cap_dbm=10,effective_until_slot=1,
                          metadata=ControlMetadata(source='expired',issued_at_slot=1))]
    sim.controller_manager.update=lambda slot_id,observation:ControlDecision(config_actions=actions if slot_id==1 else [])
    outcomes=[]
    for _ in range(6):
        sim.step();outcomes.extend(sim.state.control.actuation_outcomes)
        if sim.state.slot_id<5:assert sim.state.managed_cells['cell_1'].cell.tx_power_dbm==30
    assert sim.state.managed_cells['cell_1'].cell.tx_power_dbm==20
    assert sim.state.managed_cells['cell_0'].cell.tx_power_dbm==30
    by={x['source']:x for x in outcomes}
    assert by['future']['status']=='applied' and by['future']['apply_slot']==5
    assert by['bad']['reason']=='unknown_target_cell'
    assert by['unsupported']['reason']=='unsupported_fields:muting_ratio'
    assert by['expired']['reason']=='expired_before_actuation'
    assert by['future']['target_cell_ids']==['cell_1']
    assert by['future']['before']['cell_1']['tx_power_dbm']==30
    assert by['future']['after']['cell_1']['tx_power_dbm']==20


def test_statistical_bank_never_claims_actual_crc_or_soft_combining():
    from phy.rt_sys.reference import StatisticalBankAbstraction
    from phy.rt_sys.adapter import RtSysPhyAdapter
    with pytest.raises(ValueError,match='cannot claim waveform'):
        RtSysPhyAdapter({'device':'cpu','feedback_mode':'nr_bler_bank','harq_combining':'chase'})
    model=object.__new__(StatisticalBankAbstraction);model.statistics={}
    result,_=model._transmitted_feedback(None,None,1,1,'u',0,None,
        {'source':'independently_validated_local_bler_bank','errors':0,'samples':256})
    assert result==0 and model.statistics['statistical_tb_samples']==1


def test_arbitration_preserves_winning_field_validity_and_reports_loser(tmp_path):
    from scripts.validation.validate_control_lifecycle import PowerProbe
    from ric.interfaces.e2_like import E2LikeInterface
    sim=make_sim(tmp_path)
    sim.controller_manager.e2_interfaces=[E2LikeInterface(PowerProbe('low',1,25)),E2LikeInterface(PowerProbe('high',20,20))]
    outcomes=[];dropped=[]
    for _ in range(25):
        sim.step();outcomes.extend(sim.state.control.actuation_outcomes)
        dropped.extend(sim.state.control.arbitration_trace.dropped_items_with_reason)
        if sim.state.slot_id<15:assert sim.state.managed_cells['cell_1'].cell.tx_power_dbm==30
    assert sim.state.managed_cells['cell_1'].cell.tx_power_dbm==20
    assert any(o['apply_slot']==15 and o['status']=='applied' for o in outcomes)
    assert any(o.get('reason')=='expired_before_actuation' for o in outcomes)
    assert any(d.source=='low' and d.reason=='superseded_by_higher_priority' for d in dropped)


def test_information_screening_rejects_provably_impossible_candidates_before_decode(monkeypatch):
    from phy.rt_sys.reference import StatisticalBankAbstraction
    a=StatisticalBankAbstraction('cpu')
    called=[]
    monkeypatch.setattr(a,'waveform',lambda *args:called.append(args))
    r=a.evaluate(np.ones(324)*1e-12,'qam64',27,1,1,'u')
    assert not called and r['capacity_bytes']==0 and r['ack_source']=='capacity_outage_no_tb'
    assert a.statistics['information_rejected_candidates']==28
    assert all(p['samples']==0 for p in r['amc_probes'])
    assert a.information_lower_bound(1,3,324,np.ones(324)*100)==0


def test_nonprimary_association_controls_change_only_serving_ues_and_can_block_handover(tmp_path):
    from models.action import ConfigAction
    sim=make_sim(tmp_path)
    manager=sim.mobility_manager
    manager.noise=0;manager.filter_ms=0;manager.ttt_slots=0;manager.dwell_slots=0;manager.delay=0
    # Equal reference channels and equal loads isolate the association knobs.
    manager._build_cell_load_view=lambda state,profiles:{cid:dict(estimated_load=0.,load_penalty_db=0.,
        activity_factor=1.,connected_ue_count=0,scheduled_ue_count=0,queued_bytes=0) for cid in profiles}
    user=sim.state.ues[0];connection=sim.state.connections[user.ue_id]
    candidate=next(c for c in connection.candidate_cells if c.cell_id=='cell_1')
    manager._apply_serving_candidate(user,sim.state.links[user.ue_id],connection,candidate)
    for other in sim.state.ues[1:]:
        conn=sim.state.connections[other.ue_id]
        manager._apply_serving_candidate(other,sim.state.links[other.ue_id],conn,
            next(c for c in conn.candidate_cells if c.cell_id=='cell_0'))
    reference_hysteresis=sim.state.managed_cells['cell_0'].cell.steering_hysteresis_db
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_1',traffic_steering_preference='edge_offload',
        mobility_bias_db=30.,steering_hysteresis_db=100.),0)
    for _ in range(3):sim.step()
    assert connection.serving_cell_id=='cell_1' and connection.steering_hysteresis_db==100
    assert all(c.steering_hysteresis_db==reference_hysteresis for uid,c in sim.state.connections.items() if uid!=user.ue_id)
    assert sim.state.managed_cells['cell_0'].cell.mobility_bias_db==0
    sim._apply_rt_config_action(ConfigAction(target_cell_id='cell_1',steering_hysteresis_db=0.),3)
    for _ in range(5):sim.step()
    assert connection.serving_cell_id=='cell_0' and connection.handover_count==1
    assert next(row for row in sim.phy_adapter.trace['ues'] if row['ue_id']==user.ue_id)['cell_id']=='cell_0'


@pytest.mark.parametrize('strategy',['frequency_first','time_first'])
@pytest.mark.parametrize('mu',[False,True])
def test_rt_packing_uses_cartesian_grid_without_cross_group_overlap(strategy,mu):
    from phy.rt_sys.scheduler import RtSysScheduler
    from models.scheduling import UeGrant
    scheduler=RtSysScheduler(total_prbs_per_slot=24,total_symbols_per_slot=14,
        min_prbs_per_grant=4,min_symbols_per_grant=4,max_ues_per_slot=8)
    state=NS(scheduler=NS(packing_strategy=strategy))
    users={f'u{k}':NS(ue_id=f'u{k}',ri=1) for k in range(3)}
    grants=[UeGrant(uid,4,4,requested_bytes=100000,priority_score=1,target_mcs=12) for uid in users]
    if mu:
        grants[0].spatial_group_id=0;grants[1].spatial_group_id=0
    scheduler._repack_rt_groups(state,grants,users)
    rectangles=[{(p,s) for p in g.resource_block_indices for s in g.ofdm_symbol_indices} for g in grants]
    assert len(set.union(*rectangles))==24*14
    assert rectangles[0]&rectangles[1]==(rectangles[0] if mu else set())
    assert not rectangles[0]&rectangles[2] and not rectangles[1]&rectangles[2]
    assert all(0<g.grant_bytes_budget<=g.requested_bytes for g in grants)
    scheduler._rt_packing=strategy
    assert scheduler._max_grants_allowed(8)==(6 if strategy=='frequency_first' else 3)


def test_rt_candidate_load_counts_secondary_cell_scheduling(tmp_path):
    sim=make_sim(tmp_path);sim.step()
    sim.mobility_manager.load_filter_ms=0
    view=sim.mobility_manager._build_cell_load_view(sim.state,sim.mobility_manager._cell_radio_profiles(sim.state))
    assert any(r.scheduler.scheduled_ue_ids for cid,r in sim.state.managed_cells.items() if cid!='cell_0')
    for cid,runtime in sim.state.managed_cells.items():
        assert view[cid]['scheduled_ue_count']==len(runtime.scheduler.scheduled_ue_ids)
