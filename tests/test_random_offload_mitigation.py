"""Random conflict choices reach the real service consumer and survive leases."""
from copy import deepcopy
import random
from types import SimpleNamespace as NS

import pytest

from core.simulator import Simulator
from models.action import ConfigAction, ControlDecision
from models.policy import PolicyUpdate
from phy.rt_sys.ue_services import UeServices
from ran.local_executor import LocalExecutor
from ric.controller_manager import ControllerManager
from ric.interfaces.e2_like import E2LikeInterface
from ric.near_rt.pools.base import PoolXApp
from ric.near_rt.pools.chongyu.ue_energy_rf import UeEnergyRF
from ric.near_rt.rule_based_v1.random_arbiter import RandomOffloadArbiter
from tests.test_ue_scheduler_weights import state, observed
from traffic.packet_ledger import PacketLedger


def candidate(source, value, **extra):
    return dict(source=source, value=value, priority=20, request_id=source+'-raw', **extra)


@pytest.mark.parametrize('seed,window,slot', [(True,100,.5),(1,0,.5),(1,float('nan'),.5),(1,100,0),(1,100,.3)])
def test_bad_random_clock_or_seed_is_rejected(seed,window,slot):
    with pytest.raises(ValueError):RandomOffloadArbiter(seed,window,slot)


def test_unique_uniform_requests_hold_one_draw_and_leave_global_rng_unchanged():
    arbiter=RandomOffloadArbiter(9,100,.5)
    candidates=[candidate('z',1.),candidate('a',.5)]
    global_state=random.getstate()
    expected=random.Random('9:direct-conflict-mitigation:ue_offload_fraction:ue_4')
    first,event=arbiter.choose('ue_4','cell_0',0,candidates,.5)
    draw=expected.random()
    assert first['draw']==draw and first['winner']==('a' if draw<.5 else 'z')
    assert first['probability']==.5 and event['reason']=='direct_conflict_random_selection'
    rng=arbiter.rngs['ue_4'].getstate()
    for slot in range(1,200):
        # Re-emitted request IDs do not redraw or mutate the held action.
        again,notice=arbiter.choose('ue_4','cell_0',slot,[dict(c,request_id=str(slot)) for c in candidates],.5)
        assert again==first and notice is None
    assert arbiter.rngs['ue_4'].getstate()==rng and random.getstate()==global_state
    second,_=arbiter.choose('ue_4','cell_0',200,candidates,.5)
    assert second['draw']==expected.random() and second['generation']==2
    with pytest.raises(ValueError):arbiter.choose('ue_4','cell_0',201,[candidates[0],candidates[0]],.5)


def test_only_live_requests_contend_and_expiry_invalidates_within_window():
    arbiter=RandomOffloadArbiter(1,100,.5)
    old=[candidate('a',1.,effective_until_slot=10),candidate('z',.5,effective_from_slot=20)]
    first,_=arbiter.choose('ue_4','cell_0',0,old,.5)
    assert first['winner']=='a' and first['draw'] is None
    restored,_=arbiter.choose('ue_4','cell_0',11,old,.5)
    assert restored['winner']=='rule_withdrawal' and restored['selected_value']==.5
    assert not arbiter.valid_selection(first,11,'cell_0')
    later,_=arbiter.choose('ue_4','cell_0',20,old,.5)
    assert later['winner']=='z' and later['generation']>restored['generation']


def test_other_uid_conflicts_cannot_perturb_a_stable_uid_stream_or_handover_draws():
    independent=RandomOffloadArbiter(7,100,.5)
    interleaved=RandomOffloadArbiter(7,100,.5)
    candidates=[candidate('a',.5),candidate('b',1.)]
    for slot in (0,200,400,600):
        for other in ('ue_0','ue_3','ue_5'):
            interleaved.choose(other,'cell_0',slot,candidates,.5)
        cid='cell_0' if slot<400 else 'cell_1'
        a,_=independent.choose('ue_4',cid,slot,candidates,.5)
        b,_=interleaved.choose('ue_4',cid,slot,list(reversed(candidates)),.5)
        assert a==b and a['draw_index']==slot//200+1
        assert a['stream_key']=='ue_offload_fraction:ue_4'


def test_midwindow_source_update_is_deferred_without_falsely_claiming_new_request():
    arbiter=RandomOffloadArbiter(7,100,.5)
    old=[candidate('a',.5),candidate('b',1.)]
    initial,_=arbiter.choose('ue_4','cell_0',0,old,.5)
    changed=[dict(c,value=.9 if c['source']==initial['winner'] else c['value'],request_id='new-'+c['source']) for c in old]
    held,notice=arbiter.choose('ue_4','cell_0',100,changed,.5)
    assert held==initial and notice is None
    assert held['request_id']==initial['request_id']
    refreshed,_=arbiter.choose('ue_4','cell_0',200,changed,.5)
    assert refreshed['request_id'].startswith('new-')


class Writer(PoolXApp):
    implementation_id='test_writer'
    writable_actions=('ue_offload_fraction',)
    def __init__(self,name,values,priority=20,until=None):
        self.category_id=name
        self.values=values;self.sent=False;self.until=until;self.seen=[]
        super().__init__(['cell_0','cell_1'],name=name)
        self.priority=priority
    def run(self,observation,policy):
        self.seen.append(deepcopy(getattr(observation,'control_ownership',{})))
        decision=ControlDecision();decision.config_actions.clear()
        if not self.sent:
            self.sent=True
            decision.config_actions.append(ConfigAction(source_name=self.name,target_cell_id='cell_0',
                ue_offload_fraction=deepcopy(self.values),effective_until_slot=self.until))
        return decision


def setup(*,mode='uniform_random',seed=1,until=None):
    runtime=state()
    runtime.connections['ue_1'].serving_cell_id='cell_0'
    runtime.control=NS(actuation_outcomes=[])
    for u in runtime.ues:u.packet_ledger=PacketLedger(u,.5)
    runtime.ue_services=UeServices(runtime,{'mec':dict(enabled=True,tasks_by_ue={
        uid:dict(interval_ms=50,compute_gflop=.05,offload_fraction=.5) for uid in ('ue_0','ue_1')})},.5)
    energy=Writer('energy',{'ue_0':1.,'ue_1':.8},priority=100,until=until)
    budget=Writer('budget',{'ue_0':.5},priority=1,until=until)
    bindings={'cell_0':{app.category_id:app.name for app in (energy,budget)}}
    for app in (energy,budget):app.initial_bindings=deepcopy(bindings)
    manager=ControllerManager([],list(map(E2LikeInterface,(energy,budget))),20,100000,
        direct_conflict_mitigation=mode,mitigation_seed=seed,mitigation_window_ms=100,
        mitigation_slot_duration_ms=.5)
    manager._initial_rapp_run_completed=True
    sim=Simulator.__new__(Simulator)
    sim.state=runtime;sim.controller_manager=manager;sim.executor=LocalExecutor()
    sim._rng=random.Random(1);sim._control_reject_prob=0.
    sim._refresh_runtime_views_after_config_apply=lambda:None
    return sim,energy,budget


def tick(sim,slot,*,apply=True):
    sim.state.slot_id=slot
    decision=sim.controller_manager.update(slot,observed(sim.state))
    if apply:
        for action in decision.config_actions:
            if action.ue_offload_fraction:sim._apply_rt_config_action(action,slot)
    return decision


def mitigation_actions(decision):
    return [a for a in decision.config_actions if getattr(a,'_mitigation_records',None)]


def test_random_winner_reaches_actual_service_and_does_not_drop_unrelated_uid():
    sim,energy,budget=setup(seed=0)
    first=tick(sim,0)
    record=sim.controller_manager.direct_conflict_arbiter.current['ue_0']
    assert record['winner']=='budget'  # lower priority really can win
    assert sim.state.ue_services.offload=={'ue_0':.5,'ue_1':.8}
    assert len(mitigation_actions(first))==2
    receipts=sim.state.control.actuation_outcomes
    row=next(r for r in receipts if r.get('mitigation_records',[{}])[0].get('ue_id')=='ue_0')
    ack=row['mitigation_records'][0]
    assert row['status']=='applied' and ack['winner']=='budget'
    assert ack['request_id'] and ack['control_request_id']
    assert row['after']['cell_0']['ue_service_policies']['ue_0']['offload_fraction']==.5
    rng=sim.controller_manager.direct_conflict_arbiter.rngs['ue_0'].getstate()
    for slot in (1,20,40,100,199):tick(sim,slot)
    assert sim.controller_manager.direct_conflict_arbiter.rngs['ue_0'].getstate()==rng
    assert sim.state.ue_services.offload['ue_1']==.8
    owners=energy.seen[-1]['cell_0']
    assert owners['ue_offload_fraction'] is None
    assert owners['ue_offload_fraction_by_ue']=={'ue_0':'budget','ue_1':'energy'}


def test_default_priority_behavior_is_unchanged():
    sim,_,_=setup(mode='priority')
    result=tick(sim,0)
    assert sim.state.ue_services.offload=={'ue_0':1.,'ue_1':.8}
    assert not mitigation_actions(result) and sim.controller_manager.direct_conflict_arbiter is None


def test_selected_owner_does_not_confirm_a_queued_or_rejected_uid():
    sim,energy,_=setup(seed=0)
    decision=tick(sim,0,apply=False)
    tick(sim,20,apply=False)
    assert energy.seen[-1]['cell_0']['ue_offload_fraction_by_ue']=={'ue_0':None,'ue_1':None}
    ue0=next(a for a in mitigation_actions(decision) if 'ue_0' in a.ue_offload_fraction)
    sim._apply_rt_config_action(ue0,20)
    tick(sim,40,apply=False)
    assert energy.seen[-1]['cell_0']['ue_offload_fraction_by_ue']=={'ue_0':'budget','ue_1':None}
    assert energy.seen[-1]['cell_0']['ue_offload_fraction'] is None


def test_withdrawn_native_value_owner_is_released_without_a_fake_control_ack():
    sim,_,_=setup(seed=0)
    tick(sim,0)
    assert sim.controller_manager._mitigation_executed_owners['ue_0']=='budget'
    sim.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=20)
    result=tick(sim,20,apply=False)
    assert 'ue_0' not in sim.controller_manager._mitigation_executed_owners
    assert all('ue_0' not in a.ue_offload_fraction for a in mitigation_actions(result))
    assert any(r['reason']=='direct_conflict_baseline_confirmed' and r['ue_id']=='ue_0'
               for r in sim.controller_manager.rule_decisions)


def test_stale_native_observation_cannot_suppress_restoration_of_newer_applied_value():
    sim,_,_=setup(seed=1)
    tick(sim,0)
    assert sim.state.ue_services.offload['ue_0']==1.
    sim.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=20)
    sim.state.slot_id=20
    stale=observed(sim.state);stale.slot_id=0
    for cell in stale.paper_report['values']['cell'].values():
        cell['executed_config']['ue_offload_fraction']={uid:.5 for uid in cell['executed_config']['ue_offload_fraction']}
    result=sim.controller_manager.update(20,stale)
    restores=mitigation_actions(result)
    assert any(a.ue_offload_fraction=={'ue_0':.5} for a in restores)
    for action in restores:sim._apply_rt_config_action(action,20)
    assert sim.state.ue_services.offload=={'ue_0':.5,'ue_1':.5}


def test_held_winner_has_window_validity_and_stale_queued_action_cannot_apply():
    sim,_,_=setup(seed=1)
    initial=tick(sim,0,apply=False)
    old=next(a for a in mitigation_actions(initial) if 'ue_0' in a.ue_offload_fraction)
    assert old.effective_until_slot==199
    tick(sim,200)
    policy=deepcopy(sim.state.ue_services.offload)
    sim._apply_rt_config_action(old,0)
    assert sim.state.ue_services.offload==policy
    outcome=sim.state.control.actuation_outcomes[-1]
    assert outcome['status']=='rejected' and outcome['reason']=='stale_mitigation_selection'
    assert outcome['mitigation_records'][0]['status']=='rejected'


def test_source_expiry_restores_persistent_baseline_and_retries_until_confirmed():
    sim,_,_=setup(until=30)
    tick(sim,0)
    assert sim.state.ue_services.offload['ue_1']==.8
    pending=tick(sim,40,apply=False)
    restore=next(a for a in mitigation_actions(pending) if 'ue_1' in a.ue_offload_fraction)
    assert restore.ue_offload_fraction=={'ue_1':.5}
    assert restore._mitigation_records[0]['winner']=='rule_withdrawal'
    tick(sim,60)
    assert sim.state.ue_services.offload=={'ue_0':.5,'ue_1':.5}
    final=tick(sim,80)
    assert not mitigation_actions(final)


def test_withdrawal_and_handover_invalidate_queued_winner_preserving_other_uid():
    sim,energy,_=setup()
    decision=tick(sim,0,apply=False)
    queued=next(a for a in mitigation_actions(decision) if 'ue_0' in a.ue_offload_fraction)
    sim.state.connections['ue_0'].serving_cell_id='cell_1'
    sim.state.slot_id=5
    sim._apply_rt_config_action(queued,5)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='mitigation_ue_moved_cell'
    tick(sim,20)
    assert sim.state.ue_services.offload=={'ue_0':.5,'ue_1':.8}
    assert all('ue_0' not in values for (name,cid,field),(_,values,_) in sim.controller_manager.rule_leases.requests.items() if cid=='cell_0')
    sim.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=40)
    tick(sim,40)
    assert sim.state.ue_services.offload=={'ue_0':.5,'ue_1':.5}


def test_withdrawn_or_expired_source_cannot_execute_before_next_controller_tick():
    sim,_,_=setup(until=30)
    decision=tick(sim,0,apply=False)
    queued=next(a for a in mitigation_actions(decision) if 'ue_0' in a.ue_offload_fraction)
    sim.state.slot_id=31
    sim._apply_rt_config_action(queued,31)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='stale_mitigation_selection'
    sim2,_,_=setup()
    decision=tick(sim2,0,apply=False)
    queued=next(a for a in mitigation_actions(decision) if 'ue_0' in a.ue_offload_fraction)
    sim2.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=1)
    sim2.state.slot_id=1
    sim2._apply_rt_config_action(queued,1)
    assert sim2.state.control.actuation_outcomes[-1]['reason']=='mitigation_source_withdrawn'


def test_unsupported_direct_field_contest_fails_explicitly():
    sim,a,b=setup()
    for app,power in ((a,20.),(b,30.)):
        app.controlled_fields=['tx_power_dbm']
        app.contract=app.contract.__class__((),frozenset({'tx_power_dbm'}),cell_ids=app.allowed_cells)
        app.run=lambda obs,policy,app=app,power=power:ControlDecision(config_actions=[
            ConfigAction(source_name=app.name,target_cell_id='cell_0',tx_power_dbm=power)])
    with pytest.raises(ValueError,match='does not support direct conflict on tx_power_dbm'):tick(sim,0)


def test_x01_reads_exact_uid_owner_in_mixed_cell(monkeypatch):
    app=UeEnergyRF(['cell_0'])
    app.active_cells={'cell_0'}
    users=[NS(ue_id=uid,serving_cell_id='cell_0',ue_task_metrics={},ue_task_cohorts=[],
        kpi_metadata={'ue_energy_total_j':{'produced_at_slot':1}}) for uid in ('ue_0','ue_4')]
    app.pending={'ue_0':{},'ue_4':{}}
    seen=[]
    monkeypatch.setattr(app,'_observe',lambda u,produced,owner:seen.append((u.ue_id,owner)))
    app.run(NS(slot_id=1,ue_observations=users,control_ownership={'cell_0':{
        'ue_drx_configs':app.name,'ue_offload_fraction':None,
        'ue_offload_fraction_by_ue':{'ue_0':app.name,'ue_4':'budget'}}}),None)
    assert seen==[('ue_0',app.name),('ue_4','budget')]
