"""Atomic scheduler arbitration, real execution and lifecycle regression tests."""
from copy import deepcopy
import random
from types import SimpleNamespace as NS

import pytest

from core.simulator import Simulator
from models.action import ConfigAction, ControlDecision
from models.policy import PolicyUpdate
from ran.local_executor import LocalExecutor
from ric.controller_manager import ControllerManager
from ric.interfaces.e2_like import E2LikeInterface
from ric.near_rt.pools.base import PoolXApp
from ric.near_rt.rule_based_v1.random_arbiter import RandomSchedulerArbiter, RandomOffloadArbiter
from tests.test_ue_scheduler_weights import state, observed


def bundle(mode='latency_aware', **weights):
    return dict(scheduler_mode=mode, ue_scheduler_weights=weights)


def candidate(name, value, **extra):
    return dict(source=name, value=value, priority=20, request_id=name+'-raw',
        framework_tuning={'latency_bias': 1.} if value['scheduler_mode']=='latency_aware' else {}, **extra)


def test_independent_cell_streams_select_unchanged_whole_requests_once_per_window():
    one, many = RandomSchedulerArbiter(7,100,.5), RandomSchedulerArbiter(7,100,.5)
    choices=[candidate('a',bundle(ue_0=1.5,ue_1=1.)),
             candidate('b',bundle('round_robin',ue_0=1.,ue_1=2.))]
    expected=random.Random('7:direct-conflict-mitigation:scheduler_bundle:cell_0')
    global_state=random.getstate()
    offload=RandomOffloadArbiter(7,100,.5)
    for slot in (0,200,400):
        many.choose('cell_1',slot,choices,None,('ue_0','ue_1'))
        offload.choose('ue_0','cell_0',slot,[dict(source='a',priority=1,value=.5),dict(source='b',priority=1,value=1.)],.5)
        a,event=one.choose('cell_0',slot,choices,None,('ue_0','ue_1'))
        b,_=many.choose('cell_0',slot,list(reversed(choices)),None,('ue_1','ue_0'))
        assert a==b and a['draw']==expected.random()
        assert a['selected_value']==next(c['value'] for c in choices if c['source']==a['winner'])
        assert a['draw_index']==slot//200+1 and a['probability']==.5
        assert a['stream_key']=='scheduler_bundle:cell_0' and event['field']=='scheduler_bundle'
        for offset in (1,20,199):
            held,notice=one.choose('cell_0',slot+offset,[dict(c,request_id='refresh') for c in choices],None,('ue_0','ue_1'))
            assert held==a and notice is None
    assert random.getstate()==global_state


def test_new_same_source_value_defers_but_membership_or_validity_revokes_whole_snapshot():
    arb=RandomSchedulerArbiter(2,100,.5)
    first,_=arb.choose('c',0,[candidate('a',bundle(ue_0=1.5),effective_until_slot=120)],None,['ue_0'])
    changed=candidate('a',bundle('round_robin',ue_0=1.),request_marker='new')
    held,_=arb.choose('c',100,[changed],None,['ue_0'])
    assert held==first and held['request_id']=='a-raw'
    refreshed,_=arb.choose('c',121,[changed],None,['ue_0'])
    assert refreshed['selected_value']==changed['value'] and not arb.valid_selection(first,121,'c')
    moved,_=arb.choose('c',122,[candidate('a',bundle())],None,[])
    assert moved['serving_ue_ids']==() and moved['selected_value']['ue_scheduler_weights']=={}
    assert not arb.valid_selection(refreshed,122,'c')
    with pytest.raises(ValueError):arb.choose('c',123,[changed,changed],None,['ue_0'])


class Writer(PoolXApp):
    implementation_id='test_scheduler_writer'
    writable_actions=('scheduler_mode','ue_scheduler_weights')
    def __init__(self,name,values,priority=20,until=None):
        self.category_id=name
        self.values=values;self.sent=False;self.until=until;self.seen=[]
        super().__init__(['cell_0','cell_1'],name=name)
        self.priority=priority
    def run(self,observation,policy):
        self.seen.append(deepcopy(getattr(observation,'control_ownership',{})))
        result=ControlDecision();result.config_actions.clear()
        if not self.sent:
            self.sent=True
            for cid,values in self.values.items():
                result.config_actions.append(ConfigAction(source_name=self.name,target_cell_id=cid,
                    effective_until_slot=self.until,**deepcopy(values)))
        return result


def setup(*,seed=0,mode='uniform_random',until=None,empty=False):
    runtime=state();runtime.control=NS(actuation_outcomes=[])
    runtime.connections['ue_1'].serving_cell_id='cell_0'
    a={'cell_0':bundle(ue_0=1.5,ue_1=1.)}
    b={'cell_0':bundle(ue_0=1.,ue_1=2.)}
    if empty:a['cell_1']=bundle();b['cell_1']=bundle('round_robin')
    first=Writer('a',a,100,until);second=Writer('b',b,1,until)
    bindings={cid:{app.category_id:app.name for app in (first,second)} for cid in a}
    for app in (first,second):app.initial_bindings=deepcopy(bindings)
    manager=ControllerManager([],list(map(E2LikeInterface,(first,second))),20,100000,
        direct_conflict_mitigation=mode,mitigation_seed=seed,mitigation_window_ms=100,
        mitigation_slot_duration_ms=.5)
    manager._initial_rapp_run_completed=True
    sim=Simulator.__new__(Simulator);sim.state=runtime;sim.controller_manager=manager
    sim.executor=LocalExecutor();sim._rng=random.Random(1);sim._control_reject_prob=0.
    sim._refresh_runtime_views_after_config_apply=lambda:None
    return sim,first,second


def actions(decision):
    return [a for a in decision.config_actions if any(r.get('field')=='scheduler_bundle' for r in getattr(a,'_mitigation_records',[]))]


def tick(sim,slot,apply=True):
    sim.state.slot_id=slot
    result=sim.controller_manager.update(slot,observed(sim.state))
    if apply:
        # Apply all scheduler writes, including any unintended persistent-cache write.
        for action in result.config_actions:
            if action.scheduler_mode is not None or action.ue_scheduler_weights is not None or action.latency_bias is not None:
                sim._apply_rt_config_action(action,slot)
    return result


def test_real_executor_applies_exact_winner_bundle_with_neutral_entries_and_ack_ownership():
    sim,a,b=setup()
    sim.state.ues[0].scheduler_weight=2.
    sim.state.scheduler.latency_bias=3.
    sim.controller_manager.current_action.latency_bias=4.
    result=tick(sim,0)
    assert len(actions(result))==1
    action=actions(result)[0];record=action._mitigation_records[0]
    expected=(a if record['winner']=='a' else b).values['cell_0']
    assert record['selected_value']==expected
    assert sim.state.scheduler.scheduler_mode==expected['scheduler_mode']
    assert {u.ue_id:u.scheduler_weight for u in sim.state.ues}==expected['ue_scheduler_weights']
    assert sim.state.scheduler.latency_bias==1.
    outcome=sim.state.control.actuation_outcomes[-1]
    assert outcome['status']=='applied' and outcome['mitigation_records'][0]['winner']==record['winner']
    assert outcome['after']['cell_0']['ue_scheduler_weights']==expected['ue_scheduler_weights']
    assert outcome['after']['cell_0']['scheduler_mode']==expected['scheduler_mode']
    assert sim.state.managed_cells['cell_1'].scheduler.scheduler_mode=='round_robin'
    assert sim.state.cell.tx_power_dbm==30.
    rng=sim.controller_manager.scheduler_conflict_arbiter.rngs['cell_0'].getstate()
    for slot in (1,20,40,100,199):tick(sim,slot)
    assert sim.controller_manager.scheduler_conflict_arbiter.rngs['cell_0'].getstate()==rng
    assert a.seen[-1]['cell_0']['scheduler_mode']==record['winner']
    assert a.seen[-1]['cell_0']['ue_scheduler_weights']==record['winner']


def test_default_priority_is_unchanged_and_does_not_claim_scheduler_mitigation():
    sim,_,_=setup(mode='priority');result=tick(sim,0)
    assert not actions(result) and sim.controller_manager.scheduler_conflict_arbiter is None
    assert {u.ue_id:u.scheduler_weight for u in sim.state.ues}=={'ue_0':1.5,'ue_1':1.}


def test_queued_selection_does_not_confirm_and_old_window_rejected_without_mutation():
    sim,a,_=setup();queued=actions(tick(sim,0,False))[0]
    tick(sim,20,False)
    assert a.seen[-1]['cell_0']['scheduler_mode'] is None
    assert a.seen[-1]['cell_0']['ue_scheduler_weights'] is None
    sim._apply_rt_config_action(queued,20)
    tick(sim,40,False)
    assert a.seen[-1]['cell_0']['scheduler_mode']==queued._mitigation_records[0]['winner']
    tick(sim,200)
    current=[u.scheduler_weight for u in sim.state.ues]
    sim._apply_rt_config_action(queued,0)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='stale_mitigation_selection'
    assert [u.scheduler_weight for u in sim.state.ues]==current


def test_expiry_restores_canonical_custom_weights_and_native_mode_until_actual_ack():
    sim,_,_=setup(until=30);sim.state.ues[1].scheduler_weight=.75
    tick(sim,0)
    pending=actions(tick(sim,40,False))[0]
    assert pending.scheduler_mode=='round_robin' and pending.ue_scheduler_weights=={'ue_0':1.,'ue_1':.75}
    assert pending._mitigation_records[0]['winner']=='rule_withdrawal'
    tick(sim,60)
    assert sim.state.scheduler.scheduler_mode=='round_robin' and sim.state.scheduler.latency_bias is None
    assert [u.scheduler_weight for u in sim.state.ues]==[1.,.75]
    assert not actions(tick(sim,80)) and not sim.controller_manager.rule_leases.scheduler_managed_cells
    # Releasing the managed scope revokes an already-queued baseline action too.
    sim._apply_rt_config_action(pending,40)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='stale_mitigation_selection'


@pytest.mark.parametrize('change', ['withdraw','expiry','membership','mutation'])
def test_queued_full_bundle_invalidations_are_atomic_before_next_controller_tick(change):
    sim,_,_=setup(until=30);queued=actions(tick(sim,0,False))[0]
    sim.state.slot_id=5
    if change=='withdraw':sim.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=5)
    elif change=='expiry':sim.state.slot_id=31
    elif change=='membership':sim.state.connections['ue_1'].serving_cell_id='cell_1'
    else:queued.ue_scheduler_weights['ue_0']=.5
    sim._apply_rt_config_action(queued,5)
    assert sim.state.control.actuation_outcomes[-1]['status']=='rejected'
    assert [u.scheduler_weight for u in sim.state.ues]==[1.,1.]
    assert sim.state.scheduler.scheduler_mode=='round_robin'


def test_handover_invalidates_complete_old_maps_and_restores_uid_at_allowed_destination():
    sim,_,_=setup();tick(sim,0)
    sim.state.connections['ue_1'].serving_cell_id='cell_1'
    result=tick(sim,20)
    assert all(r['winner']=='rule_withdrawal' for a in actions(result) for r in a._mitigation_records)
    assert sim.state.scheduler.scheduler_mode=='round_robin'
    assert [u.scheduler_weight for u in sim.state.ues]==[1.,1.]
    assert not sim.controller_manager.rule_leases.scheduler_requests
    assert any(r.get('invalidation_reason')=='scheduler_membership_changed' for r in sim.controller_manager.rule_decisions)
    tick(sim,40)
    assert not sim.controller_manager.rule_leases.scheduler_managed_cells


def test_empty_cell_keeps_complete_empty_map_and_has_mode_only_execution_ownership():
    sim,a,_=setup(empty=True);result=tick(sim,0)
    empty=next(x for x in actions(result) if x.target_cell_id=='cell_1')
    assert empty.ue_scheduler_weights=={}
    assert sim.state.control.actuation_outcomes[-1]['status']=='applied'
    tick(sim,20)
    owners=a.seen[-1]['cell_1']
    assert owners['scheduler_mode']==empty._mitigation_records[0]['winner']
    assert owners['ue_scheduler_weights'] is None


@pytest.mark.parametrize('value', [dict(scheduler_mode='latency_aware'),
    bundle(ue_0=2.),bundle(ue_0=1.,ue_1=True),bundle(ue_0=1.,ue_1=float('nan')),
    dict(**bundle(ue_0=1.,ue_1=1.),latency_bias=2.)])
def test_partial_or_unsupported_bundle_cannot_silently_enter_random_mode(value):
    sim,a,_=setup();a.values['cell_0']=value
    with pytest.raises(ValueError):tick(sim,0)
    assert [u.scheduler_weight for u in sim.state.ues]==[1.,1.]


def test_same_numeric_policy_cannot_confirm_owner_from_a_report_before_actual_ack():
    sim,a,_=setup();queued=actions(tick(sim,0,False))[0]
    sim.state.slot_id=10;sim._apply_rt_config_action(queued,10)
    # Numeric policy is already identical, but a stale report predates ownership.
    sim.state.slot_id=20;report=observed(sim.state)
    for cell in report.cell_observations:cell.kpi_metadata['executed_config']['produced_at_slot']=9
    sim.controller_manager.update(20,report)
    assert a.seen[-1]['cell_0']['scheduler_mode'] is None
    tick(sim,40)
    assert a.seen[-1]['cell_0']['scheduler_mode']==queued._mitigation_records[0]['winner']
    # Reissuing an unchanged held selection does not move its first ACK boundary.
    assert sim.controller_manager._scheduler_mitigation_executed['cell_0']['applied_slot']==10
    sim.state.slot_id=60;report=observed(sim.state)
    for cell in report.cell_observations:cell.kpi_metadata['executed_config']['produced_at_slot']=15
    sim.controller_manager.update(60,report)
    assert a.seen[-1]['cell_0']['scheduler_mode']==queued._mitigation_records[0]['winner']


def test_selection_generation_replacement_revokes_old_queued_request():
    sim,a,b=setup();old=actions(tick(sim,0,False))[0]
    sim.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=20)
    tick(sim,20)
    bindings={'cell_0':{app.category_id:app.name for app in (a,b)}}
    sim.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings=bindings),slot_id=40)
    a.sent=b.sent=False;tick(sim,40)
    before=[u.scheduler_weight for u in sim.state.ues]
    sim._apply_rt_config_action(old,0)
    assert sim.state.control.actuation_outcomes[-1]['status']=='rejected'
    assert [u.scheduler_weight for u in sim.state.ues]==before


def test_standalone_tuning_cannot_mix_into_another_apps_scheduler_bundle():
    sim,_,b=setup();b.controlled_fields=['latency_bias']
    b.run=lambda obs,policy:ControlDecision(config_actions=[
        ConfigAction(source_name=b.name,target_cell_id='cell_0',latency_bias=2.)])
    with pytest.raises(ValueError,match='cannot mix standalone scheduler tuning'):tick(sim,0)


def test_actual_round_robin_winner_clears_bias_and_does_not_keep_losing_weights():
    sim,a,b=setup();b.values['cell_0']=bundle('round_robin',ue_0=1.,ue_1=2.)
    # Pick a seed whose first independent cell draw selects low-priority b.
    seed=next(s for s in range(30) if random.Random(f'{s}:direct-conflict-mitigation:scheduler_bundle:cell_0').random()>=.5)
    sim,a,b=setup(seed=seed);b.values['cell_0']=bundle('round_robin',ue_0=1.,ue_1=2.)
    sim.state.scheduler.latency_bias=3.;sim.controller_manager.current_action.scheduler_mode='latency_aware'
    result=tick(sim,0);selected=actions(result)[0]
    assert selected._mitigation_records[0]['winner']=='b'
    assert sim.state.scheduler.scheduler_mode=='round_robin' and sim.state.scheduler.latency_bias is None
    assert [u.scheduler_weight for u in sim.state.ues]==[1.,2.]


def test_stale_native_report_cannot_suppress_actual_restoration():
    sim,_,_=setup();tick(sim,0)
    sim.controller_manager.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=20)
    sim.state.slot_id=20;report=observed(sim.state)
    cfg=report.paper_report['values']['cell']['cell_0']['executed_config']
    cfg['scheduler_mode']='round_robin';cfg['ue_scheduler_weights']={'ue_0':1.,'ue_1':1.}
    result=sim.controller_manager.update(20,report)
    restored=actions(result)[0]
    assert restored._mitigation_records[0]['winner']=='rule_withdrawal'
    sim._apply_rt_config_action(restored,20)
    assert sim.state.scheduler.scheduler_mode=='round_robin'
    assert [u.scheduler_weight for u in sim.state.ues]==[1.,1.]


def test_conflict_active_set_withdrawal_is_rechecked_before_selection_sync():
    sim,_,_=setup();queued=actions(tick(sim,0,False))[0]
    original=sim.controller_manager.instance_selection
    # The selected-generation view has not consumed the newer policy yet.
    sim.controller_manager.instance_selection=NS(mode='conflict_experiment',
        apps=original.apps,enabled=original.enabled,valid=lambda p:True)
    sim.controller_manager.policy_store.snapshot.xapp_active_sets={}
    sim.state.slot_id=5;sim._apply_rt_config_action(queued,5)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='mitigation_source_withdrawn'
    assert [u.scheduler_weight for u in sim.state.ues]==[1.,1.]


def test_cluster_cache_cannot_bypass_bundle_in_a_nonprimary_managed_cell():
    sim,a,b=setup()
    for connection in sim.state.connections.values():connection.serving_cell_id='cell_1'
    for app in (a,b):
        app.values={'cell_1':app.values['cell_0']}
        app.initial_bindings={'cell_1':{x.category_id:x.name for x in (a,b)}}
    sim.controller_manager.current_action_by_scope['cluster:*']=ConfigAction(
        apply_scope='cluster',scheduler_mode='round_robin',latency_bias=4.,tx_power_dbm=25.)
    result=tick(sim,0,False)
    assert len(actions(result))==1 and actions(result)[0].target_cell_id=='cell_1'
    cluster=[x for x in result.config_actions if x.apply_scope=='cluster']
    assert cluster and any(x.tx_power_dbm==25. for x in cluster)
    assert all(x.scheduler_mode is None and x.latency_bias is None for x in cluster)
    for action in result.config_actions:
        if action.scheduler_mode is not None or action.tx_power_dbm is not None:
            sim._apply_rt_config_action(action,0)
    assert sim.state.managed_cells['cell_1'].scheduler.scheduler_mode=='latency_aware'
    assert sim.state.managed_cells['cell_0'].scheduler.scheduler_mode=='round_robin'


def test_fresh_cluster_scheduler_contest_is_explicitly_unsupported():
    sim,_,b=setup();b.controlled_fields=['scheduler_mode']
    b.run=lambda obs,policy:ControlDecision(config_actions=[
        ConfigAction(source_name=b.name,apply_scope='cluster',scheduler_mode='round_robin')])
    with pytest.raises(ValueError,match='cannot mix standalone scheduler tuning'):tick(sim,0)
