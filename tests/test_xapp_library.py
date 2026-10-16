from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from models.policy import PolicySnapshot,PolicyUpdate
from ric.near_rt.xapp_library import XAppInstance,IMPLEMENTATIONS,category_catalog
from ric.near_rt.xapp_library.selection import InstanceSelection
from ric.interfaces.e2_like import E2LikeInterface
from ric.xapp_contract import validate_action,validate_commands
from test_rule_based_v1 import view


def observed(slot=1000):
    o=view(slot);o.cell_id='a'
    for u in o.ue_observations:
        u.executed_prb_slots_total=slot;u.resource_counter_epoch='cell:1';u.delivered_bits_total=slot*100
        u.executed_prb_slots_window=20;u.resource_window_start_slot=slot-9;u.resource_window_end_slot=slot
    return o


def instance(ident,name='instance',cells=('a','b'),bindings=None,parameters=None):
    return XAppInstance(name,ident,cells,initial_bindings=bindings,parameters=parameters)


def test_catalog_has_twelve_categories_twenty_eight_distinct_implementations():
    assert len(category_catalog())==12 and len(IMPLEMENTATIONS)==28
    assert sum(s['algorithm']=='legacy_v1' for s in IMPLEMENTATIONS.values())==12
    assert all(s['training_required'] is False for s in IMPLEMENTATIONS.values())


@pytest.mark.parametrize('category',list(category_catalog()))
def test_legacy_decisions_preserved_by_instance_wrapper(category):
    from ric.near_rt.rule_based_v1 import RuleXApp
    old=RuleXApp(category,name='instance');new=instance(category+':legacy_v1')
    new.select({'a','b'},{'a':1,'b':1})
    for slot in (1000,1040,1500):
        left=old.run(observed(slot),PolicySnapshot());right=new.run(observed(slot),PolicySnapshot())
        def requests(d):
            return [x for x in d.config_actions if x.target_cell_id is not None and
                any(getattr(x,k,None) not in (None,{}) for k in old.controlled_fields)]
        assert requests(left)==requests(right)
        assert left.commands==right.commands


@pytest.mark.parametrize('ident',list(IMPLEMENTATIONS))
def test_all_implementations_deterministic_scoped_and_declared(ident):
    a=instance(ident);b=instance(ident)
    a.select({'a'},{'a':1});b.select({'a'},{'a':1})
    for slot in (1000,1500,2000):
        o=observed(slot);x=a.run(deepcopy(o),PolicySnapshot());y=b.run(deepcopy(o),PolicySnapshot())
        assert x==y
        for action in x.config_actions:
            assert action.target_cell_id in ('a',None)
            assert not validate_action(action,a.contract,o)
        assert not validate_commands(x.commands,a.contract,o)


def test_instance_and_cell_history_isolation():
    a=instance('rf_energy:sinr_pi');b=instance('rf_energy:sinr_pi',name='other')
    a.select({'a','b'},{'a':1,'b':1});b.select({'a'},{'a':1})
    a.run(observed(),PolicySnapshot());b.run(observed(),PolicySnapshot())
    original=a.children['b'];a.children['a'].state['poison']=object()
    assert 'poison' not in b.children['a'].state and 'poison' not in original.state
    a.select({'b'},{'b':1});assert a.children=={'b':original}
    a.select({'a','b'},{'a':3,'b':1});a.run(observed(1500),PolicySnapshot())
    assert 'poison' not in a.children['a'].state and a.children['b'] is original


def manager(apps):
    from ric.controller_manager import ControllerManager
    return ControllerManager([],list(map(E2LikeInterface,apps)),5,10)


def test_complete_bindings_atomic_policy_and_empty_disable():
    a=instance('rf_energy:legacy_v1',name='a0');b=instance('rf_energy:sinr_pi',name='b0')
    m=manager([a,b]);good={'a':{'rf_energy':'a0'},'b':{'rf_energy':'b0'}}
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings=good),slot_id=1)
    m.instance_selection.sync(5,observed());assert a.active_cells=={'a'} and b.active_cells=={'b'}
    saved=m.policy_store.get_snapshot()
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={'a':{'rf_energy':'missing'}},rule_based_parameters={'a0':{'pi_kp':100}}),slot_id=2)
    assert m.policy_store.get_snapshot()==saved
    assert m.policy_store.binding_receipts[-1]['status']=='rejected'
    m.policy_store.start_new_epoch();assert m.policy_store.get_snapshot().xapp_bindings==good
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={}),slot_id=10);m.instance_selection.sync(10,observed())
    assert not a.active_cells and not b.active_cells


def test_overlap_needs_explicit_selection_and_wrong_scope_rejected():
    m=manager([instance('rf_energy:legacy_v1',name='a0'),instance('rf_energy:sinr_pi',name='b0')])
    with pytest.raises(ValueError,match='Overlapping'):m.instance_selection.default_bindings()
    for bad in ({'a':{'slice_sla':'a0'}},{'unknown':{'rf_energy':'a0'}},{'a':{'rf_energy':'unknown'}}):
        with pytest.raises(ValueError):m.instance_selection.validate(bad)


def test_stale_generation_invalidates_reenable_but_not_unaffected_cell():
    a=instance('rf_energy:legacy_v1',name='a0');b=instance('rf_energy:sinr_pi',name='b0')
    m=manager([a,b]);s=m.instance_selection
    one={'a':{'rf_energy':'a0'},'b':{'rf_energy':'a0'}}
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings=one));s.sync(5,observed())
    pa=a.identity('a');pb=a.identity('b')
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={'a':{'rf_energy':'b0'},'b':{'rf_energy':'a0'}}));s.sync(10,observed())
    assert not s.valid(pa) and s.valid(pb)
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings=one));s.sync(15,observed())
    assert not s.valid(pa) and s.valid(a.identity('a')) and s.valid(pb)


def test_parameter_update_transaction_and_immutable_defaults():
    app=instance('rf_energy:sinr_pi',parameters={'pi_kp':.5})
    m=manager([app]);m.policy_store.apply_update(PolicyUpdate(rule_based_parameters={app.name:{'pi_kp':.8}}))
    assert app.parameters['pi_kp']==.5
    saved=m.policy_store.get_snapshot()
    m.policy_store.apply_update(PolicyUpdate(rule_based_parameters={app.name:{'pi_kp':-1}}))
    assert m.policy_store.get_snapshot()==saved
    assert IMPLEMENTATIONS[app.implementation_id]['defaults']['pi_kp']==.2


def test_legacy_activation_lists_cannot_silently_override_library_bindings():
    app=instance('rf_energy:sinr_pi');m=manager([app])
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={'a':{'rf_energy':app.name}}))
    before=m.policy_store.get_snapshot()
    assert m.policy_store.apply_update(PolicyUpdate(disabled_xapps=[app.name])) is False
    assert m.policy_store.get_snapshot()==before
    assert 'require xapp_bindings' in m.policy_store.binding_receipts[-1]['reason']


def test_runtime_observation_age_and_minimum_action_interval_are_consumed():
    app=instance('rf_energy:sinr_pi',parameters={'max_age_ms':7})
    assert app.contract.max_age_slots==14
    policy=PolicySnapshot(rule_based_parameters={app.name:{'max_age_ms':1.1,'min_action_ms':100}})
    assert app.runtime_contract(policy).max_age_slots==3
    app.select({'a'},{'a':1})
    assert app.run(observed(1000),policy).config_action.tx_power_dbm is not None
    assert app.run(observed(1040),policy).config_action.tx_power_dbm is None
    assert app.run(observed(1200),policy).config_action.tx_power_dbm is not None


def test_invalid_direct_policy_parameters_hold_instead_of_mutating_state():
    app=instance('ret_coverage:coverage_grid');app.select({'a'},{'a':1})
    app.run(observed(),PolicySnapshot());before=deepcopy(app.children['a'].state)
    result=app.run(observed(1500),PolicySnapshot(rule_based_parameters={app.name:{'ret_window_ms':1}}))
    assert result.config_action.ret_tilt_deg is None
    assert app.children['a'].state==before and app.last_decisions[0]['reason']=='nonrt_policy_rejected'


def test_fairness_uses_executed_counters_not_requested_limits():
    app=instance('ue_fairness:deficit_airtime');app.select({'a'},{'a':1})
    o=observed();second=deepcopy(o.ue_observations[0]);second.ue_id='extra';o.ue_observations.append(second)
    app.run(o,PolicySnapshot())
    o.slot_id=1100
    for cell in o.cell_observations:cell.kpi_metadata['executed_config']['produced_at_slot']=1100
    o.ue_observations[0].executed_prb_slots_total+=800
    d=app.run(o,PolicySnapshot())
    assert d.config_action.ue_prb_limits['extra']>d.config_action.ue_prb_limits['u0']


def test_pi_direction_saturation_and_stale_report():
    app=instance('rf_energy:sinr_pi');app.select({'a'},{'a':1})
    o=observed();x=app.run(o,PolicySnapshot());assert x.config_action.tx_power_dbm>29
    assert app.run(o,PolicySnapshot()).config_action.tx_power_dbm is None
    o=observed(1500);o.cell_observations[0].executed_config['tx_power_dbm']=30
    assert app.run(o,PolicySnapshot()).config_action.tx_power_dbm is None


def test_a3_continuity_and_ttt():
    app=instance('traffic_steering:a3_rsrp');app.select({'a'},{'a':1})
    assert not app.run(observed(1000),PolicySnapshot()).commands
    for slot in range(1005,1080,5):assert not app.run(observed(slot),PolicySnapshot()).commands
    assert app.run(observed(1080),PolicySnapshot()).commands
    # A gap beyond the reporting continuity guard restarts the trigger.
    fresh=instance('traffic_steering:a3_rsrp');fresh.select({'a'},{'a':1})
    fresh.run(observed(1000),PolicySnapshot())
    assert not fresh.run(observed(1200),PolicySnapshot()).commands


def test_a3_configuration_change_restarts_continuity_without_clearing_cooldown():
    app=instance('traffic_steering:a3_rsrp');app.select({'a'},{'a':1})
    for slot in range(1000,1200,5):
        o=observed(slot)
        if slot>=1040:o.cell_observations[0].executed_config['time_to_trigger_ms']=80.
        assert not app.run(o,PolicySnapshot()).commands
    o=observed(1200);o.cell_observations[0].executed_config['time_to_trigger_ms']=80.
    assert app.run(o,PolicySnapshot()).commands
    o=observed(1205);o.cell_observations[0].executed_config['time_to_trigger_ms']=0.
    assert not app.run(o,PolicySnapshot()).commands


def test_a3_invalid_report_breaks_the_timer():
    app=instance('traffic_steering:a3_rsrp');app.select({'a'},{'a':1})
    emitted=[]
    for slot in range(1000,1130,5):
        o=observed(slot)
        if slot==1040:o.ue_observations[0].kpi_metadata['candidate_rsrp_dbm']['valid']=False
        if app.run(o,PolicySnapshot()).commands:emitted.append(slot)
    assert emitted==[1125]


def test_read_context_does_not_grant_write_access_and_handover_target_is_separate():
    from ric.xapp_contract import project
    app=XAppInstance('source_only','traffic_steering:a3_rsrp',('a',),context_cell_ids=('a','b'))
    o=observed();projected,errors=project(o,app.contract,now_slot=o.slot_id)
    assert not errors and {c.cell_id for c in projected.cell_observations}=={'a','b'}
    from models.action import ConfigAction,CommandAction
    assert 'invalid_target_cell' in validate_action(ConfigAction(target_cell_id='b',tx_power_dbm=20),app.contract,o)
    assert not validate_commands([CommandAction('request_handover',target_ue_id='u0',target_cell_id='b')],app.contract,o)
    assert 'command_source_scope' in validate_commands([CommandAction('request_handover',target_ue_id='u1',target_cell_id='a')],app.contract,o)


def test_required_counter_missing_does_not_become_zero_or_crash():
    app=instance('ue_fairness:deficit_airtime');app.select({'a'},{'a':1})
    o=observed();o.ue_observations[0].kpi_metadata['executed_prb_slots_total']['valid']=False
    assert app.run(o,PolicySnapshot()).config_action.ue_prb_limits is None
    assert any(r['reason']=='invalid_entity_kpi' for r in app.last_decisions)


def test_ret_grid_waits_for_executed_window_and_detects_configuration_changes():
    app=instance('ret_coverage:coverage_grid');app.select({'a'},{'a':1})
    assert app.run(observed(1000),PolicySnapshot()).config_action.ret_tilt_deg==0
    for slot in range(1005,1800,5):
        o=observed(slot);o.cell_observations[0].executed_config['ret_enabled']=True
        o.executed_configs['a']['ret_enabled']=True
        app.run(o,PolicySnapshot())
    assert app.children['a'].state['a']['index']==0
    o=observed(1805);o.cell_observations[0].executed_config['ret_enabled']=True;o.executed_configs['a']['ret_enabled']=True
    assert app.run(o,PolicySnapshot()).config_action.ret_tilt_deg==3
    o=observed(2210);o.cell_observations[0].executed_config['tx_power_dbm']=28;o.executed_configs['a']['tx_power_dbm']=28
    assert app.run(o,PolicySnapshot()).config_action.ret_tilt_deg==0


def test_live_closed_loop_selection_final_values_and_old_actions(tmp_path):
    from test_paper_interfaces import make_sim
    sim=make_sim(tmp_path,cells=2,slots=25,config={'rule_observations':True})
    a=instance('slice_sla:legacy_v1',name='slicing',cells=('cell_0','cell_1'))
    b=instance('slice_sla:goodput_pi',name='pi',cells=('cell_0','cell_1'))
    m=sim.controller_manager;m.e2_interfaces=list(map(E2LikeInterface,[a,b]));m._initial_rapp_run_completed=True
    m.policy_store.binding_validator=m.instance_selection.validate
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={'cell_0':{'slice_sla':'slicing'},'cell_1':{'slice_sla':'slicing'}}))
    for _ in range(12):sim.step()
    child=a.children['cell_1'];old=deepcopy(sim.state.control.active_config_actions[-1])
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={'cell_0':{'slice_sla':'pi'},'cell_1':{'slice_sla':'slicing'}}))
    for _ in range(5):sim.step()
    assert a.children['cell_1'] is child and 'cell_0' not in a.children
    assert b.active_cells=={'cell_0'}
    assert sim.state.managed_cells['cell_0'].scheduler.slice_weight_adjustment
    # Explicit generation check uses an action from the switched cell.
    old._library_provenances=[dict(instance_id='slicing',category_id='slice_sla',cell_id='cell_0',selection_version=1)]
    sim._apply_rt_config_action(old,sim.state.slot_id)
    assert sim.state.control.actuation_outcomes[-1]['reason']=='stale_xapp_selection'
    for u in sim.state.ues:
        s=u.packet_ledger.stats
        assert s['initial_bytes']+s['offered_bytes']==s['pending_bytes']+s['overflow_bytes']+s['delivered_bytes']


def test_power_methods_have_distinct_decisions_for_the_same_measured_fixture():
    o=observed();u=o.ue_observations[0];v=deepcopy(u);v.ue_id='extra'
    u.subband_cqi=[15,15,15,15,1,1,1,1];u.queue_bytes=200;u.throughput_mbps=.1
    v.subband_cqi=[1,1,1,1,15,15,15,15];v.queue_bytes=2000;v.throughput_mbps=2.
    o.ue_observations.append(v);vectors=[]
    for ident in ('legacy_v1','cqi_proportional','backlog_cqi'):
        app=instance('throughput_power:'+ident);app.select({'a'},{'a':1})
        vectors.append(app.run(deepcopy(o),PolicySnapshot()).config_action.prb_power_w)
    assert len({tuple(x) for x in vectors})==3


def test_max_min_beam_is_not_a_parameter_variant_of_weighted_mean():
    o=observed();u=o.ue_observations[0];u.priority=10;v=deepcopy(u);v.priority=1;v.ue_id='extra'
    for row in u.beam_reference_reports:row['reference_sinr_db']=20 if row['codebook_id']==0 else 10
    for row in v.beam_reference_reports:row['reference_sinr_db']=-5 if row['codebook_id']==0 else 0
    o.ue_observations.append(v)
    old=instance('beam_management:legacy_v1');new=instance('beam_management:max_min_reference')
    for a in (old,new):a.select({'a'},{'a':1})
    assert old.run(deepcopy(o),PolicySnapshot()).config_action.codebook_id is None
    assert new.run(deepcopy(o),PolicySnapshot()).config_action.codebook_id==1


def test_public_resource_counter_matches_actual_transmissions_and_has_delayed_metadata(tmp_path):
    from test_paper_interfaces import make_sim
    from ric.xapp_contract import project
    sim=make_sim(tmp_path,cells=1,slots=10,config={'rule_observations':True})
    totals={u.ue_id:0 for u in sim.state.ues}
    for _ in range(8):
        sim.step()
        for row in sim.phy_adapter.trace['ues']:
            if row['actual_decode']:totals[row['ue_id']]+=len(row['resource_allocation']['prbs'])
    for uid,value in totals.items():assert sim.state.paper.published['ue'][uid]['executed_prb_slots_total']==value
    app=XAppInstance('counter','ue_fairness:deficit_airtime',('cell_0',))
    o=sim.state.rt_du_observation
    v,errors=project(o,app.contract,now_slot=sim.state.slot_id)
    assert not errors
    for u in v.ue_observations:
        m=u.kpi_metadata['executed_prb_slots_total']
        assert m['produced_at_slot']<=m['delivered_at_slot']<=sim.state.slot_id
        assert u.executed_prb_slots_total<=totals[u.ue_id]


def test_switch_missing_observation_holds_then_explicit_disable_restores(tmp_path):
    from test_paper_interfaces import make_sim
    sim=make_sim(tmp_path,cells=1,slots=30,config={'rule_observations':True})
    a=XAppInstance('old','slice_sla:legacy_v1',('cell_0',))
    b=XAppInstance('new','slice_sla:service_deficit',('cell_0',))
    m=sim.controller_manager;m.e2_interfaces=list(map(E2LikeInterface,[a,b]));m._initial_rapp_run_completed=True
    m.policy_store.binding_validator=m.instance_selection.validate
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={'cell_0':{'slice_sla':'old'}}))
    for _ in range(12):sim.step()
    consumer=sim.state.managed_cells['cell_0'].scheduler
    weights=deepcopy(consumer.slice_weight_adjustment);assert weights
    bad=deepcopy(sim.state.rt_du_observation)
    for report in bad.paper_report['values']['ue'].values():report['requested_dl_rate_bps']=None
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={'cell_0':{'slice_sla':'new'}}))
    d=m.update(15,bad)
    for action in d.config_actions:sim._apply_rt_config_action(action,15)
    assert consumer.scheduler_mode=='slice_aware' and consumer.slice_weight_adjustment==weights
    assert not b.statistics['action_requested']
    m.policy_store.apply_update(PolicyUpdate(xapp_bindings={}))
    d=m.update(20,bad)
    for action in d.config_actions:sim._apply_rt_config_action(action,20)
    assert consumer.scheduler_mode=='round_robin' and consumer.slice_weight_adjustment=={}
