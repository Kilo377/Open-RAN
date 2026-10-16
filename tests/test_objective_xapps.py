from copy import deepcopy
from types import SimpleNamespace as NS
import pytest
from test_xapp_library import observed,manager
from ric.near_rt.xapp_library import XAppInstance,IMPLEMENTATIONS
from ric.near_rt.xapp_library.objectives import OBJECTIVES,OBJECTIVE_ALGORITHMS,SEARCH_ALGORITHMS
from models.policy import PolicySnapshot,PolicyUpdate
from models.action import ConfigAction
from ric.xapp_contract import KPI_REGISTRY


def app(k,name='goal',mode='exclusive',priority=None):
    ident=next(i for i,s in IMPLEMENTATIONS.items() if s['algorithm']==k)
    return XAppInstance(name,ident,('a','b'),selection_mode=mode,priority=priority)


def report(slot=1000):
    o=observed(slot)
    # Multiple real measured service cohorts, no controller-private state.
    second=deepcopy(o.ue_observations[0]);second.ue_id='u2';second.slice_id='embb';second.throughput_mbps=.8
    second.subband_cqi=[14]*4+[1]*4;second.queue_bytes=10000
    o.ue_observations[0].subband_cqi=[1]*4+[10]*4;o.ue_observations[0].queue_bytes=10000
    o.ue_observations.append(second)
    for u in o.ue_observations:
        u.delivered_bits_total=slot*(100 if u.ue_id=='u0' else 400)
    for c in o.cell_observations:
        c.rf_energy_total_j=slot*.001;c.rf_efficiency_bit_per_j=1e6;c.config_revision=0
    return o


def execute(k,o=None):
    x=app(k);x.select({'a'},{'a':1});return x,x.run(o or report(),PolicySnapshot())


def test_objective_catalog_real_metrics_and_no_overridable_goal():
    assert len(OBJECTIVES)==18 and len(IMPLEMENTATIONS)==46
    for k in OBJECTIVE_ALGORITHMS:
        x=app(k);s=x.spec
        assert s['objective_id'] and s['evaluation_kpis'] and s['training_required'] is False
        assert all(n in KPI_REGISTRY for n in s['evaluation_kpis']+s['required_kpis'])
        with pytest.raises(ValueError):x.validate_parameters({'objective_id':'pretend'})
        if k in SEARCH_ALGORITHMS:
            with pytest.raises(ValueError):x.validate_parameters({'objective_window_ms':399})


@pytest.mark.parametrize('k',sorted(OBJECTIVE_ALGORITHMS))
def test_objectives_deterministic_and_same_report_not_reused(k):
    a,x=execute(k);b,y=execute(k)
    assert x==y
    assert a.identity('a')['objective_id']==a.spec['objective_id']
    z=a.run(report(),PolicySnapshot())
    assert not any(any(getattr(c,f,None) not in (None,{}) for f in a.controlled_fields) for c in z.config_actions) and not z.commands


def test_power_and_resource_goals_produce_distinct_decisions_on_identical_input():
    def request(k):return execute(k)[1].config_action
    assert request('sum_service_power').prb_power_w!=request('weak_user_power').prb_power_w
    caps=[request(k).ue_prb_limits for k in ('sum_service_caps','proportional_utility_caps','starvation_guard_caps')]
    assert len({tuple(sorted(x.items())) for x in caps})==3
    a=request('sla_count_weights');b=request('worst_slice_weights')
    assert a.slice_weight_adjustment!=b.slice_weight_adjustment
    for k in ('sum_service_power','weak_user_power'):
        p=request(k).prb_power_w
        assert min(p)>=0 and sum(p)==pytest.approx(10**(-.1))
    for c in caps:assert sum(c.values())<=8


def test_sla_admission_differs_from_weakest_sinr_power():
    o=report();u=o.ue_observations[0];u.reference_sinr_db=-20;u.hol_delay_ms=0;u.throughput_mbps=.01
    o.ue_observations[-1].throughput_mbps=.01;o.ue_observations[-1].hol_delay_ms=0
    x,d=execute('sla_admission_power',o)
    assert d.config_action.tx_power_dbm is None
    assert any(r['reason']=='no_near_admission_or_deadline_trigger' for r in x.last_decisions)


def test_deadline_goal_differs_from_both_rate_goals():
    o=report();u=deepcopy(o.ue_observations[0]);u.ue_id='third';u.slice_id='voice';u.requested_dl_rate_bps=1e6
    u.throughput_mbps=1.;u.deadline_misses_window=9;o.ue_observations.append(u)
    actions=[execute(k,deepcopy(o))[1].config_action.slice_weight_adjustment for k in ('sla_count_weights','worst_slice_weights','deadline_loss_weights')]
    assert len({tuple(sorted(x.items())) for x in actions})==3


def test_coverage_and_continuity_differ_under_short_fade():
    o=report()
    for u in o.ue_observations:
        if u.serving_cell_id=='a':u.rsrp_dbm=-109;u.reference_sinr_db=-5
    assert execute('coverage_margin_steering',o)[1].commands
    assert not execute('continuity_first_steering',o)[1].commands


def test_concurrent_transaction_lifecycle_and_retained_history():
    a=app('sum_service_power','a',mode='conflict_experiment');b=app('weak_user_power','b',mode='conflict_experiment')
    m=manager([a,b]);s=m.instance_selection
    good={'a':{'throughput_power':['b','a']}}
    m.policy_store.apply_update(PolicyUpdate(xapp_active_sets=good));s.sync(5,report())
    a.run(report(),PolicySnapshot());b.run(report(),PolicySnapshot());child=b.children['a'];old=a.identity('a');keep=b.identity('a')
    for bad in (["a","a"],["a","missing"]):
        saved=m.policy_store.get_snapshot()
        assert m.policy_store.apply_update(PolicyUpdate(xapp_active_sets={'a':{'throughput_power':bad}})) is False
        assert m.policy_store.get_snapshot()==saved
    assert m.policy_store.apply_update(PolicyUpdate(xapp_bindings={})) is False
    m.policy_store.apply_update(PolicyUpdate(xapp_active_sets={'a':{'throughput_power':['b']}}));s.sync(10,report())
    assert not s.valid(old) and s.valid(keep) and b.children['a'] is child
    m.policy_store.apply_update(PolicyUpdate(xapp_active_sets=good));s.sync(15,report())
    assert not s.valid(old) and b.children['a'] is child
    assert 'a' not in a.children
    m.policy_store.start_new_epoch();assert m.policy_store.get_snapshot().xapp_active_sets==good
    m.policy_store.apply_update(PolicyUpdate(xapp_active_sets={}));s.sync(20,report());assert not a.active_cells and not b.active_cells


def test_concurrent_same_goal_duplicate_and_exclusive_sets_rejected():
    a=app('sum_service_power','a','conflict_experiment');b=app('sum_service_power','b','conflict_experiment');m=manager([a,b])
    assert m.policy_store.apply_update(PolicyUpdate(xapp_active_sets={'a':{'throughput_power':['a','b']}})) is False
    m=manager([app('sum_service_power')]);assert m.policy_store.apply_update(PolicyUpdate(xapp_active_sets={})) is False


def test_lease_winner_deterministic_and_withdrawal_restores_loser():
    from ric.near_rt.rule_based_v1.leases import RuleLeases
    a=app('sum_service_power','a','conflict_experiment');b=app('weak_user_power','b','conflict_experiment');m=manager([a,b]);s=m.instance_selection
    m.policy_store.apply_update(PolicyUpdate(xapp_active_sets={'a':{'throughput_power':['a','b']}}));s.sync(5,report())
    o=report();o.paper_report={'values':{'cell':{'a':{'executed_config':o.executed_configs['a']}}}}
    r=RuleLeases();m.rule_leases=r
    aa=ConfigAction(source_name='a',target_cell_id='a',prb_power_w=[.1]*8);bb=ConfigAction(source_name='b',target_cell_id='a',prb_power_w=[.05]*8)
    winners={};r.reconcile(m,o,winners,{},[bb,aa]);assert winners['cell:a']['prb_power_w'][2]=='b'
    m.policy_store.apply_update(PolicyUpdate(xapp_active_sets={'a':{'throughput_power':['a']}}));s.sync(10,o)
    winners={};r.reconcile(m,o,winners,{},[]);assert winners['cell:a']['prb_power_w'][2]=='a'


def test_search_does_not_score_unexecuted_overridden_or_short_window():
    x=app('sum_goodput_grid');x.select({'a'},{'a':1})
    x.run(report(),PolicySnapshot())
    o=report(1500);o.control_ownership={'a':{f:'other' for f in ('ret_enabled','ret_codebook_id','ret_tilt_deg')}}
    x.run(o,PolicySnapshot());s=x.children['a'].state['a']
    assert not s['scores'] and any(r['reason']=='search_waiting_for_ownership' for r in x.last_decisions)
    # Actual configuration and ownership are required before accumulating time.
    target=s['pending']['target']
    for slot in (1600,2000):
        o=report(slot);o.cell_observations[0].executed_config.update(target);o.executed_configs['a'].update(target)
        o.control_ownership={'a':{f:x.name for f in target}};x.run(o,PolicySnapshot())
    assert not s['scores']
    o=report(2500);o.cell_observations[0].executed_config.update(target);o.executed_configs['a'].update(target)
    o.control_ownership={'a':{f:x.name for f in target}};x.run(o,PolicySnapshot())
    assert len(s['scores'])==1


def test_scoped_example_builds_without_duplicate_cell_ids(tmp_path):
    from pathlib import Path
    from phy.rt_sys.scenario import build_case
    path=Path(__file__).resolve().parents[1]/'configs/rt_sys/xapp_library/throughput_power__sum_service_power.json'
    import json
    data=json.loads(path.read_text());data['xapp_library']['instances'][0]['cell_ids']=['cell_0']
    data['xapp_library'].pop('initial_selection');data['xapp_library']['bindings']={'cell_0':{'throughput_power':'throughput_power__sum_service_power'}}
    path=tmp_path/'scoped.json';path.write_text(json.dumps(data))
    _,c,_=build_case(num_cells=3,num_ues=4,total_slots=8,config_path=path)
    item=next(x for x in c['ric']['xapps'] if x['type']=='xapp_instance')
    assert item['cell_ids']==['cell_0'] and item['selection_mode']=='exclusive'


def test_mro_targets_distinguish_pingpong_and_failed_weak_link():
    o=report(1500)
    events=[dict(kind='handover_success',slot_id=1000+j*10,ue_id='u0',from_cell='a' if j%2==0 else 'b',to_cell='b' if j%2==0 else 'a') for j in range(10)]
    o.ue_observations[0].mobility_events=events
    assert execute('pingpong_guard',o)[1].config_action.time_to_trigger_ms==60.
    assert execute('failure_guard',o)[1].config_action.time_to_trigger_ms is None
    o=report(1500);o.ue_observations[0].reference_sinr_db=-10
    o.ue_observations[0].mobility_events=[dict(kind='handover_failure',slot_id=1000+j*10,ue_id='u0',from_cell='a',to_cell='b') for j in range(5)]
    failure=app('failure_guard');failure.select({'a'},{'a':1})
    for slot in range(1400,1501,5):
        sample=report(slot);sample.ue_observations[0].reference_sinr_db=-10
        if slot==1500:sample.ue_observations[0].mobility_events=o.ue_observations[0].mobility_events
        decision=failure.run(sample,PolicySnapshot())
    assert decision.config_action.time_to_trigger_ms==20.
    assert execute('pingpong_guard',o)[1].config_action.time_to_trigger_ms is None


def test_starvation_uses_new_executed_counter_not_requested_caps():
    x,_=execute('starvation_guard_caps')
    o=report(1100);o.ue_observations[0].executed_prb_slots_total+=800
    # Other active UE had no executed resources, despite any prior request.
    o.ue_observations[-1].executed_prb_slots_total=1000
    x.run(o,PolicySnapshot())
    state=x.children['a'].state['a']['last_resource']
    assert state['u0'][2]==1100 and state['u2'][2]==1000


def test_search_context_change_resets_evaluation_and_invalid_report_holds():
    x=app('sum_goodput_beam');x.select({'a'},{'a':1})
    x.run(report(),PolicySnapshot());child=x.children['a']
    o=report(1500);o.executed_configs['b']['tx_power_dbm']=25
    x.run(o,PolicySnapshot())
    assert not child.state['a']['scores']
    assert any(r['reason']=='search_context_changed' for r in x.last_decisions)
    o=report(2000)
    for u in o.ue_observations:u.kpi_metadata['delivered_bits_total']['valid']=False
    d=x.run(o,PolicySnapshot())
    assert not any(c.codebook_id is not None for c in d.config_actions)


def test_starvation_request_changes_after_real_resource_history():
    x,_=execute('starvation_guard_caps')
    o=report(1100);o.ue_observations[0].executed_prb_slots_total+=800;o.ue_observations[-1].executed_prb_slots_total=1000
    d=x.run(o,PolicySnapshot());assert d.config_action.ue_prb_limits['u2']>d.config_action.ue_prb_limits['u0']


def evaluate_one_window(k,mutate=None):
    x=app(k);x.select({'a'},{'a':1});x.run(report(1000),PolicySnapshot())
    child=x.children['a'];target=deepcopy(child.state['a']['pending']['target'])
    for slot in (1500,2300):
        o=report(slot)
        if mutate:mutate(o)
        o.cell_observations[0].executed_config.update(target);o.executed_configs['a'].update(target)
        o.control_ownership={'a':{f:x.name for f in target}};x.run(o,PolicySnapshot())
    return x,child.state['a']['scores'][0]['score']


def test_ret_and_beam_goals_evaluate_distinct_actual_metrics():
    coverage,cs=evaluate_one_window('coverage_fraction_grid')
    goodput,gs=evaluate_one_window('sum_goodput_grid')
    assert len(cs)==2 and len(gs)==1 and cs!=gs
    throughput,ts=evaluate_one_window('sum_goodput_beam')
    sla,ss=evaluate_one_window('sla_count_beam')
    assert len(ts)==1 and len(ss)==2 and ts!=ss
    assert any(row['reason']=='objective_window_evaluated' for row in sla.last_decisions)


def test_rf_efficiency_uses_energy_and_goodput_deltas():
    _,score=evaluate_one_window('rf_efficiency_search')
    # Both serving UE cumulative counters increase: (100+400)*800 bits / .8 J.
    assert score==pytest.approx((500000.,))


def test_unknown_concurrent_category_and_mixed_policy_rejected_atomically():
    x=app('sum_service_power',mode='conflict_experiment');m=manager([x]);before=m.policy_store.get_snapshot()
    for update in (PolicyUpdate(xapp_active_sets={'a':{'unknown':[]}}),PolicyUpdate(xapp_bindings={},xapp_active_sets={})):
        assert m.policy_store.apply_update(update) is False
        assert m.policy_store.get_snapshot()==before


def test_transient_config_change_back_to_same_values_invalidates_window():
    x=app('sum_goodput_beam');x.select({'a'},{'a':1});x.run(report(1000),PolicySnapshot())
    s=x.children['a'].state['a'];target=deepcopy(s['pending']['target'])
    for slot,revision in ((1500,0),(2300,2)):
        o=report(slot);o.cell_observations[0].executed_config.update(target);o.executed_configs['a'].update(target)
        o.cell_observations[0].config_revision=revision;o.control_ownership={'a':{f:x.name for f in target}}
        x.run(o,PolicySnapshot())
    assert not s['scores']
    assert any(r['reason']=='search_configuration_interrupted' for r in x.last_decisions)
