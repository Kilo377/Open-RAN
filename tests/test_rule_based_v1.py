"""Causal, bounded functional baselines with measured fixtures and real consumers."""
from copy import deepcopy
from types import SimpleNamespace as NS
import math
import pytest
from models.policy import PolicySnapshot,PolicyUpdate
from models.action import ConfigAction
from ric.near_rt.rule_based_v1 import RuleXApp,CATALOG,PROFILES
from ric.near_rt.rule_based_v1.apps import waterfill_caps
from ric.xapp_contract import KPI_REGISTRY,project,validate_action,validate_commands


def view(slot=1000):
    configs={cid:dict(n_prbs=8,tx_power_dbm=29.,power_cap_dbm=None,nominal_tx_power_dbm=30.,
        cell_sleep_mode='active',codebook_id=0,beam_profile_id=0,prb_power_w=None,power_group_size_prbs=1,
        ue_prb_limits={},cio_offsets_db={},neighbor_cell_ids=None,time_to_trigger_ms=40.,steering_hysteresis_db=3.,
        ret_enabled=False,ret_tilt_deg=0.,ret_codebook_id=0,scheduler_mode='round_robin',latency_bias=1.,slice_weight_adjustment={}) for cid in ['a','b']}
    def metadata():return dict(valid=True,sample_count=10,produced_at_slot=slot,delivered_at_slot=slot,age_slots=0)
    ues=[]
    for i,cid in enumerate(configs):
        u=NS(ue_id='u'+str(i),serving_cell_id=cid,slice_id='urllc' if i==0 else 'embb',priority=1,
            queue_bytes=2000,hol_delay_ms=20.,latency_budget_ms=10.,throughput_mbps=.1,offered_rate_bps=1e6,
            requested_dl_rate_bps=1e6,reference_sinr_db=0.,rsrp_dbm=-85.,subband_cqi=[5]*8,
            subband_prb_groups=[[j] for j in range(8)],resource_allocation={'prbs':[0,1]},association={'cell_id':cid},
            candidate_rsrp_dbm={'a':-80.,'b':-75.},candidate_sinr_db={'a':0.,'b':10.},connection_state='connected',
            handover_pending_target_cell_id=None,mobility_events=[],beam_reference_reports=[dict(cell_id=cid,codebook_id=cb,
                beam_profile_id=0,reference_sinr_db=10.*cb,valid=True) for cb in [0,1]],
            deadline_misses_window=1,deadline_packets_window=10,business_window_s=.1,first_tx_bler=.1)
        u.kpi_metadata={k:metadata() for k,s in KPI_REGISTRY.items() if s.entity=='ue'};ues.append(u)
    cells=[NS(cell_id=cid,executed_config=cfg,config_produced_at_slot=slot,cell_sleep_mode='active',
        resource_grid_utilization=.9 if cid=='a' else .1,kpi_metadata={k:metadata() for k,s in KPI_REGISTRY.items() if s.entity=='cell'}) for cid,cfg in configs.items()]
    return NS(slot_id=slot,ue_observations=ues,cell_observations=cells,executed_configs=deepcopy(configs),
        rule_report_produced_at_slot=slot,paper_interfaces_enabled=True)


@pytest.mark.parametrize('kind',list(CATALOG))
def test_all_apps_deterministic_and_owned(kind):
    o=view();a=RuleXApp(kind);b=RuleXApp(kind)
    assert a.contract.available
    x=a.run(deepcopy(o),PolicySnapshot());y=b.run(deepcopy(o),PolicySnapshot())
    assert x==y
    for action in x.config_actions:assert not validate_action(action,a.contract,o)
    assert not validate_commands(x.commands,a.contract,o)
    assert all(k in KPI_REGISTRY for k in a.contract.required_kpis)
    assert a.run(o,PolicySnapshot()).commands==[]
    assert all(e['reason'] in ('no_new_report','action_dwell') for e in a.last_decisions)


@pytest.mark.parametrize('kind',['throughput_power','rf_energy','ue_fairness','slice_sla','latency_guard','interference_coordination','ret_coverage','beam_management','load_balancing'])
def test_control_has_actual_request_and_bounds(kind):
    o=view();app=RuleXApp(kind);d=app.run(o,PolicySnapshot())
    actions=[a for a in d.config_actions if a.target_cell_id]
    assert actions
    for a in actions:
        if a.prb_power_w:assert len(a.prb_power_w)==8 and sum(a.prb_power_w)<=1+1e-12 and min(a.prb_power_w)>=0
        if a.ue_prb_limits:assert sum(a.ue_prb_limits.values())<=8
        if a.ret_tilt_deg is not None:assert 0<=a.ret_tilt_deg<=15
        if kind=='slice_sla':assert a.scheduler_mode=='slice_aware'


def test_partial_entities_hold_and_no_missing_zero_fill():
    o=view();app=RuleXApp('rf_energy');o.ue_observations[0].kpi_metadata['reference_sinr_db']['valid']=False
    d=app.run(o,PolicySnapshot())
    assert not any(a.target_cell_id=='a' for a in d.config_actions)
    assert any(a.target_cell_id=='b' for a in d.config_actions)
    assert any(e['reason']=='invalid_entity_kpi' for e in app.last_decisions)


def test_ts_three_distinct_reports_and_cooldown():
    a=RuleXApp('traffic_steering')
    for slot in [1000,1005]:assert not a.run(view(slot),PolicySnapshot()).commands
    o=view(1010);d=a.run(o,PolicySnapshot());assert len(d.commands)==1
    assert d.commands[0].target_ue_id=='u0' and d.commands[0].target_cell_id=='b'
    for slot in [1015,1020,1025]:assert not a.run(view(slot),PolicySnapshot()).commands


def test_mro_samples_reasons_and_pingpong():
    app=RuleXApp('mobility_robustness');o=view()
    assert not app.run(o,PolicySnapshot()).config_action.time_to_trigger_ms
    o=view(1500)
    for u in o.ue_observations:
        u.mobility_events=[dict(kind='handover_success',slot_id=1000+j*10,ue_id=u.ue_id,
            from_cell='a' if j%2==0 else 'b',to_cell='b' if j%2==0 else 'a') for j in range(5)]
    d=app.run(o,PolicySnapshot());assert d.config_action.time_to_trigger_ms==60
    fresh=view(1600)
    for u in fresh.ue_observations:
        u.mobility_events=[dict(kind='handover_failure',slot_id=1000+j*10,ue_id=u.ue_id,reason='target_unavailable') for j in range(5)]
    assert app.run(fresh,PolicySnapshot()).config_action.time_to_trigger_ms is None


def test_sleep_migrate_first_and_information_guard():
    app=RuleXApp('cell_sleep');o=view();o.cell_observations[0].resource_grid_utilization=.01
    for u in o.ue_observations:u.hol_delay_ms=0.
    assert not app.run(o,PolicySnapshot()).commands
    o=view(1500);o.cell_observations[0].resource_grid_utilization=.01
    for u in o.ue_observations:u.hol_delay_ms=0.
    d=app.run(o,PolicySnapshot());assert d.commands and not any(a.cell_sleep_mode=='deep_sleep' for a in d.config_actions)
    o=view(2000);o.cell_observations[0].resource_grid_utilization=.01;o.ue_observations=o.ue_observations[1:]
    d=app.run(o,PolicySnapshot());assert d.config_action.cell_sleep_mode=='deep_sleep'
    o=view(2500);o.cell_observations[0].cell_sleep_mode='deep_sleep';o.executed_configs['a']['cell_sleep_mode']='deep_sleep'
    o.cell_observations[1].resource_grid_utilization=.9
    assert app.run(o,PolicySnapshot()).config_action.cell_sleep_mode=='active'


def test_beam_ret_compatibility_and_causal_ret():
    app=RuleXApp('beam_management');o=view()
    for c in o.cell_observations:c.executed_config['ret_enabled']=True
    assert app.run(o,PolicySnapshot()).config_action.codebook_id is None
    ret=RuleXApp('ret_coverage');d=ret.run(view(),PolicySnapshot());a=d.config_action
    assert len([x for x in d.config_actions if x.target_cell_id])==1
    o=view(2000)
    for c in o.cell_observations:
        if c.cell_id==a.target_cell_id:
            c.executed_config.update(ret_enabled=True,ret_tilt_deg=a.ret_tilt_deg)
    o.executed_configs['b']['tx_power_dbm']=28.
    ret.run(o,PolicySnapshot())
    assert any(e['reason']=='ret_effect_not_attributable' for e in ret.last_decisions)


def test_policy_parameter_override_roundtrip_and_physical_bounds():
    from ric.non_rt.policy_store import PolicyStore
    store=PolicyStore();store.apply_update(PolicyUpdate(rule_based_parameters={'rf_energy':{'power_step_db':1.}}))
    snap=store.get_snapshot();assert snap.rule_based_parameters['rf_energy']['power_step_db']==1
    app=RuleXApp('rf_energy');assert app.run(view(),snap).config_action.tx_power_dbm==30. # clamp
    with pytest.raises(ValueError):RuleXApp('ret_coverage',parameters={'ret_max_deg':20})
    with pytest.raises(ValueError):RuleXApp('ue_fairness',parameters={'made_up':1})


def test_integer_max_min_demand_and_ties():
    assert waterfill_caps({'a':1,'b':8},{'a':1,'b':1},8)=={'a':1,'b':7}
    assert sum(waterfill_caps({'a':8,'b':8},{'a':2,'b':1},8).values())==8


def test_factory_configuration_opt_in_and_namespace():
    from phy.rt_sys.scenario import build_case
    from core.factory import build_controller_registry
    for key in [*CATALOG,*PROFILES]:
        _,c,_=build_case(config_path='configs/rt_sys/rule_based_v1/'+key+'.json')
        reg=build_controller_registry(c['ric']);assert reg.xapps and not reg.rapps
        assert all(a.name.startswith('rule_') and a.contract.available for a in reg.xapps)
        assert c['phy']['rt_sys']['paper_interfaces']['rule_observations']


def test_leases_rearbitrate_disabled_and_restore_only_owned_fields():
    from ric.near_rt.rule_based_v1.leases import RuleLeases
    low=RuleXApp('throughput_power',name='low');high=RuleXApp('interference_coordination',name='high',priority=60)
    enabled={'low','high'};manager=NS(e2_interfaces=[NS(xapp=low),NS(xapp=high)],
        _is_xapp_enabled=lambda n:n in enabled,current_action_by_scope={})
    leases=RuleLeases();o=NS(paper_report={'values':{'cell':{'a':{'executed_config':{'prb_power_w':None}}}}})
    winners={};scopes={}
    leases.reconcile(manager,o,winners,scopes,[ConfigAction(target_cell_id='a',source_name='low',prb_power_w=[.1]),
        ConfigAction(target_cell_id='a',source_name='high',prb_power_w=[.2])])
    assert winners['cell:a']['prb_power_w'][1]==[.2]
    enabled.remove('high');winners={};leases.reconcile(manager,o,winners,scopes,[])
    assert winners['cell:a']['prb_power_w'][1]==[.1]
    enabled.clear();winners={};leases.reconcile(manager,o,winners,scopes,[])
    assert winners['cell:a']['clear_prb_power'][1] is True


def test_public_rule_report_truth_isolation_and_partial_contract():
    from test_xapp_contract import fixture
    o=fixture();o.paper_report={'values':{'ue':{'u':{'mobility_events':[], 'beam_reference_reports':[]}},
        'cell':{'c':{'executed_config':{'n_prbs':8}}}},'produced_at_slot':3,'window_slots':2,
        'reference_window_slots':4,'connection_model_enabled':True}
    o.paper_interfaces_enabled=True;o.delivered_at_slot=5
    app=RuleXApp('mobility_robustness');v,errors=project(o,app.contract)
    assert not errors and v.cell_observations[0].executed_config=={'n_prbs':8}
    assert v.ue_observations[0].mobility_events==[]
    assert not hasattr(v,'rt_channel_snapshot')
    v.cell_observations[0].executed_config['n_prbs']=99
    assert o.paper_report['values']['cell']['c']['executed_config']['n_prbs']==8


@pytest.mark.parametrize('kind',['throughput_power','ue_fairness','slice_sla','beam_management','ret_coverage'])
def test_known_channel_closed_loop_and_multicell_consumers(tmp_path,kind):
    from test_paper_interfaces import make_sim
    from ric.interfaces.e2_like import E2LikeInterface
    sim=make_sim(tmp_path,cells=2,slots=12,config={'rule_observations':True,'beam_reports':True})
    app=RuleXApp(kind,parameters={'min_action_ms':1})
    sim.controller_manager.e2_interfaces=[E2LikeInterface(app)]
    sim.controller_manager._initial_rapp_run_completed=True
    scopes=set()
    for _ in range(12):
        sim.step()
        scopes.update(a.target_cell_id for a in sim.state.control.arbitration_trace.received_config_actions if a.target_cell_id)
    assert app.statistics['action_requested']>0,(kind,app.statistics)
    assert scopes
    if kind=='slice_sla':assert any(r.scheduler.scheduler_mode=='slice_aware' for r in sim.state.managed_cells.values())
    if kind=='throughput_power':assert any(r.cell.prb_power_w is not None for r in sim.state.managed_cells.values())
    for row in sim.phy_adapter.trace['cells'].values():assert row['average_used_w']<=row['budget_w']+1e-10
    for ue in sim.state.ues:
        m=ue.packet_ledger.stats
        assert m['initial_bytes']+m['offered_bytes']==m['pending_bytes']+m['delivered_bytes']+m['overflow_bytes']
    assert not any(x['reason'].startswith('kpi_contract_rejected') for x in sim.state.control.arbitration_trace.dropped_items_with_reason)


def test_nonrt_catalog_exposes_only_enabled_rule_namespace():
    from ric.non_rt.intent_orchestration_helpers import available_xapp_catalog_for_llm
    entries=available_xapp_catalog_for_llm(['rule_slice_sla','rule_mobility_robustness'],backend='sionna_rt_sys')
    assert len(entries)==2 and all(e['available'] and not e['training_required'] for e in entries)
    assert not any(e['xapp_name'].startswith('rule_') for e in available_xapp_catalog_for_llm())


def test_ret_suppresses_held_beam_requests_before_and_after_actuation():
    from ric.near_rt.rule_based_v1.leases import RuleLeases
    ret=RuleXApp('ret_coverage');beam=RuleXApp('beam_management')
    manager=NS(e2_interfaces=[NS(xapp=ret),NS(xapp=beam)],_is_xapp_enabled=lambda n:True,
        current_action_by_scope={'cell:a':ConfigAction(target_cell_id='a',codebook_id=1,beam_profile_id=0)})
    leases=RuleLeases();o=NS(paper_report={'values':{'cell':{'a':{'executed_config':{'ret_enabled':False}}}}})
    received=[ConfigAction(target_cell_id='a',source_name=beam.name,codebook_id=1,beam_profile_id=0),
        ConfigAction(target_cell_id='a',source_name=ret.name,ret_enabled=True,ret_codebook_id=0,ret_tilt_deg=1.)]
    winners={'cell:a':{'codebook_id':(60,1,beam.name)}};scopes={}
    leases.reconcile(manager,o,winners,scopes,received)
    assert 'codebook_id' not in winners['cell:a'] and manager.current_action_by_scope['cell:a'].codebook_id is None
    assert winners['cell:a']['ret_enabled'][1]
    winners={};leases.reconcile(manager,o,winners,scopes,[])
    assert 'codebook_id' not in winners['cell:a']


@pytest.mark.parametrize('kind',['traffic_steering','load_balancing','interference_coordination','cell_sleep'])
def test_real_projection_legacy_candidate_pairs_are_accepted(kind):
    a=RuleXApp(kind)
    for slot in [1000,1005,1010]:
        o=view(slot)
        for u in o.ue_observations:u.candidate_rsrp_dbm=tuple(u.candidate_rsrp_dbm.items())
        d=a.run(o,PolicySnapshot())
    if kind=='traffic_steering':assert d.commands
    if kind=='load_balancing':assert a.statistics['action_requested']
    if kind=='interference_coordination':assert a.statistics['action_requested']


def test_invalid_nonrt_update_is_rejected_without_state_mutation():
    app=RuleXApp('rf_energy');policy=PolicySnapshot(rule_based_parameters={'rf_energy':{'max_power_dbm':-100}})
    result=app.run(view(),policy)
    assert result.config_action.tx_power_dbm is None and app.last_decisions[0]['reason']=='nonrt_policy_rejected'
    assert app.parameters['max_power_dbm']==30
