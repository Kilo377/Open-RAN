"""Synthetic estimator checks, never reported as simulated network evidence."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from experiments.context_identification import design
from experiments.run_rcp_perturbation import atomic
from smo import context_identify as analyzer
from smo.contracts import KPIS,build_graph,validate_influences
from smo.population import graph_view

ROOT=Path(__file__).resolve().parents[1]


def setup_design(tmp_path):
    specs,grids,contrasts=design(ROOT)
    atomic(tmp_path/'design.json',dict(specs=specs,grids=grids,contrasts=contrasts,
        scenarios={'native_1x':dict(description='Native synthetic fixture'),'burst_2x':dict(description='Burst synthetic fixture')}))
    return specs


def cases(specs,context,gain):
    result=[]
    for seed in (101,102,103):
        for arm,s in specs.items():
            y=10+seed+gain*s['offload_probability']+.03*s['drx_cycle_ms']
            result.append(dict(name=f'{context}_{seed}_{arm}',context=context,seed=seed,arm=arm,stage='grid',duration_s=1.8,
                scoring_start_ms=400.,scoring_end_ms=1600.,metrics={k:y for k in KPIS},
                proof={'source_sha256':'a'*64},origin=dict(worker_id='fixture'),performance={}))
    return result


def test_all_rcps_all_kpis_are_audited_and_new_cycle_fits_load_as_graph(tmp_path,monkeypatch):
    specs=setup_design(tmp_path);rows=cases(specs,'native_1x',2.)
    monkeypatch.setattr(analyzer,'extract',lambda *a,**kw:deepcopy(rows))
    out=tmp_path/'native';r=analyzer.analyze(tmp_path,'native_1x',out)
    assert len(r['coverage'])==5 and all(x['count']==len(KPIS)for x in r['coverage'].values())
    data=json.loads((out/'all-effects.json').read_text())
    power=next(x for x in data['estimates']if x['contrast_id']=='offload_population_range'and x['kpi']=='ue_power_w')
    assert power['estimate']==pytest.approx(2.) and power['sample_count']==3
    cycle=next(x for x in data['estimates']if x['contrast_id']=='drx_cycle_population_range'and x['kpi']=='network_goodput_mbps')
    assert cycle['estimate']==pytest.approx(.03) and cycle['scalar_component']=='cycle_ms'
    assert cycle['horizon_ms']==1500 and cycle['scoring_window_ms']==1200
    assert cycle['metric_window']==dict(kind='counter_difference',start_ms=400.,end_ms=1600.)
    fraction=next(x for x in data['estimates']if x['kpi']=='urllc_failure_fraction')
    assert fraction['metric_window']['kind']=='cumulative_ratio_including_common_warmup'
    delay=next(x for x in data['estimates']if x['kpi']=='mec_offloaded_latency_ms')
    assert delay['metric_window']['conditional_on_completion']
    assert any(e.get('sensitivities')for e in validate_influences(graph_view(build_graph()),out/'influences.json')['edges'])


def test_opposite_scenario_gains_never_collapse_to_a_pooled_zero(tmp_path,monkeypatch):
    specs=setup_design(tmp_path)
    monkeypatch.setattr(analyzer,'extract',lambda artifact,context,duration:cases(specs,context,2 if context=='native_1x'else-3))
    for context in ('native_1x','burst_2x'):analyzer.analyze(tmp_path,context,tmp_path/context)
    def estimate(context):
        d=json.loads((tmp_path/context/'all-effects.json').read_text())
        return next(r for r in d['estimates']if r['contrast_id']=='offload_population_range'and r['kpi']=='network_goodput_mbps')['estimate']
    assert estimate('native_1x')==pytest.approx(2)
    assert estimate('burst_2x')==pytest.approx(-3)


def test_untested_other_context_component_is_distinct_from_a_missing_planned_response(tmp_path,monkeypatch):
    specs=setup_design(tmp_path);path=tmp_path/'design.json';design=json.loads(path.read_text())
    design['grids'].append(dict(id='steady_only_component',parameter='ue_scheduler_weights',contexts=['steady_2x']))
    atomic(path,design);monkeypatch.setattr(analyzer,'extract',lambda *a,**kw:cases(specs,'native_1x',2.))
    analyzer.analyze(tmp_path,'native_1x',tmp_path/'out');matrix=json.loads((tmp_path/'out/all-effects.json').read_text())
    assert matrix['not_tested_designs'][0]['contrast_id']=='steady_only_component'
    assert not any(r['contrast_id']=='steady_only_component'for r in matrix['unavailable'])


def test_missing_outcomes_are_unavailable_not_zero_and_zero_effects_are_retained(tmp_path,monkeypatch):
    specs=setup_design(tmp_path);rows=cases(specs,'native_1x',0.)
    for r in rows:r['metrics']['mec_offloaded_latency_ms']=None
    monkeypatch.setattr(analyzer,'extract',lambda *a,**kw:deepcopy(rows))
    out=tmp_path/'native';analyzer.analyze(tmp_path,'native_1x',out)
    data=json.loads((out/'all-effects.json').read_text())
    assert not any(x['kpi']=='mec_offloaded_latency_ms'for x in data['estimates'])
    assert any(x['kpi']=='mec_offloaded_latency_ms'for x in data['unavailable'])
    zero=next(x for x in data['estimates']if x['contrast_id']=='offload_population_range'and x['kpi']=='network_goodput_mbps')
    assert zero['estimate']==0 and zero['confidence_interval']==[0,0]
    assert zero['fit_diagnostics']['assessment']=='constant_response_observed'


def held_state_fixture(tmp_path,monkeypatch,*,rejection=None,wrong_weight=False,allow=False):
    """Minimal telemetry fixture reproducing a one-slot serving-cell receipt lag."""
    from smo import identify
    from experiments.rcp_perturbation import vector
    from experiments.run_rcp_perturbation import atomic,sha
    specs,_,_=design(ROOT);spec=specs['native'];target=vector(spec,perturbed=True)
    study=tmp_path/'study';folder=study/'cases'/'handover';folder.mkdir(parents=True)
    ids=spec['groups']['all'];cells=['cell_0','cell_1','cell_2'];digest='a'*64
    cfg=dict(source_sha256=digest,manifest=dict(slot_duration_ms=.5,ues=[dict(ue_id=u)for u in ids],cells=[dict(cell_id=c)for c in cells]),
        configs=dict(ric=dict(xapps=[],rapps=[],freeze_learning=True),ue=dict(ues=[dict(ue_id=u,slice_id='urllc'if u in spec['groups']['urllc']else'embb')for u in ids])),rcp_perturbation=spec)
    source_files={'run_status.json':dict(completed=True,replay=dict(audits_passed=True),source_sha256=digest,cache_key='fixture'),
        'final_summary.json':dict(packet_metrics={u:dict(initial_bytes=0,offered_bytes=0,delivered_bytes=0,pending_bytes=0,overflow_bytes=0)for u in ids}),
        'config_snapshot.json':cfg,'initial_conditions.json':dict(exogenous_environment_sha256='fixture'),
        'perturbation_spec.json':spec,'performance.json':{}}
    for name,value in source_files.items():atomic(folder/name,value)
    atomic(study/'checkpoints/handover.json',dict(files_sha256={name:sha(folder/name)for name in source_files}))
    trace=[]
    for slot in range(1,10):
        trace.append(dict(slot_id=slot,ues=[dict(ue_id=u,cell_id='cell_1'if u=='ue_8'and slot>=4 else'cell_0',packet_overflow_bytes=0)for u in ids],
            traffic_arrivals={u:dict(exogenous_bytes=0)for u in ids},ue_service_events=[],
            ue_services={u:dict(ue_energy_total_j=0,ue_active_fraction=1,ue_service_policy=dict(drx=target['ue_drx_configs'][u],offload_fraction=target['ue_offload_fraction'].get(u,.5)))for u in ids},
            ap_services=dict(cells={c:dict(ap_energy_total_j=slot*.015,ap_available_fraction=1,ap_policy='active')for c in cells})))
    receipts=[dict(slot_id=1,status='applied',after={c:dict(scheduler_mode='round_robin',ue_scheduler_weights=target['ue_scheduler_weights']if c=='cell_0'else{})for c in cells})]
    if rejection:receipts.append(dict(slot_id=4,status='rejected',reason=rejection))
    if wrong_weight:receipts.append(dict(slot_id=5,status='applied',after={'cell_1':dict(scheduler_mode='round_robin',ue_scheduler_weights={'ue_8':.3})}))
    requests=[dict(slot_id=1,requested_vector=vector(spec,perturbed=False))]
    streams={'rt_sys_trace':trace,'control_receipts':receipts,'perturbation_requests':requests}
    monkeypatch.setattr(identify,'records',lambda folder,kind:iter(deepcopy(streams[kind])))
    plan=dict(source_sha256=digest,specs={'native':spec},scoring_start_ms=1,scoring_end_ms=3,duration_s=.0045,switch_ms=20,slots=9,
        expected_task_count=0,expected_task_cohort_count=0)
    if allow:plan['allowed_refresh_rejection_reasons']=['ue_scheduler_weight_target_outside_cell_scope']
    return identify.audit_case(study,dict(name='handover',arm='native',seed=1),plan,population=True)


def test_handover_receipt_cell_lag_does_not_erase_persistent_ue_control(tmp_path,monkeypatch):
    result=held_state_fixture(tmp_path,monkeypatch,rejection='ue_scheduler_weight_target_outside_cell_scope',allow=True)
    assert result['proof']['verified_held_parameter_slots']==4
    assert result['proof']['tolerated_refresh_rejections']


@pytest.mark.parametrize('options',[
    dict(rejection='ue_scheduler_weight_target_outside_cell_scope',allow=False),
    dict(rejection='some_other_failure',allow=True),
    dict(rejection='ue_scheduler_weight_target_outside_cell_scope',allow=True,wrong_weight=True),
])
def test_allowed_handover_refresh_never_suppresses_a_wrong_held_value_or_other_rejection(tmp_path,monkeypatch,options):
    with pytest.raises(ValueError):held_state_fixture(tmp_path,monkeypatch,**options)
