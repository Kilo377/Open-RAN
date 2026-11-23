"""Independent parameter fitting, units, provenance and conditional use."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from experiments.current_environment_calibration import atomic, digest, sha
from experiments.fit_current_environment_gains import fit_scalar, summarize_contrast, estimate
from smo.contracts import KPIS, build_graph, validate_influences
from smo.current_calibration import verify_current_calibration
from smo.influence_units import align_graph
from smo.population import graph_view

ROOT=Path(__file__).resolve().parents[1]


def test_essential_mirror_contains_every_original_file_consumed_by_auditor():
    import importlib.util
    from experiments.fit_current_environment_gains import MEASUREMENT_FILES
    spec=importlib.util.spec_from_file_location('pk_mirror',ROOT/'scripts/mirror-current-environment-calibration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert set(module.ESSENTIAL)==set(MEASUREMENT_FILES)
    assert {'run/rt_sys_trace.jsonl.gz','run/control_receipts.jsonl.gz','run/final_summary.json','run/config_snapshot.json'}<=set(MEASUREMENT_FILES)


def test_scalar_fit_uses_independent_seed_slopes_not_slot_pseudoreplication():
    seeds=[1,2,3]
    points=[dict(x=x,values={str(s):100*s+(s+1)*x for s in seeds}) for x in [0,.25,.5,.75,1]]
    row=fit_scalar(points,seeds)
    assert row['estimate']==pytest.approx(3)
    assert row['sample_count']==3 and row['fit_point_count']==5
    assert row['confidence_interval'][0]<2 and row['confidence_interval'][1]>4
    assert row['fit_diagnostics']['within_seed_r2']<1 # common slope exposes heterogeneity


def test_missing_zero_offload_latency_is_not_imputed_or_counted_as_a_sample():
    points=[dict(x=x,values={str(s):None if x==0 else 5+2*x for s in [1,2,3]}) for x in [0,.25,.5,.75,1]]
    row=fit_scalar(points,[1,2,3])
    assert row['fit_levels']==[.25,.5,.75,1] and row['estimate']==2
    assert row['sample_count']==3 and row['grid_points'][0]['mean']is None
    assert fit_scalar(points[:3],[1,2,3])is None


def test_categorical_difference_and_observed_zero_are_not_guessed():
    rows=[dict(seed=s,low=10+s,high=8+s) for s in [1,2,3]]
    assert summarize_contrast(rows)['estimate']==-2
    assert summarize_contrast([dict(seed=1,low=1,high=None)])is None
    zero=summarize_contrast([dict(seed=s,low=0,high=0) for s in [1,2,3]])
    assert zero['estimate']==0 and zero['confidence_interval']==[0,0]


def test_three_level_scalar_drx_timer_has_explicit_units_and_is_not_an_enum(tmp_path):
    row=dict(parameter='ue_drx_configs',kpi='ue_power_w',kind='local_derivative',scalar_component='cycle_ms',
        estimate=-.01,confidence_interval=[-.02,-.001],unit='W / ms',x_unit='ms',sample_count=3,
        horizon_ms=1200,scope='DRX-enabled population',context='current',method='seed-blocked three-level affine fit',
        data_sha256='a'*64,design='controlled_perturbation',members=['ue_0','ue_1','ue_4','ue_8'],population_control_eligible=True)
    path=tmp_path/'gain.json';atomic(path,dict(schema='oran.intervention-edges.v1',estimates=[row]))
    graph=align_graph(validate_influences(graph_view(build_graph()),path))
    selected=next(e for e in graph['edges'] if e.get('sensitivity'))
    assert selected['sensitivity']['projection_units']['valid']
    row.pop('scalar_component');atomic(path,dict(schema='oran.intervention-edges.v1',estimates=[row]))
    with pytest.raises(ValueError,match='finite contrast'):validate_influences(graph_view(build_graph()),path)


def test_conditional_weight_gains_do_not_add_two_scheduler_contexts():
    code="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const row=(mode,gain)=>({kind:'local_derivative',parameter:'ue_scheduler_weights',contrast_id:mode,
 members:['u','v'],estimate:gain,confidence_interval:[gain,gain],conditioning:{scheduler_mode:mode}});
const edge={source:'rcp:ue_scheduler_weights',sensitivities:[row('latency_aware',2),row('qos_deadline',9)]};
const base={u:1,v:1},target={u:2,v:1},frame={states:{}};
const run=mode=>c.SmoConflictRisk.project(edge,base,target,frame,false,{target:{scheduler_mode:{cell:mode}},actions:[{parameter:'ue_scheduler_weights',ue_id:'u',cell_id:'cell'}]});
process.stdout.write(JSON.stringify([run('latency_aware'),run('qos_deadline'),run('round_robin')]));"""
    lat,qos,unknown=json.loads(subprocess.check_output(['node','-e',code],cwd=ROOT,text=True))
    assert lat['drive']==1 and lat['calibrations']==['latency_aware']
    assert qos['drive']==4.5 and qos['calibrations']==['qos_deadline']
    assert unknown['drive']is None and unknown['action_coverage']['complete']is False


def fixture(tmp_path):
    ids=['ue_offload_fraction','ue_drx_configs','scheduler_mode','ue_scheduler_weights','cell_sleep_mode']
    recipe=dict(id='current',dynamics_sha256='a'*64,validated_source_archive_sha256='b'*64,
        validated_source_sha256='c'*64,profiles={n:dict(sha256=str(i)*64) for i,n in enumerate(['traffic_profile','ue_services_config','ap_services_profile','base_config'],1)},frozen_models={'x':'d'*64})
    observations=[dict(case=dict(seed=s),proof=dict(physical_source_sha256='c'*64,source_archive_sha256='b'*64)) for s in [1,2,3]]
    design=dict(seeds=[1,2,3],combination_evaluation_data_used=False,learned_xapps_enabled=False)
    matrix=dict(scenario=dict(id='current',calibration_status='controlled_parameter_calibration',dynamics_sha256='a'*64,
        source_archive_sha256='b'*64,profiles=recipe['profiles']),estimates=[dict(parameter=p,kpi='ue_power_w',calibration_environment_id='current',combination_evaluation_data_used=False) for p in ids])
    paths={'influences':tmp_path/'gains.json','design':tmp_path/'design.json','measurements':tmp_path/'data.json'}
    atomic(paths['design'],design)
    matrix['provenance']=dict(design_sha256=digest(design),measurement_sha256=digest(observations))
    for row in matrix['estimates']:row['data_sha256']=digest(observations)
    for key,value in [('influences',matrix),('measurements',dict(design_sha256=sha(paths['design']),measurements=observations))]:atomic(paths[key],value)
    cert=dict(schema='oran.smo.current-environment-calibration.v1',environment_id='current',dynamics_sha256='a'*64,
        source_archive_sha256='b'*64,profile_sha256s={k:v['sha256'] for k,v in recipe['profiles'].items()},frozen_models=recipe['frozen_models'],
        combination_evaluation_data_used=False,learned_xapps_invoked=False,audited_run_count=3,seeds=[1,2,3],
        files={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
    cert_path=tmp_path/'certificate.json';atomic(cert_path,cert)
    entry=dict(id='current',influences=str(paths['influences']),current_calibration=str(cert_path))
    return entry,recipe,cert,cert_path


def test_certificate_binds_current_environment_and_independent_data(tmp_path):
    entry,recipe,cert,path=fixture(tmp_path)
    result=verify_current_calibration(entry,recipe)
    assert result['current_calibration'] and result['audited_runs']==3
    assert result['combination_evaluation_data_used']is False
    old=deepcopy(recipe);old['profiles']['traffic_profile']['sha256']='e'*64
    with pytest.raises(ValueError,match='profiles differ'):verify_current_calibration(entry,old)
    cert['combination_evaluation_data_used']=True;atomic(path,cert)
    with pytest.raises(ValueError,match='independent'):verify_current_calibration(entry,recipe)


def test_changed_matrix_and_unverified_current_gains_are_rejected(tmp_path):
    entry,recipe,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='certificate'):verify_current_calibration({k:v for k,v in entry.items() if k!='current_calibration'},recipe)
    Path(entry['influences']).write_text('{}')
    with pytest.raises(ValueError,match='Changed calibration evidence'):verify_current_calibration(entry,recipe)


def test_origin_bridge_rejects_a_different_network_start_or_changed_prefix(tmp_path):
    entry,recipe,cert,path=fixture(tmp_path)
    recipe['validated_origin']={'sha256':'f'*64};recipe['baseline_slot']=2040
    bridge=tmp_path/'origin.json'
    proof=dict(schema='oran.current-calibration-origin-equivalence.v1',environment_id='current',reference_origin_sha256='f'*64,
        checked_slots=2040,all_target_kpis_exactly_equal=True,physical_prefix_hashes_equal=True,deployed_combination_after_kpis_used=False)
    atomic(bridge,proof);cert['files']['origin_equivalence']=dict(path=str(bridge),sha256=sha(bridge));atomic(path,cert)
    assert verify_current_calibration(entry,recipe)['current_calibration']
    proof['physical_prefix_hashes_equal']=False;atomic(bridge,proof);cert['files']['origin_equivalence']['sha256']=sha(bridge);atomic(path,cert)
    with pytest.raises(ValueError,match='physical prefix'):verify_current_calibration(entry,recipe)
    proof['physical_prefix_hashes_equal']=True;proof['reference_origin_sha256']='e'*64
    atomic(bridge,proof);cert['files']['origin_equivalence']['sha256']=sha(bridge);atomic(path,cert)
    with pytest.raises(ValueError,match='different UI origin'):verify_current_calibration(entry,recipe)


def test_full_fit_produces_compatible_population_units_without_historical_gains(tmp_path):
    users=['ue_0','ue_1','ue_2','ue_4','ue_6','ue_8'];cells=['cell_0','cell_1','cell_2']
    drx=dict(enabled=True,cycle_ms=40,on_duration_ms=4,inactivity_ms=8,wakeup_ms=.5,offset_ms=0)
    base=dict(ue_offload_fraction={'ue_0':.5,'ue_4':.8},ue_drx_configs={u:deepcopy(drx) for u in users},
        scheduler_mode=dict.fromkeys(cells,'round_robin'),ue_scheduler_weights=dict.fromkeys(users,1),cell_sleep_mode=dict.fromkeys(cells,'active'))
    arms=[dict(id='native',worker='agx2',kind='reference',reference='native',parameter=None,vector=base)]
    for ident,mode in [('latency','latency_aware'),('qos','qos_deadline')]:
        v=deepcopy(base);v['scheduler_mode']=dict.fromkeys(cells,mode)
        arms.append(dict(id=ident,worker='agx2',kind='reference',reference='native',parameter='scheduler_mode',vector=v))
    for i,x in enumerate([0,.25,.5,.75,1]):
        v=deepcopy(base);v['ue_offload_fraction']=dict.fromkeys(['ue_0','ue_4'],x)
        arms.append(dict(id='offload'+str(i),worker='agx2',kind='grid',reference='native',parameter='ue_offload_fraction',
            component='mec_workload',members=['ue_0','ue_4'],level=x,vector=v))
    for i,x in enumerate([20,80]):
        v=deepcopy(base)
        for u in users:v['ue_drx_configs'][u]['cycle_ms']=x
        arms.append(dict(id='drx'+str(i),worker='agx2',kind='grid',reference='native',parameter='ue_drx_configs',
            component='drx_enabled_population',scalar_component='cycle_ms',members=users,level=x,vector=v))
    for i,x in enumerate([.5,2]):
        v=deepcopy(arms[1]['vector']);v['ue_scheduler_weights'].update(ue_0=x,ue_4=x)
        arms.append(dict(id='weight'+str(i),worker='agx2',kind='grid',reference='latency',parameter='ue_scheduler_weights',
            component='mec_priority',members=['ue_0','ue_4'],level=x,vector=v))
    for mode in ['micro_sleep','light_sleep']:
        v=deepcopy(base);v['cell_sleep_mode']=dict.fromkeys(cells,mode)
        arms.append(dict(id='sleep_'+mode,worker='agx2',kind='category',reference='native',parameter='cell_sleep_mode',vector=v))
    design=dict(environment_id='current',seeds=[1,2,3],source_pins=dict(sha256='a'*64),source_archive_sha256='b'*64,
        profiles={},traffic_load=1.5,scoring_start_slot=2440,scoring_end_slot=4440,arms=arms)
    data=[]
    for arm in arms:
        for seed in design['seeds']:
            data.append(dict(worker_id='agx2',case=dict(seed=seed,arm=arm['id']),
                metrics={'kpi:'+k:10+seed+sum(arm['vector']['ue_offload_fraction'].values()) for k in KPIS}))
    matrix=estimate(design,data);path=tmp_path/'current.json';atomic(path,matrix)
    graph=align_graph(validate_influences(graph_view(build_graph()),path))
    assert {r['parameter'] for r in matrix['estimates']}==set(base)
    assert all(r['data_sha256']==digest(data) and not r['combination_evaluation_data_used'] for r in matrix['estimates'])
    assert all(r['projection_units']['valid'] for e in graph['edges'] for r in e.get('sensitivities',[]))
    assert len([r for r in matrix['estimates'] if r['parameter']=='scheduler_mode' and r['kpi']=='ue_power_w'])==2
    from smo.statistical_filter import filter_graph
    screened=filter_graph(graph,confidence_level=.90,retain_measured_contexts=True)
    assert screened['statistical_filter']['recalculated_estimate_count']==len(matrix['estimates'])
    assert all(r['independent_unit']==('seed_pair'if r['kind']=='finite_contrast'else'seed_block')for r in matrix['estimates'])
