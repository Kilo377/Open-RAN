from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from experiments.fit_sleep_profile_gains import collect_pairs, digest, estimate_rows, overlay, topology, _verify_seal, _verify_scene


BASE = {c: 'active' for c in ('cell_0', 'cell_1', 'cell_2')}
PROFILE = dict(cell_0='light_sleep', cell_1='deep_sleep', cell_2='light_sleep')


def fixture():
    plan = dict(profiles={'native_1x': {'joint': PROFILE}}, training_seeds=[11, 12, 13],
        reference_profile=BASE, plan_sha256='a'*64, switch_slot=2040,
        scoring_start_slot=2440, scoring_end_slot=4440, slot_duration_ms=.5)
    nodes = [dict(id='rcp:cell_sleep_mode', kind='rcp', dtype='categorical')]
    nodes += [dict(id='kpi:'+k, kind='kpi', primary=True, unit=u) for k, u in (
        ('ap_power_w', 'W'), ('urllc_failure_fraction', 'ratio'), ('mec_offloaded_latency_ms', 'ms'))]
    graph = dict(metric_scope='ue_population.v1', nodes=nodes, edges=[dict(
        id='influence:cell_sleep_mode:'+k, kind='influences',
        source='rcp:cell_sleep_mode', target='kpi:'+k) for k in ('ap_power_w', 'urllc_failure_fraction')])
    pairs = []
    # Large variation in reference power cancels within independent seed pairs.
    for seed, low, delta in ((11, 50., -8.), (12, 100., -10.), (13, 200., -12.)):
        before = {'kpi:ap_power_w': dict(value=low, aggregation='counter_delta_over_equal_window'),
            'kpi:urllc_failure_fraction': dict(value=0.),
            'kpi:mec_offloaded_latency_ms': dict(value=15., sample_count=4)}
        after = deepcopy(before)
        after['kpi:ap_power_w']['value'] += delta
        after['kpi:urllc_failure_fraction']['value'] = .02
        after['kpi:mec_offloaded_latency_ms']['value'] = 17.
        pairs.append(dict(scene='native_1x', seed=seed, profile_id='joint',
            reference=before, profile=after, audit={'valid': True, 'seed': seed}))
    return plan, graph, pairs


def test_seed_paired_whole_profile_fit_not_slot_pseudoreplication():
    plan, graph, pairs = fixture()
    rows, missing = estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)
    power = next(r for r in rows if r['kpi']=='ap_power_w')
    assert not missing
    assert power['estimate']==-10.
    assert power['sample_count']==3 and power['observed_run_count']==6
    assert power['standard_error']==pytest.approx(2/math.sqrt(3))
    assert power['confidence_interval']==pytest.approx([-10-4.302652729696142*2/math.sqrt(3),
                                                      -10+4.302652729696142*2/math.sqrt(3)])
    assert power['from']==BASE and power['to']==PROFILE
    assert power['members']==sorted(BASE)
    assert power['independent_unit']=='seed_pair'
    assert power['scoring_window_ms']==1000 and power['horizon_ms']==1200
    assert power['validation_combination_outcomes_used'] is False


def test_undefined_mec_pairs_are_never_zero_imputed():
    plan, graph, pairs = fixture()
    pairs[0]['profile']['kpi:mec_offloaded_latency_ms'] = dict(value=None, reason='no_mature_task_samples')
    rows, missing = estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)
    mec = next(r for r in rows if r['kpi']=='mec_offloaded_latency_ms')
    assert mec['sample_count']==2 and mec['estimate']==2 and mec['missing_pair_count']==1
    assert mec['seed_pairs'][0]['paired_difference'] is None
    pairs[1]['reference']['kpi:mec_offloaded_latency_ms']['value'] = None
    rows, missing = estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)
    assert all(r['kpi']!='mec_offloaded_latency_ms' for r in rows)
    assert missing[0]['reason']=='fewer_than_two_valid_independent_pairs'
    assert len(missing[0]['seed_pairs'])==3


def test_zero_url_reference_keeps_canonical_probability_scale():
    plan, graph, pairs = fixture()
    rows, _ = estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)
    row = next(r for r in rows if r['kpi']=='urllc_failure_fraction')
    assert row['estimate']==.02
    assert row['importance']['kpi_scale']==1.
    assert row['importance']['signed_reference_importance']==.02


def test_retained_legacy_energy_node_remains_explicitly_unmeasurable():
    plan, graph, pairs = fixture()
    graph['nodes'].append(dict(id='kpi:ue_energy_j', kind='kpi', unit='J'))
    for pair in pairs:
        for arm in ('reference', 'profile'):
            pair[arm]['kpi:ue_energy_j'] = dict(value=None, reason='unsupported_window_metric')
    rows, missing = estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)
    assert not any(r['kpi']=='ue_energy_j' for r in rows)
    legacy = next(m for m in missing if m['kpi']=='ue_energy_j')
    assert all(p['low']['reason']=='unsupported_window_metric' for p in legacy['seed_pairs'])


def test_overlay_keeps_original_topology_and_inputs(tmp_path):
    plan, graph, pairs = fixture()
    rows, _ = estimate_rows(pairs, plan, 'native_1x', graph, digest(pairs))
    before = deepcopy(graph)
    result, excluded, audit = overlay(graph, rows, tmp_path/'influences.json', provenance={'data':'independent'})
    assert graph==before
    assert topology(result)==topology(graph)
    assert [r['kpi'] for r in excluded]==['mec_offloaded_latency_ms']
    assert audit['all_link_gains_complete'] and audit['estimate_count']==2
    assert result['profile_calibration']['topology_unchanged'] is True
    assert all(len(e['sensitivities'])==1 for e in result['edges'])


def test_duplicate_or_missing_training_seed_is_rejected():
    plan, graph, pairs = fixture()
    for bad in (pairs[:-1], [pairs[0], pairs[0], pairs[-1]]):
        with pytest.raises(ValueError, match='Every preregistered'):
            estimate_rows(bad, plan, 'native_1x', graph, 'b'*64)


def test_manifest_cannot_point_to_validation_or_change_window():
    plan, graph, pairs = fixture()
    base = dict(schema='oran.profile-calibration-cases.v1', calibration_plan_sha256=plan['plan_sha256'],
        heldout_after_kpis_read=False, scoring_window=dict(start_slot=2440,end_slot=4440,slot_duration_ms=.5), cases=[])
    with pytest.raises(ValueError, match='Complete, unique'):
        collect_pairs(base, plan, {'native_1x':graph})
    base['scoring_window']['start_slot']=2040
    with pytest.raises(ValueError, match='scoring window'):
        collect_pairs(base, plan, {'native_1x':graph})
    base['heldout_after_kpis_read']=True
    with pytest.raises(ValueError, match='manifest contract'):
        collect_pairs(base, plan, {'native_1x':graph})


def test_canonical_measurements_are_converted_before_fitting_and_importance():
    plan, graph, pairs = fixture()
    original = deepcopy(pairs)
    raw, _ = estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)
    units = {'ap_power_w':'mW', 'urllc_failure_fraction':'%', 'mec_offloaded_latency_ms':'s'}
    for node in graph['nodes']:
        if node['kind']=='kpi':node['unit']=units[node['id'][4:]]
    converted, _ = estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)
    for a, b, multiplier in zip(raw, converted, (1000., 100., .001)):
        assert b['estimate']==pytest.approx(a['estimate']*multiplier)
        assert b['confidence_interval']==pytest.approx([v*multiplier for v in a['confidence_interval']])
        assert b['reference_kpi_mean']==pytest.approx(a['reference_kpi_mean']*multiplier)
        assert b['metric_unit_conversion_factor']==multiplier
        assert b['importance']['signed_reference_importance']==pytest.approx(a['importance']['signed_reference_importance'])
    url = next(r for r in converted if r['kpi']=='urllc_failure_fraction')
    assert url['estimate']==2 and url['importance']['kpi_scale']==100
    assert url['importance']['signed_reference_importance']==.02
    assert pairs==original


def test_incompatible_graph_unit_is_rejected_instead_of_relabeling():
    plan, graph, pairs = fixture()
    graph['nodes'][1]['unit']='ms'
    with pytest.raises(ValueError, match='compatible canonical unit'):
        estimate_rows(pairs, plan, 'native_1x', graph, 'b'*64)


def sealed_fixture(tmp_path):
    """All evidence is synthetic; never load a physical training/validation run."""
    from phy.rt_sys.scenario import build_case
    from experiments.urllc_scenarios import apply_named_scenario
    plan, _, _ = fixture()
    plan['validation_seeds_excluded']=[1, 2]
    case=dict(id='c'*32, scene='native_1x', seed=11, arm='joint', profile=deepcopy(PROFILE),
        source_sha256='f'*64)
    run=tmp_path/'worker'/'jobs'/case['id']/'data'/'run'
    run.mkdir(parents=True)
    case['run_path']=str(run)
    root=Path(__file__).resolve().parents[1]
    manifest, configs, _=build_case('munich',3,6,11,1,device='cpu',
        config_path=root/'configs/rt_sys/agx_munich_6ue_depth2_fast.json',traffic_load=1,xapp_pool='chongyu')
    apply_named_scenario(manifest, configs, 'native')
    config=dict(manifest=manifest,configs=configs,control_mode='sleep_profile_probe',source_sha256=case['source_sha256'],
        sleep_profile_probe=dict(profile_id='joint', calibration_plan_sha256=plan['plan_sha256']))
    (run/'config_snapshot.json').write_text(json.dumps(config))
    (run/'run_status.json').write_text('{}')
    (run/'rt_sys_trace.jsonl').write_text('synthetic-sealed-evidence-only\n')
    (run.parent/'parameter-admission-proof.json').write_text('{}')
    seal=dict(completed=True, source_sha256=case['source_sha256'],
        case={k:deepcopy(case[k]) for k in ('id','scene','seed','arm','profile')},
        physical_plan_sha256=plan['plan_sha256'],heldout_after_kpis_read=False)
    checkpoint=run.parents[3]/'checkpoints'/(case['id']+'.json')
    checkpoint.parent.mkdir()
    def reseal():
        seal['files_sha256']={str(p.relative_to(run.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in run.parents[1].rglob('*') if p.is_file()}
        checkpoint.write_text(json.dumps(seal))
    reseal()
    return plan,case,config,seal,checkpoint,reseal


def test_seal_verifies_complete_registered_case(tmp_path):
    plan,case,_,seal,_,_=sealed_fixture(tmp_path)
    assert _verify_seal(case,plan)==seal


@pytest.mark.parametrize('field,bad',[('scene','steady_2x'),('seed',12),('arm','REF'),
    ('profile',BASE),('id','d'*32)])
def test_sealed_case_cannot_be_reassigned_to_another_identity(tmp_path,field,bad):
    plan,case,_,seal,checkpoint,_=sealed_fixture(tmp_path)
    seal['case'][field]=bad
    checkpoint.write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='Unsealed or mismatched'):
        _verify_seal(case,plan)


@pytest.mark.parametrize('mutation',['load','urllc_name','burst_overlay','logical_channel'])
def test_scene_is_checked_against_actual_arrivals_not_only_sealed_label(tmp_path,mutation):
    plan,case,config,_,_,reseal=sealed_fixture(tmp_path)
    if mutation=='load':config['configs']['ue']['ues'][0]['arrival_rate_bps']*=2
    elif mutation=='urllc_name':config['manifest']['urllc_scenario']['name']='burst'
    elif mutation=='burst_overlay':config['manifest']['traffic_profile']={'sha256':'e'*64}
    else:config['configs']['ue']['ues'][0]['logical_channels']=[dict(arrival_rate_bps=1)]
    (Path(case['run_path'])/'config_snapshot.json').write_text(json.dumps(config));reseal()
    with pytest.raises(ValueError,match='actual arrival recipe'):
        _verify_seal(case,plan)


def test_validation_seed_and_unsealed_tamper_are_rejected(tmp_path):
    plan,case,config,_,_,_=sealed_fixture(tmp_path)
    plan['validation_seeds_excluded'].append(case['seed'])
    with pytest.raises(ValueError,match='registered training probe'):_verify_seal(case,plan)
    plan['validation_seeds_excluded'].remove(case['seed'])
    (Path(case['run_path'])/'rt_sys_trace.jsonl').write_text('changed after seal')
    with pytest.raises(ValueError,match='changed after sealing'):_verify_seal(case,plan)


@pytest.mark.parametrize('scene,load,scenario',[('native_1x',1.,'native'),
    ('steady_2x',2.,'native'),('burst_2x',2.,'burst')])
def test_all_registered_actual_scene_recipes_pass_and_wrong_scene_fails(scene,load,scenario):
    from phy.rt_sys.scenario import build_case
    from experiments.urllc_scenarios import apply_named_scenario
    root=Path(__file__).resolve().parents[1]
    manifest,configs,_=build_case('munich',3,6,1,1,device='cpu',
        config_path=root/'configs/rt_sys/agx_munich_6ue_depth2_fast.json',traffic_load=load,xapp_pool='chongyu')
    apply_named_scenario(manifest,configs,scenario)
    config=dict(manifest=manifest,configs=configs)
    _verify_scene(config,scene)
    other='steady_2x' if scene!='steady_2x' else 'native_1x'
    with pytest.raises(ValueError,match='actual arrival recipe'):_verify_scene(config,other)


def collection_fixture(tmp_path):
    plan=dict(schema='oran.parameter-profile-calibration-plan.v1',
        profiles={s:{'joint':deepcopy(PROFILE)} for s in ('native_1x','steady_2x','burst_2x')},
        training_seeds=[20261125,20261126,20261127], validation_seeds_excluded=[20261201,20261202],
        reference_profile=deepcopy(BASE),switch_slot=2040,total_slots=4440,
        scoring_start_slot=2440,scoring_end_slot=4440,slot_duration_ms=.5)
    plan['plan_sha256']=digest(plan)
    cases=[dict(id=str(i),scene=scene,seed=seed,arm=arm,source_sha256='a'*64,
        run_path=str(tmp_path/str(i))) for i,(scene,seed,arm) in enumerate(
        (s,n,a) for s in plan['profiles'] for n in plan['training_seeds'] for a in ('REF','joint'))]
    manifest=dict(schema='oran.profile-calibration-cases.v1',calibration_plan_sha256=plan['plan_sha256'],
        heldout_after_kpis_read=False,scoring_window=dict(start_slot=2440,end_slot=4440,slot_duration_ms=.5),cases=cases)
    return plan,manifest


@pytest.mark.parametrize('mutation,reason',[('source','source versions differ'),
    ('id','unique IDs and paths'),('path','unique IDs and paths')])
def test_collection_rejects_source_mixing_and_reused_run_identity_before_reading_runs(tmp_path,mutation,reason):
    plan,manifest=collection_fixture(tmp_path)
    if mutation=='source':manifest['cases'][0]['source_sha256']='b'*64
    elif mutation=='id':manifest['cases'][0]['id']=manifest['cases'][1]['id']
    else:manifest['cases'][0]['run_path']=manifest['cases'][1]['run_path']
    with pytest.raises(ValueError,match=reason):collect_pairs(manifest,plan,{})


def test_self_consistent_plan_cannot_reclassify_heldout_seed_as_training(tmp_path):
    plan,manifest=collection_fixture(tmp_path)
    old=plan['training_seeds'][0];new=plan['validation_seeds_excluded'].pop()
    plan['training_seeds'][0]=new
    plan['plan_sha256']=digest({k:v for k,v in plan.items() if k!='plan_sha256'})
    manifest['calibration_plan_sha256']=plan['plan_sha256']
    for case in manifest['cases']:
        if case['seed']==old:case['seed']=new
    with pytest.raises(ValueError,match='independent fixed protocol'):collect_pairs(manifest,plan,{})


def test_overlay_updates_uncertain_and_observed_zero_context_metadata(tmp_path):
    plan,graph,pairs=fixture()
    for pair,delta in zip(pairs,(-8.,4.,-1.)):
        pair['profile']['kpi:ap_power_w']['value']=pair['reference']['kpi:ap_power_w']['value']+delta
        pair['profile']['kpi:urllc_failure_fraction']['value']=0.
    for edge in graph['edges']:
        edge['statistical_selection']=dict(retained_contrasts=[],significant_contrasts=[],
            uncertain_contrasts=[],observed_zero_contrasts=[])
    graph['statistical_filter']=dict(original_estimate_count=0,active_estimate_count=0,
        significant_estimate_count=0,retained_uncertain_estimate_count=0,retained_observed_zero_estimate_count=0)
    rows,_=estimate_rows(pairs,plan,'native_1x',graph,digest(pairs))
    result,_,_=overlay(graph,rows,tmp_path/'estimates.json',provenance={})
    selections={e['target']:e['statistical_selection'] for e in result['edges']}
    assert selections['kpi:ap_power_w']['uncertain_contrasts']==['joint:ap_power_w']
    assert selections['kpi:urllc_failure_fraction']['observed_zero_contrasts']==['joint:urllc_failure_fraction']
    assert result['statistical_filter']['active_estimate_count']==2
    assert result['statistical_filter']['retained_uncertain_estimate_count']==1
    assert result['statistical_filter']['retained_observed_zero_estimate_count']==1


def test_fitted_row_is_accepted_as_one_exact_joint_profile_by_real_projector(tmp_path):
    if not shutil.which('node'):pytest.skip('Node required for production projector')
    plan,graph,pairs=fixture()
    rows,_=estimate_rows(pairs,plan,'native_1x',graph,digest(pairs))
    result,_,_=overlay(graph,rows,tmp_path/'estimates.json',provenance={})
    edge=next(e for e in result['edges'] if e['target']=='kpi:ap_power_w')
    script="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8'));
const r=c.SmoConflictRisk.project(d.edge,d.base,d.profile,{states:{}},false);
process.stdout.write(JSON.stringify(r));"""
    response=json.loads(subprocess.check_output(['node','-e',script],text=True,
        cwd=Path(__file__).resolve().parents[1],input=json.dumps(dict(edge=edge,base=BASE,profile=PROFILE))))
    assert response['drive']==-10 and response['status']=='joint_profile_reference_projection'
    assert len(response['components'])==1
    assert response['calibrations']==['joint:ap_power_w']
