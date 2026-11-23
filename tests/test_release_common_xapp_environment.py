"""Gate tests use fabricated temporary metadata, never measured gain claims.

The positive unit fixture is deliberately synthetic. Production publication
also requires the real archive, all original file seals and actual F trace.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import release_common_xapp_environment as release


def pin(value):
    return release.digest(value)


def seal(value):
    return {**value,'plan_sha256':pin(value)}


def reseal(value):
    value.pop('plan_sha256',None)
    value['plan_sha256']=pin(value)


@pytest.fixture
def metadata():
    """All scalar effects are invented to exercise validation branches only."""
    models={name:pin(name) for name in release.MODEL_PATHS}; runner=pin('synthetic-runner')
    scene='synthetic-pressure'; seeds=[101,102]
    design={'study_id':'synthetic-fixture-not-a-real-experiment','stage':'validation',
        'arms':['OFF',*release.APPS],'recipes':[{'id':scene}],
        'assignments':[{'worker_id':h,'seed':s,'recipe_ids':[scene]} for h,s in zip(['agx1','agx2'],seeds)],
        'discovery_seeds':[100],'runner_sha256':runner,'include_native_reference':False}
    report={'schema':'oran.environment-validation-analysis.v1','analysis_role':'validation',
        'comparison':'single_app_after_minus_same_origin_off_after','training_performed':False,'new_simulations':0,
        'recording_admission':{'receiver_seals_required':True,'all_published_raw_sha256_required':True},
        'pending_cases':[],'intentionally_not_run_cases':[],'environment_exclusions':None,
        'same_environment_two_seed_validation_passed':True,'all_six_targets_numerically_positive':True,
        'all_six_controls_verified_and_numerically_positive':True,'complete_cases':[],'paired_results':[],
        'singleton_summary':[],'recipe_assessments':[{'scene':scene,
            'same_environment_independent_two_seed_validation_passed':True,
            'common_complete_seeds_across_all_apps':seeds}]}
    common={'scene':scene,'slots':6440,'baseline_slot':2040,'origin_observe_slots':2000,'observe_slots':4000,
        'traffic_load':1.5,'traffic_profile':'/scope/protocol/profiles/traffic.json','traffic_profile_sha256':pin('traffic'),
        'ue_services_config':'/scope/protocol/profiles/service.json','ue_services_config_sha256':pin('service')}
    plans=[]
    for host,seed in zip(['agx1','agx2'],seeds):
        origin=pin([host,'origin']); before={'window_slots':2000,'states':{'fixture':{'value':1}}}
        states={'kpi:'+key:{'value':.1 if arm=='E' else 10.,'sample_count':10} for arm,(_,key,_,_) in release.APPS.items()}
        states['kpi:urllc_failure_fraction'].update(failure_events=10,offered_packets=100)
        flows={w:{k:pin([seed,w,k]) for k in ['exogenous_arrivals_sha256','task_births_sha256']} for w in ['before','after']}
        records={}; cases=[]
        for arm in ['OFF',*release.APPS]:
            ident=pin([host,seed,arm])[:32]
            case={**common,'id':ident,'name':f'{scene}_{seed}_{arm}','origin_key':f'{scene}_{seed}',
                'worker_id':host,'seed':seed,'arm':arm,'app_ids':[] if arm=='OFF' else ['xapp:'+release.APPS[arm][0]]}
            after={'window_slots':4000,'states':deepcopy(states)}
            if arm!='OFF':
                _,key,direction,_=release.APPS[arm]
                after['states']['kpi:'+key]['value'] += direction*(.01 if arm=='E' else 1.)
                if arm=='E':
                    after['states']['kpi:'+key].update(value=.09,failure_events=9)
            row={'case':case,'status':'complete','stage':'validation','worker_id':host,
                'source_sha256':release.SOURCE,'source_archive_sha256':release.ARCHIVE,'runner_sha256':runner,
                'frozen_models':models,'origin_id':origin,'actual_system_hostname':host,
                'checkpoint':{'path':'synthetic-only','sha256':pin(ident)},'flows':deepcopy(flows),
                'windows':{'before':deepcopy(before),'after':after}}
            if arm!='OFF':
                row['origin_guard']={**{k:True for k in release.GUARD_FLAGS},'origin_id':origin}
                evidence={'job_id':ident,'checkpoint_sha256':row['checkpoint']['sha256'],
                    'source_sha256':release.SOURCE,'source_archive_sha256':release.ARCHIVE,
                    'runner_sha256':runner,'worker_id':host,'status':'ready',
                    'activation_verified':True,'frozen_verified':True,'consumer_verified':True,'actual_changed':True,
                    'consumer_fields_unresolved':0,'consumer_mismatches':[],'trace_consumer_mismatches':[],
                    'own_applied_receipt_count':2,'changed_receipt_count':1,'decision_count':1,
                    'before_baseline_binding_slots':[],'unexpected_active_apps':[],
                    'first_applied_receipt_slot':2100}
                row['control_provenance']={'qualified':True,'evidence':evidence}
                if arm=='F':
                    row['post_first_apply_ue4_quota']={'status':'observed','target_ue_id':'ue_4',
                        'start_basis':'first_actual_own_applied_consumer_action_even_if_noop',
                        'first_applied_slot':2100,'after_end_slot':6040,'post_apply_exposure_slots':3941,
                        'mean_compliant':True,'maximum_compliant':True,'mean_gflops':.5,'maximum_gflops':.5,'budget_gflops':.75}
            records[arm]=row;cases.append(case);report['complete_cases'].append(row)
        off=records['OFF']
        for arm in release.APPS:
            on=records[arm]; _,key,direction,_=release.APPS[arm]
            improvement=direction*(on['windows']['after']['states']['kpi:'+key]['value']-states['kpi:'+key]['value'])
            pair={'stage':'validation','scene':scene,'seed':seed,'worker_id':host,'origin_key':on['case']['origin_key'],
                'arm':arm,'app_id':release.APPS[arm][0],'target_kpi':key,'job_id':on['case']['id'],'off_job_id':off['case']['id'],
                'before':on['windows']['before'],'after':on['windows']['after'],'off_after':off['windows']['after'],
                'before_comparison':{'comparable_under_origin_protocol':True},
                'external_input_comparison':{k:True for k in release.EXOG_FLAGS},
                'target':{'status':'improved','target_improvement':improvement,
                    'improvement_percent':100*improvement/states['kpi:'+key]['value'],
                    'numerical_tolerance':0 if arm=='E' else 1e-9},
                'control_provenance':on['control_provenance']}
            if arm=='F':pair['post_first_apply_ue4_quota']=on['post_first_apply_ue4_quota']
            report['paired_results'].append(pair)
        plans.append(seal({'stage':'validation','worker_id':host,'study_id':design['study_id'],'design_sha256':pin(design),
            'source_sha256':release.SOURCE,'source_archive_sha256':release.ARCHIVE,'runner_sha256':runner,
            'frozen_models':models,'cases':cases}))
    for arm in release.APPS:
        report['singleton_summary'].append({'scene':scene,'arm':arm,'planned_pairs':2,'paired':2,
            'all_planned_pairs_numerically_positive':True,'control_provenance_qualified_all_pairs':True,
            'missing_job_ids':[],'intentionally_not_run_job_ids':[],'per_seed':[{'seed':s} for s in seeds],
            'released_ue4_quota_satisfied_all_pairs':True if arm=='F' else None})
    protocol=seal({'stage':'validation','design_sha256':pin(design),'recipes':{scene:{k:common[k] for k in
        ['traffic_load','traffic_profile','traffic_profile_sha256','ue_services_config','ue_services_config_sha256']}},
        'discovery_seeds':[100],'known_nonfresh_seeds':[99],'frozen_models':models,
        'source_sha256':release.SOURCE,'source_archive_sha256':release.ARCHIVE,'runner_sha256':runner,
        'physical_run_count':14,'observe_slots':4000,'baseline_slot':2040,'origin_observe_slots':2000,'tail_slots':400,'total_slots':6440})
    return report,plans,design,protocol


def validate(metadata):
    return release.validate_report(*metadata)


def declared_criteria(metadata):
    _,_,design,protocol=metadata
    result={'schema':'oran.common-xapp-validation-criteria.v1','common_recipe':design['recipes'][0],
        'required_apps':list(release.APPS),'fresh_seeds':[a['seed']for a in design['assignments']],
        'all_six_targets_positive_in_each_seed':True,'real_frozen_changed_controls_required':True,
        'same_own_OFF_origin_and_before_after_exogenous_hashes_required':True,
        'training_or_manual_actions':False,'default_or_KG_changed':False,
        'C_minimum_improvement_percent_each_seed':1.,'F_quota':{'ue4_gflops_maximum':.75},
        'D_cohort_diagnostic':'Synthetic fixture: report membership and common-task means without replacing the KPI.',
        'measurement':{**{k:protocol[k]for k in ['baseline_slot','origin_observe_slots','observe_slots','tail_slots','total_slots']},'slot_ms':.5}}
    if design.get('runner_version')==4:
        result.update(geometry_change={'ue_route_overrides':deepcopy(release.ROUTE_OVERRIDE),
            'other_five_UEs_unchanged':True,'all_UE_speed_mps':1.5},
            OFF_admission_gate={'explicit_ON_ADMISSION_ALLOWED_required':True})
    return result


def test_synthetic_two_seed_matrix_exercises_positive_gate_without_publication(metadata):
    result=validate(metadata)
    assert result['seeds']==[101,102]
    assert len(result['paired_targets'])==12
    assert {x['target_kpi'] for x in result['paired_targets']}=={x[1]for x in release.APPS.values()}


def test_synthetic_v4_positive_matrix_requires_own_OFF_base_and_explicit_gate(metadata):
    value=validate(covered_metadata(metadata))
    assert value['recipe']['base_config_sha256']==pin('synthetic-covered-base')
    assert len(value['paired_targets'])==12
    release.validate_criteria(declared_criteria(metadata),value,metadata[2],metadata[3])


def test_v4_recipe_base_pin_cannot_differ_from_cases(metadata):
    covered_metadata(metadata)
    protocol=metadata[3]
    protocol['recipes'][metadata[2]['recipes'][0]['id']]['base_config_sha256']=pin('other-base')
    reseal(protocol)
    with pytest.raises(release.ReleaseError,match='protocol_recipe_pin_mismatch:base_config_sha256'):
        validate(metadata)


def test_v4_ON_cannot_borrow_a_different_base_from_its_own_OFF(metadata):
    covered_metadata(metadata)
    metadata[0]['complete_cases'][1]['case']['base_config_sha256']=pin('other-base')
    reseal(metadata[1][0])
    with pytest.raises(release.ReleaseError,match='cross_recipe_physical_inputs'):
        validate(metadata)


def test_v3_cannot_export_an_arbitrary_external_base(metadata):
    for row in metadata[0]['complete_cases']:
        row['case'].update(base_config='/scope/protocol/profiles/unreviewed.json',base_config_sha256=pin('bad-base'))
    for plan in metadata[1]:reseal(plan)
    with pytest.raises(release.ReleaseError,match='external_base_requires_reviewed_v4'):
        validate(metadata)


@pytest.mark.parametrize('change',[
    lambda c:c['radio'].update(slot_duration_ms=1),
    lambda c:c['radio'].update(num_rx_ant=True),
    lambda c:c.update(ue_speed_mps=0),
    lambda c:c['rt_sys']['paper_interfaces'].update(connection_events=True),
    lambda c:c['ue_route_overrides'].update(ue_4=dict(route_index=4,start_fraction=.4)),
    lambda c:c['ue_route_overrides']['ue_0'].update(start_fraction=.6),
    lambda c:c['ue_route_overrides']['ue_0'].update(route_index=1),
    lambda c:c['ue_route_overrides']['ue_0'].update(new_waypoints=[]),
    lambda c:c.update(unapproved_new_setting=True),
])
def test_v4_base_semantic_whitelist_rejects_all_other_changes(change):
    frozen=(release.ROOT/release.BASE_CONFIG).read_bytes()
    base={**json.loads(frozen),'ue_route_overrides':deepcopy(release.ROUTE_OVERRIDE)}
    change(base)
    with pytest.raises(release.ReleaseError,match='only_UE0_road0_phase065'):
        release.validate_geometry_base(release.encoded(base),frozen)


def test_known_producer_metadata_pins_are_not_additional_recipes(metadata):
    _,plans,_,protocol=metadata
    for name,path in release.RECIPE_METADATA_PINS.items():
        protocol['recipes'][name]=pin(path)
        for plan in plans:plan.setdefault('frozen_physical_files',{})[path]=pin(path)
    reseal(protocol)
    for plan in plans:reseal(plan)
    validate(metadata)
    protocol['recipes']['unknown_sha256']=pin('unexpected');reseal(protocol)
    with pytest.raises(release.ReleaseError,match='unknown_protocol_recipe_metadata'):validate(metadata)
    protocol['recipes'].pop('unknown_sha256');protocol['recipes']['second-real-recipe']={};reseal(protocol)
    with pytest.raises(release.ReleaseError,match='one_common_recipe'):validate(metadata)


def test_protocol_metadata_pins_must_match_both_worker_source_pins(metadata):
    _,plans,_,protocol=metadata
    for name,path in release.RECIPE_METADATA_PINS.items():
        protocol['recipes'][name]=pin(path)
        for plan in plans:plan.setdefault('frozen_physical_files',{})[path]=pin(path)
    plans[1]['frozen_physical_files'][release.AP_CONFIG]=pin('wrong AP')
    reseal(protocol)
    for plan in plans:reseal(plan)
    with pytest.raises(release.ReleaseError,match='protocol_recipe_metadata_pin_mismatch'):validate(metadata)


def long_metadata(metadata):
    report,plans,design,protocol=metadata
    design.update(observe_slots=20000)
    design['recipes'][0]['initial_ue4_offload_fraction']=.8
    protocol.update(observe_slots=20000,total_slots=22440,design_sha256=pin(design))
    for row in report['complete_cases']:
        row['case'].update(slots=22440,observe_slots=20000)
        row['windows']['after']['window_slots']=20000
        if row['case']['arm']=='F':
            row['post_first_apply_ue4_quota'].update(after_end_slot=22040,post_apply_exposure_slots=19941)
    for plan in plans:plan['design_sha256']=pin(design);reseal(plan)
    reseal(protocol)
    return metadata


def covered_metadata(metadata):
    """Synthetic positive v4 metadata only; no measured outcomes are invented."""
    report,plans,design,protocol=long_metadata(metadata)
    design['runner_version']=4
    design['runner_sha256']=release.V4_RUNNER_SHA256
    base='/scope/protocol/profiles/'+design['study_id']+'_ue0_road0_phase065_base.json'
    for plan in plans:
        plan.update(schema='oran.environment-validation-worker-plan.v4',source_root='/scope/source',
            runner_path='/scope/runner/worker-v4.py',runner_sha256=release.V4_RUNNER_SHA256,
            require_on_admission=True,geometry_change=deepcopy(release.GEOMETRY_CHANGE),design_sha256=pin(design))
        for case in plan['cases']:case.update(base_config=base,base_config_sha256=pin('synthetic-covered-base'))
        reseal(plan)
    for row in report['complete_cases']:
        row['runner_sha256']=release.V4_RUNNER_SHA256
        if row['case']['arm']!='OFF':row['control_provenance']['evidence']['runner_sha256']=release.V4_RUNNER_SHA256
    protocol.update(runner_version=4,runner_sha256=release.V4_RUNNER_SHA256,require_on_admission=True,
        ON_admission_marker='ON_ADMISSION_ALLOWED',geometry_change=deepcopy(release.GEOMETRY_CHANGE),design_sha256=pin(design))
    protocol['recipes'][design['recipes'][0]['id']].update(base_config=base,base_config_sha256=pin('synthetic-covered-base'))
    reseal(protocol)
    return metadata


def test_long_window_routing80_matrix_uses_dynamic_timing_and_same_gate(metadata):
    validate(long_metadata(metadata))
    pair=next(p for p in metadata[0]['paired_results']if p['arm']=='F')
    assert pair['post_first_apply_ue4_quota']['after_end_slot']==22040


def test_declared_C_visibility_threshold_applies_to_each_seed_without_rescoring(metadata):
    gate=validate(metadata);criteria=declared_criteria(metadata)
    release.validate_criteria(criteria,gate,metadata[2],metadata[3])
    next(r for r in gate['paired_targets']if r['arm']=='C')['effect']['improvement_percent']=.999
    with pytest.raises(release.ReleaseError,match='C_minimum_not_met_each_seed'):
        release.validate_criteria(criteria,gate,metadata[2],metadata[3])


@pytest.mark.parametrize('change,reason',[
    (lambda c:c['measurement'].update(observe_slots=20000),'measurement_window'),
    (lambda c:c['fresh_seeds'].append(999),'recipe_apps_seeds'),
    (lambda c:c.update(C_minimum_improvement_percent_each_seed=0),'C_one_percent'),
    (lambda c:c['F_quota'].update(ue4_gflops_maximum=1),'F_quota'),
    (lambda c:c.update(real_frozen_changed_controls_required=False),'required_rule'),
])
def test_criteria_bytes_cannot_describe_another_rule_or_window(metadata,change,reason):
    criteria=declared_criteria(metadata);change(criteria)
    with pytest.raises(release.ReleaseError,match=reason):
        release.validate_criteria(criteria,validate(metadata),metadata[2],metadata[3])


@pytest.mark.parametrize('change,reason',[
    (lambda r:r.update(analysis_role='discovery'),'final_validation_role'),
    (lambda r:r['pending_cases'].append({'synthetic':'not complete'}),'complete_unexcluded'),
    (lambda r:r['intentionally_not_run_cases'].append({'synthetic':'not run'}),'complete_unexcluded'),
    (lambda r:r['complete_cases'].pop(),'partial_or_duplicate'),
    (lambda r:r['paired_results'].pop(),'complete_six_app_seed_matrix'),
    (lambda r:r['paired_results'][0].update(off_job_id=r['paired_results'][-1]['off_job_id']),'own_same_host_OFF'),
    (lambda r:r['paired_results'][0]['external_input_comparison'].update(after_task_births_equal=False),'external_input'),
    (lambda r:r['complete_cases'][1]['flows']['after'].update(task_births_sha256=pin('different')),'external_input_hash'),
    (lambda r:r['complete_cases'][1]['origin_guard'].update(configuration_equal=False),'native_origin_guard'),
    (lambda r:r['paired_results'][0]['control_provenance'].update(qualified=False),'qualified_real_control'),
    (lambda r:r['paired_results'][0]['control_provenance']['evidence'].update(actual_changed=False),'executed_changed_control'),
    (lambda r:r['paired_results'][0]['control_provenance']['evidence'].update(consumer_mismatches=['bad']),'unresolved_control'),
    (lambda r:r['paired_results'][5]['post_first_apply_ue4_quota'].update(maximum_gflops=1.),'quota_failed'),
    (lambda r:r['paired_results'][5]['post_first_apply_ue4_quota'].update(first_applied_slot=2200),'post_ACK_window'),
])
def test_false_success_metadata_cannot_bypass_individual_gates(metadata,change,reason):
    change(metadata[0])
    with pytest.raises(release.ReleaseError,match=reason):validate(metadata)


@pytest.mark.parametrize('arm,delta',[('A',0),('A',1),('C',0),('C',-1),('E',0),('E',.01)])
def test_unchanged_and_negative_actual_target_rejected_despite_positive_labels(metadata,arm,delta):
    pair=next(x for x in metadata[0]['paired_results'] if x['arm']==arm)
    key='kpi:'+release.APPS[arm][1];base=pair['off_after']['states'][key]['value']
    pair['after']['states'][key]['value']=base+delta
    if arm=='E':pair['after']['states'][key]['failure_events']=round((base+delta)*100)
    with pytest.raises(release.ReleaseError,match='zero_or_negative_target_effect'):validate(metadata)


def test_complete_per_seed_matrix_cannot_pool_different_recipe_bytes(metadata):
    report,plans,_,_=metadata
    row=report['complete_cases'][-1]; row['case']['traffic_profile_sha256']=pin('different-recipe')
    reseal(plans[1])
    with pytest.raises(release.ReleaseError,match='cross_recipe_physical_inputs'):validate(metadata)


def test_sub_counter_resolution_goodput_gain_cannot_be_relabelled_success(metadata):
    pair=next(p for p in metadata[0]['paired_results'] if p['arm']=='C')
    key='kpi:network_goodput_mbps'
    for state in (pair['after']['states'][key],pair['off_after']['states'][key]):state['resolution']=4e-6
    pair['after']['states'][key]['value']=pair['off_after']['states'][key]['value']+1e-6
    pair['target']['target_improvement']=1e-6
    with pytest.raises(release.ReleaseError,match='zero_or_negative_target_effect:C'):validate(metadata)


@pytest.mark.parametrize('value',[0,438,8000,0.,438.,8000.,float(2**53-1),2**53])
def test_exact_integer_counter_representations_are_accepted_without_rounding(value):
    result=release.integral_counter(value)
    assert type(result) is int and result==value


@pytest.mark.parametrize('value',[438.1,float('nan'),float('inf'),True,False,None,-1,-1.,float(2**53),'438'])
def test_fractional_unsafe_or_unresolved_counts_are_rejected(value):
    with pytest.raises(release.ReleaseError,match='URLLC_integer_ratio_evidence_required'):
        release.integral_counter(value)


@pytest.mark.parametrize('field,value',[('failure_events',9.5),('failure_events',float('nan')),
    ('failure_events',True),('offered_packets',float(2**53)),('offered_packets',None),('offered_packets',False)])
def test_positive_pair_rejects_invalid_ratio_counter_metadata(metadata,field,value):
    pair=next(p for p in metadata[0]['paired_results']if p['arm']=='E')
    pair['after']['states']['kpi:urllc_failure_fraction'][field]=value
    with pytest.raises(release.ReleaseError,match='URLLC_integer_ratio_evidence_required'):validate(metadata)


@pytest.mark.parametrize('field',['failure_events','offered_packets'])
def test_missing_ratio_counter_is_not_assumed_zero(metadata,field):
    pair=next(p for p in metadata[0]['paired_results']if p['arm']=='E')
    pair['after']['states']['kpi:urllc_failure_fraction'].pop(field)
    with pytest.raises(release.ReleaseError,match='URLLC_integer_ratio_evidence_required'):validate(metadata)


def test_integer_valued_float_counters_preserve_the_same_full_gate_result(metadata):
    original=validate(metadata)
    for pair in metadata[0]['paired_results']:
        if pair['arm']!='E':continue
        for window in ['after','off_after']:
            state=pair[window]['states']['kpi:urllc_failure_fraction']
            for field in ['failure_events','offered_packets']:state[field]=float(state[field])
    assert validate(metadata)==original


def test_declared_discovery_seed_cannot_be_reused_as_validation(metadata):
    _,plans,design,protocol=metadata
    design['discovery_seeds'].append(101);protocol['discovery_seeds']=design['discovery_seeds']
    protocol['design_sha256']=pin(design);reseal(protocol)
    for plan in plans:plan['design_sha256']=pin(design);reseal(plan)
    with pytest.raises(release.ReleaseError,match='fresh_seed_history'):validate(metadata)


def test_single_app_report_not_enough_even_if_headline_flags_claim_success(metadata):
    report,plans,design,protocol=metadata
    design['arms']=['OFF','A'];protocol['design_sha256']=pin(design);reseal(protocol)
    for plan in plans:plan['design_sha256']=pin(design);reseal(plan)
    with pytest.raises(release.ReleaseError,match='exactly_six_singleton'):validate(metadata)


def test_plan_digest_source_model_and_runner_checks(metadata):
    plans=metadata[1]
    plans[0]['runner_sha256']=pin('different-runner')
    with pytest.raises(release.ReleaseError,match='digest_mismatch'):validate(metadata)
    reseal(plans[0])
    with pytest.raises(release.ReleaseError,match='runner_pin_mismatch'):validate(metadata)


def test_publisher_is_create_only_idempotent_and_rejects_changed_or_extra_files(tmp_path):
    output=tmp_path/'synthetic-fixture-bundle'
    files={'configs/traffic.json':b'{"fixture_only":true}\n','environment.json':b'{"test_only":true}\n'}
    assert release.publish_bundle(files,output) is True
    assert release.publish_bundle(files,output) is False
    original={p.relative_to(output):p.read_bytes()for p in output.rglob('*')if p.is_file()}
    with pytest.raises(release.ReleaseError,match='overwrite'):
        release.publish_bundle({**files,'environment.json':b'changed'},output)
    assert {p.relative_to(output):p.read_bytes()for p in output.rglob('*')if p.is_file()}==original
    (output/'unrelated.txt').write_text('do not touch')
    with pytest.raises(release.ReleaseError,match='overwrite'):release.publish_bundle(files,output)
    assert (output/'unrelated.txt').read_text()=='do not touch'


def test_publisher_rejects_escape_symlink_and_runtime_default_destination(tmp_path):
    for name in ['../outside','/tmp/absolute-fixture']:
        with pytest.raises(release.ReleaseError,match='unsafe_release'):release.publish_bundle({name:b'x'},tmp_path/'no')
    target=tmp_path/'target';target.mkdir();link=tmp_path/'link';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(release.ReleaseError,match='symlink'):release.publish_bundle({'x':b'x'},link)
    with pytest.raises(release.ReleaseError,match='dedicated_outputs'):release.publish_bundle({'x':b'x'},release.ROOT/'configs')
    assert list(target.iterdir())==[]


def test_real_F_trace_is_checked_after_first_ACK_including_first_slot(tmp_path):
    path=tmp_path/'rt_sys_trace.jsonl'
    def write(values):
        path.write_text(''.join(json.dumps({'slot_id':n,'ue_services':{'ue_4':{'ue_service_policy':{'offload_fraction':v}}}})+'\n'
                                for n,v in values))
    row={'run_path':str(tmp_path),'post_first_apply_ue4_quota':{'first_applied_slot':3,'after_end_slot':5,
        'mean_gflops':.5,'maximum_gflops':.5,'budget_gflops':.75}}
    config={'configs':{'phy':{'rt_sys':{'ue_services':{'mec':{'tasks_by_ue':{'ue_4':{'interval_ms':50,'compute_gflop':.05}}}}}},
                      'ric':{'xapps':[{'implementation_id':release.APPS['F'][0],'parameters':{'budget_gflops':.75}}]}}}
    write([(1,1),(2,1),(3,.5),(4,.5),(5,.5)])
    release.verify_F_quota(row,config)
    write([(1,.5),(2,.5),(3,1),(4,.5),(5,.5)])
    with pytest.raises(release.ReleaseError,match='actual_post_ACK_quota_failed'):release.verify_F_quota(row,config)
    write([(3,.5),(5,.5)])
    with pytest.raises(release.ReleaseError,match='trace_gap'):release.verify_F_quota(row,config)


def temporary_evidence(metadata,tmp_path,monkeypatch):
    """Fabricate file seals for packaging tests; mock ONLY archive admission.

    This does not claim a network was simulated. The production CLI has no
    option to bypass its required exact CBE source archive.
    """
    report,plans,design,protocol=metadata;root=tmp_path/'protocol';root.mkdir()
    profiles=root/'profiles';profiles.mkdir()
    traffic=b'{"fixture_only":"traffic"}\n';service=b'{"fixture_only":"service"}\n'
    (profiles/'traffic.json').write_bytes(traffic);(profiles/'service.json').write_bytes(service)
    archive={n:('synthetic model '+n).encode()for n in release.MODEL_PATHS}
    v4=design.get('runner_version')==4
    frozen_base=(release.ROOT/release.BASE_CONFIG).read_bytes()if v4 else b'{"fixture_only":"base"}\n'
    archive.update({release.BASE_CONFIG:frozen_base,release.AP_CONFIG:b'{"fixture_only":"AP"}\n'})
    monkeypatch.setattr(release,'archive_files',lambda path:archive)
    if v4:
        runner=release.ROOT/'experiments/environment_validation_worker_v4.py'
        base_name=design['study_id']+'_ue0_road0_phase065_base.json'
        raw=release.encoded({**json.loads(frozen_base),'ue_route_overrides':deepcopy(release.ROUTE_OVERRIDE)})
        (profiles/base_name).write_bytes(raw)
        protocol['recipes'][design['recipes'][0]['id']].update(base_config_sha256=release.sha_bytes(raw))
        for plan in plans:
            for case in plan['cases']:case['base_config_sha256']=release.sha_bytes(raw)
    else:
        runner=tmp_path/'fixture-worker.py';runner.write_bytes(b'# Test fixture, never executed\n')
    runner_pin=release.file_sha(runner);design['runner_sha256']=runner_pin
    models=protocol['frozen_models']
    for n in models:models[n]=release.sha_bytes(archive[n])
    recipe=protocol['recipes'][design['recipes'][0]['id']]
    recipe.update(traffic_profile_sha256=release.sha_bytes(traffic),ue_services_config_sha256=release.sha_bytes(service))
    proof_by_host={}
    for assignment in design['assignments']:
        body={'schema':'oran.evaluation-origin.v1','seed':assignment['seed'],'baseline_slot':2040,
              'source_sha256':release.SOURCE,'source_archive_sha256':release.ARCHIVE,
              'observe_slots':2000,'before_frame':{'slot':2040},'after_kpis_used':False}
        proof_by_host[assignment['worker_id']]={**body,'origin_id':pin(body)}
    for plan in plans:
        plan.update(runner_sha256=runner_pin,design_sha256=pin(design),source_root='/scope/source',
                    frozen_physical_files={n:release.sha_bytes(archive[n])for n in [release.BASE_CONFIG,release.AP_CONFIG]})
        for case in plan['cases']:
            case.update(traffic_profile_sha256=release.sha_bytes(traffic),ue_services_config_sha256=release.sha_bytes(service))
        reseal(plan)
    for row in report['complete_cases']:
        case=row['case'];host=row['worker_id'];ident=case['id'];proof=proof_by_host[host]
        row.update(runner_sha256=runner_pin,origin_id=proof['origin_id'])
        job=tmp_path/'workers'/host/'jobs'/ident;run=job/'data/run';run.mkdir(parents=True)
        row['run_path']=str(run)
        profile={'interval_ms':50,'compute_gflop':.05}
        config={'source_sha256':release.SOURCE,
            'manifest':{'rng_seed':case['seed'],'total_slots':case['slots'],'traffic_profile':{'sha256':release.sha_bytes(traffic)}},
            'ue_services_profile':{'sha256':release.sha_bytes(service)},
            'configs':{'phy':{'rt_sys':{'ue_services':{'mec':{'tasks_by_ue':{'ue_4':profile}}}}},
                      'ric':{'xapps':[{'implementation_id':release.APPS['F'][0],'parameters':{'budget_gflops':.75}}]}}}
        if v4:
            config['manifest']['ue_route_overrides']=deepcopy(release.ROUTE_OVERRIDE)
        (run/'config_snapshot.json').write_bytes(release.encoded(config))
        (run/'run_status.json').write_bytes(release.encoded({'completed':True,'fixture_only':True}))
        timeline=[{'at_ms':1020.,'operation':'set','apps':[]if case['arm']=='OFF'else[release.APPS[case['arm']][0]],'cell':'*'}]
        (job/'data/timeline.json').write_bytes(release.encoded(timeline))
        if v4:
            plan=next(p for p in plans if p['worker_id']==host)
            (job/'data/request.json').write_bytes(release.encoded(dict(schema='oran.environment-run-request.v4',
                case=case,plan_sha256=plan['plan_sha256'],base_config_sha256=case['base_config_sha256'],
                geometry_change=deepcopy(release.GEOMETRY_CHANGE),require_on_admission=True,scheduling_only_change=False)))
            (job/'data/command.json').write_bytes(release.encoded({'argv':['python','frozen-case.py',
                '--config',release.BASE_CONFIG,'--config',case['base_config']]}))
        if case['arm']=='F':
            q=row['post_first_apply_ue4_quota']
            frames=[{'slot_id':n,'ue_services':{'ue_4':{'ue_service_policy':{'offload_fraction':.5}}}}
                    for n in range(q['first_applied_slot'],q['after_end_slot']+1)]
        else:frames=[{'slot_id':case['slots'],'fixture_only':True}]
        (run/'rt_sys_trace.jsonl').write_text(''.join(json.dumps(x)+'\n'for x in frames))
        if case['arm']!='OFF':
            row['origin_guard']['origin_id']=proof['origin_id']
            (run/'evaluation_origin_verified.json').write_bytes(release.encoded(row['origin_guard']))
            (job/'data/expected_origin.json').write_bytes(release.encoded(proof))
        files={str(p.relative_to(job)):release.file_sha(p)for p in job.rglob('*')if p.is_file()}
        row['files_sha256']=files
        plan=next(p for p in plans if p['worker_id']==host)
        checkpoint={'completed':True,'case':case,'plan_sha256':plan['plan_sha256'],'models_unchanged':True,
                    'training_performed':False,'post_deployment_kpis_used_for_model_training':False,
                    'files_sha256':files,**{k:row[k]for k in ['source_sha256','source_archive_sha256','runner_sha256',
                         'frozen_models','origin_id','actual_system_hostname']}}
        if v4:
            checkpoint.update(schema='oran.environment-validation-run.v4',base_config_sha256=case['base_config_sha256'],
                geometry_change=deepcopy(release.GEOMETRY_CHANGE),require_on_admission=True,scheduling_only_change=False)
        cp=tmp_path/'workers'/host/'checkpoints'/(ident+'.json');cp.parent.mkdir(exist_ok=True);cp.write_bytes(release.encoded(checkpoint))
        row['checkpoint']={'path':str(cp),'sha256':release.file_sha(cp)}
        row['receiver_admission']={'required':True,'checkpoint_sha256':row['checkpoint']['sha256'],
                                   'verification':{'seal_sha256':row['checkpoint']['sha256']}}
        if case['arm']!='OFF':
            row['control_provenance']['evidence'].update(runner_sha256=runner_pin,checkpoint_sha256=row['checkpoint']['sha256'])
    protocol.update(runner_sha256=runner_pin,design_sha256=pin(design));reseal(protocol)
    (root/'design.json').write_bytes(release.encoded(design));(root/'protocol.json').write_bytes(release.encoded(protocol))
    (root/'validation').mkdir();refs=[]
    for plan in plans:
        path=root/'validation'/(plan['worker_id']+'.json');path.write_bytes(release.encoded(plan));refs.append({'path':str(path),'sha256':release.file_sha(path)})
    report['plan_inputs']=refs
    (root/'manifest.json').write_bytes(release.encoded({'files_sha256':{str(p.relative_to(root)):release.file_sha(p)
                                                       for p in root.rglob('*')if p.is_file()}}))
    audit=tmp_path/'fixture-activation.json';audit.write_bytes(release.encoded({'schema':'oran.environment-activation-audit.v1',
        'entries':[r['control_provenance']['evidence']for r in report['complete_cases']if r['case']['arm']in release.APPS]}))
    report['activation_audit']={'path':str(audit),'sha256':release.file_sha(audit)}
    result=tmp_path/'analysis/results.json';result.parent.mkdir();result.write_bytes(release.encoded(report))
    (result.parent/'analysis-manifest.json').write_bytes(release.encoded({'plans':refs,'complete':14,'planned':14,'pending':0,
        'paired_single_apps':12,'intentionally_not_run':0,'files':{'results.json':{'sha256':release.file_sha(result)}}}))
    criteria=tmp_path/'fixture-criteria.json';criteria.write_bytes(release.encoded(declared_criteria(metadata)))
    reader=tmp_path/'fixture-D-reader.py';reader.write_text('# Synthetic metadata test reader; never executed\n')
    by_id={r['case']['id']:r for r in report['complete_cases']};diagnostics=[]
    for pair in report['paired_results']:
        if pair['arm']!='D':continue
        evidence={}
        for label,jid in [('OFF',pair['off_job_id']),('D',pair['job_id'])]:
            row=by_id[jid];run=Path(row['run_path']);evidence[label]={'checkpoint':row['checkpoint'],
                'trace':{'path':str(run/'rt_sys_trace.jsonl'),'sha256':row['files_sha256']['data/run/rt_sys_trace.jsonl']},
                'config':{'path':str(run/'config_snapshot.json'),'sha256':row['files_sha256']['data/run/config_snapshot.json']}}
        cert={'schema':'oran.environment-MEC-completed-cohort-diagnostic.v2','scene':pair['scene'],'seed':pair['seed'],
            'worker_id':pair['worker_id'],'source_results':{'path':str(result),'sha256':release.file_sha(result)},
            'code':{'path':str(reader),'sha256':release.file_sha(reader)},'evidence':evidence,
            'exact_primary_birth_exposure_used':True,'terminal_followup_beyond_primary_end_used':False,
            'primary_metric_or_analysis_unchanged':True,'canonical_target':pair['target'],
            **{k:True for k in ['eligible_task_keys_equal','eligible_task_clock_route_equal','completed_task_keys_equal','failed_task_keys_equal']},
            'OFF_completed_count':10,'D_completed_count':10,'OFF_failed':{},'D_failed':{},'OFF_pending':[],'D_pending':[],
            'OFF_only_completed':[],'D_only_completed':[],'common_completed_count':10,
            'common_completed_OFF_mean_ms':10.,'common_completed_D_mean_ms':9.,'common_completed_mean_change_ms':-1.}
        path=tmp_path/('fixture-D-'+str(pair['seed'])+'.json');path.write_bytes(release.encoded(cert))
        diagnostics.append((path,release.file_sha(path)))
    return result,root,tmp_path/'archive-admission-mocked-only-in-test.tgz',runner,criteria,release.file_sha(criteria),diagnostics


def test_full_temporary_packaging_path_preserves_exact_bytes_without_publication(metadata,tmp_path,monkeypatch):
    args=temporary_evidence(metadata,tmp_path,monkeypatch)
    files=release.prepare_release(*args)
    assert files['configs/traffic.json']==(args[1]/'profiles/traffic.json').read_bytes()
    assert files['configs/ue-services.json']==(args[1]/'profiles/service.json').read_bytes()
    value=json.loads(files['environment.json'])
    assert len(value['six_target_KPIs'])==6 and len(value['paired_targets'])==12
    assert value['automatic_runtime_default_UI_Git_or_global_changes'] is False
    assert 'No borrowed' in value['labels']['KG']
    assert value['validation_criteria']['sha256']==release.file_sha(args[4])
    assert value['D_identity_diagnostics']['complete'] is True
    assert files['evidence/criteria.json']==args[4].read_bytes()
    assert {n for n in files if n.startswith('timelines/')}=={'timelines/'+a+'.json'for a in ['OFF',*release.APPS]}
    for arm in ['OFF',*release.APPS]:
        row=next(r for r in metadata[0]['complete_cases']if r['case']['arm']==arm)
        assert files['timelines/'+arm+'.json']==(Path(row['run_path']).parents[1]/'data/timeline.json').read_bytes()
    reproduction=json.loads(files['reproducibility.json'])
    argv=release.reproduction_arguments(reproduction,tmp_path/'artifact','D',123)
    assert argv[argv.index('--xapps-start')+1]==''
    assert argv[argv.index('--slots')+1]=='6440'
    assert argv[argv.index('--xapp-toggle-timeline')+1].endswith('/timelines/D.json')
    assert argv[argv.index('--live-state')+1].startswith('outputs/common_xapp_replay/')
    assert json.loads(files['timelines/D.json'])[0]['at_ms']==1020.
    destination=tmp_path/'only-synthetic-test-publication'
    assert release.publish_bundle(files,destination) is True
    assert release.publish_bundle(release.prepare_release(*args),destination) is False
    assert not (tmp_path/'published').exists()


def test_full_v4_temporary_packaging_copies_exact_external_and_frozen_base(metadata,tmp_path,monkeypatch):
    args=temporary_evidence(covered_metadata(metadata),tmp_path,monkeypatch)
    files=release.prepare_release(*args)
    value=json.loads(files['environment.json'])
    base_name=metadata[2]['study_id']+'_ue0_road0_phase065_base.json'
    assert files['configs/base.json']==(args[1]/'profiles'/base_name).read_bytes()
    assert files['configs/frozen-base.json']==(release.ROOT/release.BASE_CONFIG).read_bytes()
    assert value['geometry_change']==release.GEOMETRY_CHANGE
    assert value['base_config_evidence']['external_base_sha256']==release.sha_bytes(files['configs/base.json'])
    assert json.loads(files['evidence/base-config-difference.json'])==release.validate_geometry_base(
        files['configs/base.json'],files['configs/frozen-base.json'])
    assert value['observed_window']['observe_slots']==20000
    assert 'configs/base.json' in json.loads(files['reproducibility.json'])['argv_template'][
        json.loads(files['reproducibility.json'])['argv_template'].index('--config')+1]
    destination=tmp_path/'synthetic-unit-artifact-only'
    assert release.publish_bundle(files,destination)
    assert not release.publish_bundle(release.prepare_release(*args),destination)


@pytest.mark.parametrize('change',[
    lambda request,command,config:request.update(base_config_sha256=pin('wrong-base')),
    lambda request,command,config:request.update(require_on_admission=False),
    lambda request,command,config:command['argv'].extend(['--config','/different.json']),
    lambda request,command,config:config['manifest']['ue_route_overrides']['ue_0'].update(start_fraction=.6),
])
def test_v4_recording_cannot_hide_another_command_request_or_actual_route(metadata,tmp_path,monkeypatch,change):
    temporary_evidence(covered_metadata(metadata),tmp_path,monkeypatch)
    row=metadata[0]['complete_cases'][0];plan=metadata[1][0];run=Path(row['run_path']);job=run.parents[1]
    paths=[job/'data/request.json',job/'data/command.json',run/'config_snapshot.json']
    values=[release.read(p)for p in paths];change(*values)
    for p,v in zip(paths,values):p.write_bytes(release.encoded(v))
    checkpoint=release.read(row['checkpoint']['path'])
    files={str(p.relative_to(job)):release.file_sha(p)for p in job.rglob('*')if p.is_file()}
    checkpoint['files_sha256']=row['files_sha256']=files
    Path(row['checkpoint']['path']).write_bytes(release.encoded(checkpoint))
    pin_value=release.file_sha(row['checkpoint']['path']);row['checkpoint']['sha256']=pin_value
    row['receiver_admission'].update(checkpoint_sha256=pin_value,verification={'seal_sha256':pin_value})
    with pytest.raises(release.ReleaseError,match='recorded_v4_base_request_command_or_route'):
        release.verify_recording(row,plan)


def test_real_covered_protocol_schema_and_base_bytes_are_compatible_without_performance_claim():
    root=Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122/common-validation-covered/protocol')
    if not root.exists():pytest.skip('Optional owned protocol is not present')
    design=release.read(root/'design.json');protocol=release.read(root/'protocol.json')
    plans=[release.read(root/'validation'/f'{wid}.json')for wid in ('agx1','agx2')]
    release.validate_geometry_protocol(design,protocol,plans)
    recipes,_=release.protocol_recipes(protocol);case=next(iter(recipes.values()))
    raw=release.checked(root/'profiles'/Path(case['base_config']).name,case['base_config_sha256'])
    archive=release.archive_files(Path('/Users/kilobao/Desktop/ORAN/SMO/quantification-study-20261122/versions/v2/source.tgz'))
    value=release.validate_geometry_base(raw,archive[release.BASE_CONFIG])
    assert value['external_base_sha256']==case['base_config_sha256']
    assert all(release.sha_bytes(archive[n])==p for n,p in protocol['frozen_models'].items())


def test_reproduction_argument_recipe_rejects_output_escape_and_unlisted_apps(metadata,tmp_path):
    gate=validate(metadata);reproduction=release.reproduction_recipe(gate,metadata[3])
    for arm,seed in [('D','../../outside'),('../../outside',7),('A',True),('A',2**32),('A',-1)]:
        with pytest.raises(release.ReleaseError):release.reproduction_arguments(reproduction,tmp_path,arm,seed)
    reproduction['argv_template'][reproduction['argv_template'].index('--live-state')+1]='../escape.json'
    with pytest.raises(release.ReleaseError,match='under_project_outputs'):
        release.reproduction_arguments(reproduction,tmp_path,'A',7)


def test_timelines_require_exact_cross_seed_bytes_and_native_warmup(tmp_path):
    rows=[]
    for seed in [101,102]:
        for arm in ['OFF',*release.APPS]:
            job=tmp_path/str(seed)/arm;run=job/'data/run';run.mkdir(parents=True)
            path=job/'data/timeline.json'
            value=[{'at_ms':1020.,'operation':'set','apps':[]if arm=='OFF'else[release.APPS[arm][0]],'cell':'*'}]
            path.write_bytes(release.encoded(value))
            rows.append({'case':{'scene':'fixture','arm':arm,'seed':seed,'baseline_slot':2040},'run_path':str(run),
                         'files_sha256':{'data/timeline.json':release.file_sha(path)}})
    assert len(release.sealed_timelines(rows,'fixture'))==7
    row=next(r for r in rows if r['case']['seed']==102 and r['case']['arm']=='D')
    path=Path(row['run_path']).parents[1]/'data/timeline.json'
    path.write_bytes(path.read_bytes()+b'\n');row['files_sha256']['data/timeline.json']=release.file_sha(path)
    with pytest.raises(release.ReleaseError,match='timeline_bytes_differ'):
        release.sealed_timelines(rows,'fixture')
    value=json.loads(path.read_bytes());value[0]['at_ms']=0;path.write_bytes(release.encoded(value))
    row['files_sha256']['data/timeline.json']=release.file_sha(path)
    with pytest.raises(release.ReleaseError,match='post_warmup_singleton'):
        release.sealed_timelines(rows,'fixture')


def test_file_tampering_rejected_even_with_unchanged_positive_metadata(metadata,tmp_path,monkeypatch):
    args=temporary_evidence(metadata,tmp_path,monkeypatch)
    path=Path(metadata[0]['complete_cases'][0]['run_path'])/'rt_sys_trace.jsonl'
    path.write_text('tampered raw data')
    with pytest.raises(release.ReleaseError,match='pinned_file_changed'):release.prepare_release(*args)


def test_full_synthetic_long_packaging_keeps_twenty_thousand_observation_slots(metadata,tmp_path,monkeypatch):
    args=temporary_evidence(long_metadata(metadata),tmp_path,monkeypatch)
    files=release.prepare_release(*args)
    assert json.loads(files['environment.json'])['observed_window']['observe_slots']==20000
    reproduction=json.loads(files['reproducibility.json'])
    assert reproduction['after_interval']=={'left_exclusive':2040,'right_inclusive':22040}
    assert reproduction['window_slots']['total_slots']==22440


@pytest.mark.parametrize('change,reason',[
    (lambda c:c['source_results'].update(sha256=pin('old-partial-result')),'final_results_or_worker'),
    (lambda c:c['evidence']['D']['checkpoint'].update(sha256=pin('other-case')),'pair_recording_mismatch'),
    (lambda c:c.update(canonical_target={'status':'improved'}),'primary_metric_changed'),
    (lambda c:c.update(exact_primary_birth_exposure_used=False),'primary_metric_changed'),
])
def test_independent_D_diagnostics_cannot_be_borrowed_from_another_result_or_case(metadata,tmp_path,monkeypatch,change,reason):
    args=list(temporary_evidence(metadata,tmp_path,monkeypatch));path,_=args[6][0]
    cert=json.loads(path.read_bytes());change(cert);path.write_bytes(release.encoded(cert))
    args[6][0]=(path,release.file_sha(path))
    with pytest.raises(release.ReleaseError,match=reason):release.prepare_release(*args)


def test_unequal_D_completed_membership_is_reported_without_new_performance_threshold(metadata,tmp_path,monkeypatch):
    args=list(temporary_evidence(metadata,tmp_path,monkeypatch));path,_=args[6][0]
    cert=json.loads(path.read_bytes());cert.update(completed_task_keys_equal=False,common_completed_count=0,
        common_completed_OFF_mean_ms=None,common_completed_D_mean_ms=None,common_completed_mean_change_ms=None,
        OFF_only_completed=[['ue_0',n]for n in range(10)],D_only_completed=[['ue_4',n]for n in range(10)])
    path.write_bytes(release.encoded(cert));args[6][0]=(path,release.file_sha(path))
    files=release.prepare_release(*args)
    copied=json.loads(files['evidence/D-diagnostics/seed-101.json'])
    assert copied['common_completed_count']==0 and copied['completed_task_keys_equal'] is False
    assert json.loads(files['environment.json'])['D_identity_diagnostics']['complete'] is True


def test_compact_report_cannot_be_used_for_publication(tmp_path):
    p=tmp_path/'compact-results.json';p.write_text(json.dumps({'analysis_role':'validation','paired_results':[]}))
    with pytest.raises(release.ReleaseError,match='full_results_not_compact'):
        release.prepare_release(p,tmp_path,tmp_path/'none',tmp_path/'none')


@pytest.mark.parametrize('relative',[
    'analysis/validation-partial-16-630dc728-bf675988-exec4/results.json',
    'iterations/r2/analysis/discovery-v1/results.json',
    'iterations/r3/analysis/discovery-v1/results.json',
    'iterations/r4/analysis/discovery-v1/results.json',
])
def test_existing_unsuccessful_R1_R2_R3_R4_reports_are_not_publishable(relative):
    path=Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122')/relative
    if not path.exists():pytest.skip('optional retained read-only regression data unavailable')
    report=json.loads(path.read_text())
    with pytest.raises(release.ReleaseError,match='final_validation_role_required'):
        release.validate_report(report,[],{}, {})
    assert report.get('all_six_controls_verified_and_numerically_positive') is False


def test_actual_R5_mild_six_pair_subgates_are_coherent_but_discovery_stays_unpublishable():
    root=Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122')
    alias=root/'iterations/r5/analysis/latest-discovery.json'
    if not alias.exists():pytest.skip('optional retained R5 discovery unavailable')
    result=Path(json.loads(alias.read_text())['output'])/'results.json'
    report=json.loads(result.read_text());by_id={r['case']['id']:r for r in report['complete_cases']}
    pairs=[p for p in report['paired_results']if p['scene']=='light0_hot4_175_rare6']
    if {p['arm']for p in pairs}!=set(release.APPS):pytest.skip('R5 mild six pairs not mirrored yet')
    assert len(pairs)==6
    for pair in pairs:
        release.positive_pair(pair,by_id[pair['job_id']],by_id[pair['off_job_id']])
    assert report['analysis_role']=='discovery'
    with pytest.raises(release.ReleaseError,match='final_validation_role_required'):
        release.prepare_release(result,root/'no-protocol-read',root/'no-archive-read',root/'no-runner-read')


def test_actual_LONG_protocol_schema_separates_metadata_and_preserves_one_recipe():
    root=Path('/Users/kilobao/Desktop/ORAN/SMO/environment-validation-20261122/common-validation-long/protocol')
    if not root.exists():pytest.skip('optional frozen LONG protocol unavailable')
    protocol=json.loads((root/'protocol.json').read_text());release.sealed_json(protocol)
    recipes,pins=release.protocol_recipes(protocol)
    assert list(recipes)==['light0_hot4_185_rare6_route80']
    assert set(pins)==set(release.RECIPE_METADATA_PINS)
    assert (protocol['observe_slots'],protocol['total_slots'])==(20000,22440)
    for host in ['agx1','agx2']:
        plan=json.loads((root/'validation'/(host+'.json')).read_text());release.sealed_json(plan)
        for name,path in release.RECIPE_METADATA_PINS.items():assert pins[name]==plan['frozen_physical_files'][path]
