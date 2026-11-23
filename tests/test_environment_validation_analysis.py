from copy import deepcopy
import json

import pytest

from experiments.analyze_environment_validation import (
    APPS,SOURCE,ExtraWindows,WindowIndex,analyze,bind_activation_audit,bind_environment_exclusions,digest,effect,environment_contrasts,file_sha,full_birth_followup,
    latency_diagnostic,make_html,pair_results,post_first_apply_quota,preview_comparison,recipe_assessments,singleton_summary,
)


def test_relative_effect_signs_and_zero_ratio_do_not_create_a_denominator():
    assert effect({'value':8},{'value':10},-1)['improvement_percent']==20
    assert effect({'value':11},{'value':10},1)['improvement_percent']==10
    result=effect({'value':.07},{'value':0},-1,ratio=True)
    assert result['status']=='worsened'and result['improvement_percent']is None
    assert result['improvement_percentage_points']==pytest.approx(-7)
    assert effect({'value':None,'reason':'no_mature_task_samples'},{'value':10},-1)['status']=='unavailable'


def test_numeric_guard_keeps_raw_delta_without_counting_floating_noise_as_success():
    x=effect({'value':5-1e-14},{'value':5},-1)
    assert x['status']=='unchanged'and x['change']!=0
    assert x['numerical_tolerance']==1e-9
    assert effect({'value':90+1e-10},{'value':90},-1)['status']=='unchanged'


def test_goodput_half_byte_quantum_floor_is_separate_from_practical_significance():
    def metric(v):return dict(value=v,resolution=4e-6,aggregation='counter_delta_over_equal_window')
    assert effect(metric(30+1e-6),metric(30),1)['status']=='unchanged'
    x=effect(metric(30+4e-6),metric(30),1)
    assert x['status']=='improved'and x['change_in_counter_quanta']==pytest.approx(1)
    assert x['practical_significance_claimed']is False


def test_url_failure_direction_uses_exact_counts_and_checks_ratio_consistency():
    def metric(f,n):return dict(value=f/n,failure_events=f,offered_packets=n)
    x=effect(metric(85,4000),metric(86,4000),-1,ratio=True)
    assert x['status']=='improved'and x['numerical_tolerance']==0
    assert x['classification_basis']=='exact_integer_failure_event_ratio_comparison'
    assert effect(metric(172,8000),metric(86,4000),-1,ratio=True)['status']=='unchanged'
    broken=metric(85,4000);broken['value']=.1
    assert effect(broken,metric(86,4000),-1,ratio=True)['status']=='unavailable'


def config():
    return dict(manifest=dict(slot_duration_ms=1),configs=dict(ue=dict(ues=[dict(ue_id='u')]),
        phy=dict(rt_sys=dict(ue_services=dict(mec=dict(tasks_by_ue={
            'u':dict(interval_ms=50,compute_gflop=.05)}))))))


def extra(*,missing_exo_slot=None,gap=False):
    w=ExtraWindows(config())
    for slot in [0,1,2,3]:
        if gap and slot==2:continue
        arrivals=dict(u=dict(exogenous_bytes=8,mec_result_bytes=2))if slot!=missing_exo_slot else{}
        users=[dict(ue_id='u',packet=dict(offered_bytes=slot*10,delivered_bytes=slot*6,
            overflow_bytes=slot,pending_bytes=slot*3))]
        w.observe(dict(slot_id=slot,traffic_arrivals=arrivals),users,
            dict(u=dict(ue_service_policy=dict(offload_fraction=1 if slot==2 else .5))))
    return w


def test_exact_flows_separate_exogenous_and_endogenous_arrivals_and_conserve():
    r=extra().window(0,3)
    assert r['offered_packet_bytes']==30
    assert r['exogenous_offered_bytes']==24 and r['mec_result_offered_bytes']==6
    assert r['delivered_packet_bytes']==18 and r['overflow_bytes']==3 and r['pending_bytes_change']==9
    assert r['packet_conservation_residual_bytes']==0
    assert r['all_packet_offered_load_mbps']==pytest.approx(.08)
    demand=r['per_workload_expected_demand_gflops']['u']
    assert demand['value']==pytest.approx(2/3)and demand['minimum']==.5 and demand['maximum']==1


def test_missing_arrival_data_is_not_silently_counted_as_zero_and_gaps_fail():
    r=extra(missing_exo_slot=2).window(0,3)
    assert r['exogenous_offered_bytes']is None
    assert r['offered_packet_bytes']==30
    assert extra(gap=True).window(0,3)['status']=='unavailable'


def preview():
    def action(cell,proposed,current=1):return dict(parameter='ue_scheduler_weights',cell_id=cell,
        ue_id='u'+cell,current=current,proposed=proposed,changed=proposed!=current)
    def app(name,acts):return dict(app_id='xapp:'+name,status='proposed',inference_invoked=True,
        observation_slot=200,input_sha256='a'*64,actions=acts)
    return dict(slot=200,apps=[app('throughput_scheduler_mlp',[action('0',2),action('1',1)]),
        app('latency_scheduler_rf',[action('0',2)])])


def test_proposal_overlap_distinguishes_shared_scope_whole_bundle_and_maintained():
    r=preview_comparison(preview())
    assert r['shared_resource_equal']is True and r['whole_bundle_equal']is False
    assert r['throughput_only_fields']==1 and r['throughput_maintained_fields']==1
    assert r['changed_shared_equal']is True
    p=preview();p['apps'][1]['actions'][0]['proposed']=.5
    assert preview_comparison(p)['shared_resource_equal']is False


def test_noncontemporaneous_future_or_missing_policy_observations_are_rejected():
    for observation in [None,199,201]:
        p=preview();p['apps'][1]['observation_slot']=observation
        assert preview_comparison(p)['status']=='unavailable'
    p=preview();p['apps'][0]['inference_invoked']=False
    assert preview_comparison(p)['status']=='unavailable'
    p=preview();p['apps'][1]['input_sha256']='b'*64
    assert preview_comparison(p)['reason']=='same_public_observation_hash_required'


def run(arm='OFF',*,origin='o',worker='a',scene='selected'):
    states={'kpi:'+v[2]:dict(value=1)for v in APPS.values()}
    return dict(status='complete',stage='validation',worker_id=worker,origin_id=origin,
        source_sha256='source',input_model_sha256={'m':'sha'},
        case=dict(id=arm+scene,origin_key=scene,scene=scene,seed=1,arm=arm,
            app_ids=[]if arm=='OFF'else['xapp:'+APPS[arm][0]],
            baseline_slot=2040,origin_observe_slots=2000,observe_slots=4000),
        windows=dict(before=dict(states=deepcopy(states),window_slots=2000),after=dict(states=deepcopy(states),window_slots=4000)),
        flows={period:dict(exogenous_arrivals_sha256='a'*64,task_births_sha256='b'*64)for period in ('before','after')})


def test_app_effect_requires_same_origin_same_host_and_identical_before():
    off=run();on=run('C');on['windows']['after']['states']['kpi:network_goodput_mbps']['value']=2
    paired,_=pair_results([off,on]);assert paired[0]['target']['improvement_percent']==100
    on['origin_id']='different'
    with pytest.raises(ValueError,match='different_origin'):pair_results([off,on])
    on=run('C',worker='b');assert pair_results([off,on])[0]==[]
    on=run('C');on['windows']['before']['states']['kpi:ue_power_w']['value']=3
    with pytest.raises(ValueError,match='before_windows_differ'):pair_results([off,on])


def test_native_environment_contrast_does_not_become_an_app_effect():
    native=run(scene='native_reference',origin='n');selected=run()
    selected['windows']['after']['states']['kpi:network_goodput_mbps']['value']=2
    paired,baselines=pair_results([native,selected]);assert paired==[]
    contrasts=environment_contrasts(baselines)
    assert len(contrasts)==1 and contrasts[0]['native_off_job_id']=='OFFnative_reference'
    assert contrasts[0]['comparison']=='selected_OFF_after_vs_native_reference_OFF_after'


def test_future_exogenous_traffic_or_birth_mismatch_cannot_be_called_paired():
    off=run();on=run('C');on['flows']['after']['exogenous_arrivals_sha256']='c'*64
    with pytest.raises(ValueError,match='same_origin_exogenous_arrivals_differ'):pair_results([off,on])
    on=run('C');on['flows']['after']['task_births_sha256']='c'*64
    with pytest.raises(ValueError,match='same_origin_exogenous_task_births_differ'):pair_results([off,on])
    on=run('C');on['flows']['after'].pop('task_births_sha256')
    with pytest.raises(ValueError,match='complete_external_input_evidence_required'):pair_results([off,on])


def test_task_birth_identity_excludes_controlled_route_but_includes_frozen_compute():
    def signature(route,compute=.05):
        cfg=config();cfg['configs']['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue']['u']['compute_gflop']=compute
        index=ExtraWindows(cfg)
        for slot in range(3):
            event=dict(kind='task_generated',ue_id='u',task_id=1,created_s=.001,time_s=.001,
                deadline_s=.1,offloaded=route,serving_cell='controlled-cell')
            index.observe(dict(slot_id=slot,traffic_arrivals={'u':dict(exogenous_bytes=0,mec_result_bytes=0)},
                ue_service_events=[event]if slot==1 else[]),
                [dict(ue_id='u',packet=dict(offered_bytes=0,delivered_bytes=0,overflow_bytes=0,pending_bytes=0))],
                {'u':dict(ue_service_policy=dict(offload_fraction=.5))})
        return index.window(0,2)['task_births_sha256']
    assert signature(True)==signature(False)
    assert signature(True)!=signature(True,.1)


def test_mec_deadline_diagnostic_includes_failures_without_inventing_completion_latency():
    profiles={'u':dict(deadline_ms=100)}
    x=latency_diagnostic(dict(value=20,generated=10,sample_count=8,failed=2,pending=0),profiles)
    assert x['deadline_penalized_mean_ms']==36
    x=latency_diagnostic(dict(value=None,generated=2,sample_count=0,failed=2,pending=0),profiles)
    assert x['deadline_penalized_mean_ms']==100
    x=latency_diagnostic(dict(value=None,generated=2,sample_count=1,failed=0,pending=1),profiles)
    assert x['deadline_penalized_mean_ms']is None
    assert x['reason']=='completed_latency_sum_unavailable'


def test_faster_conditional_mec_mean_with_failed_tasks_is_flagged():
    off=run();on=run('D')
    off['windows']['after']['states']['kpi:mec_offloaded_latency_ms'].update(value=20,generated=10,sample_count=10,failed=0,pending=0)
    on['windows']['after']['states']['kpi:mec_offloaded_latency_ms'].update(value=10,generated=10,sample_count=8,failed=2,pending=0)
    for row in (off,on):row['network_config']=dict(task_profiles={'u':dict(deadline_ms=100)})
    paired,_=pair_results([off,on]);r=paired[0]
    assert r['target']['status']=='improved'
    assert r['mec_completion_diagnostic']['selection_effect_concern']is True
    assert r['mec_completion_diagnostic']['after']['deadline_penalized_mean_ms']==28
    assert r['mec_completion_diagnostic']['off_after']['deadline_penalized_mean_ms']==20


def sealed_pilot(tmp_path):
    worker=tmp_path/'worker';run=worker/'jobs/id/data/run';run.mkdir(parents=True)
    (worker/'checkpoints').mkdir()
    case=dict(id='id',name='pilot',origin_key='pilot',scene='pilot',seed=7,arm='OFF',app_ids=[],
        worker_id='a',slots=12,baseline_slot=5,origin_observe_slots=3,observe_slots=4)
    plan=dict(stage='pilot',worker_id='a',source_sha256=SOURCE,source_archive_sha256='archive',runner_sha256='runner',
        frozen_models={str(i):str(i)*64 for i in range(6)},cases=[case])
    plan['plan_sha256']=digest(plan);planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan))
    cfg=config();cfg['source_sha256']=SOURCE;cfg['input_model_sha256']={'model':'frozen'}
    cfg['manifest'].update(ues=[dict(ue_id='u')],cells=[dict(cell_id='c')],total_slots=12,rng_seed=7)
    cfg['configs']['ue']['ues'][0]['slice_id']='urllc'
    cfg['configs']['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue']['u']['deadline_ms']=1
    (run/'config_snapshot.json').write_text(json.dumps(cfg))
    (run/'run_status.json').write_text(json.dumps(dict(completed=True,source_sha256=SOURCE)))
    rows=[]
    for slot in range(1,13):
        packet=dict(offered_bytes=slot*10,delivered_bytes=slot*8,pending_bytes=slot*2,overflow_bytes=0,
            admitted_packets=slot,overflow_packets=0,deadline_failed_packets=0,completed_packets=slot,
            completion_latency_ms_sum=0,hol_delay_ms=1)
        user=dict(ue_id='u',cell_id='c',queue_bytes=slot*2,**{'packet_'+k:v for k,v in packet.items()})
        service=dict(ue_energy_total_j=slot*.005,ue_active_fraction=1,
            ue_service_policy=dict(offload_fraction=.5),ue_task_metrics={k:0 for k in (
                'generated','completed','failed','on_time_completed','offloaded_generated','offloaded_completed','offloaded_failed')})
        rows.append(dict(slot_id=slot,ues=[user],traffic_arrivals={'u':dict(exogenous_bytes=10,mec_result_bytes=0)},
            ue_services={'u':service},ap_services=dict(cells={'c':dict(ap_energy_total_j=slot*.03,ap_available_slots_total=slot)}),
            ue_service_events=[]))
    (run/'rt_sys_trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    seal=dict(completed=True,case=case,worker_id='a',actual_system_hostname='host-a',stage='pilot',
        source_sha256=SOURCE,source_archive_sha256='archive',runner_sha256='runner',plan_sha256=plan['plan_sha256'],
        frozen_models=plan['frozen_models'],models_unchanged=True,training_performed=False,
        post_deployment_kpis_used_for_model_training=False,origin_id='origin',
        files_sha256={str(p.relative_to(run.parents[1])):file_sha(p)for p in run.iterdir()})
    (worker/'checkpoints/id.json').write_text(json.dumps(seal))
    return planpath,worker,run


def test_completed_sealed_pilot_writes_readable_report_without_imputing_mec_delay(tmp_path):
    plan,worker,_=sealed_pilot(tmp_path)
    output=tmp_path/'report';r=analyze([plan],{'a':worker},output,plots=False)
    assert len(r['baselines'])==1 and r['paired_results']==[]
    state=r['baselines'][0]['windows']['after']['states']['kpi:mec_offloaded_latency_ms']
    assert state['value']is None and state['reason']=='no_mature_task_samples'
    assert r['baselines'][0]['flows']['after']['packet_conservation_residual_bytes']==0
    assert 'Environment validation'in(output/'index.html').read_text()
    assert (output/'analysis-manifest.json').is_file()


def test_changed_sealed_raw_file_is_rejected_before_analysis(tmp_path):
    plan,worker,run=sealed_pilot(tmp_path)
    with(run/'rt_sys_trace.jsonl').open('a')as stream:stream.write('{}\n')
    with pytest.raises(ValueError,match='sealed_recording_changed'):
        analyze([plan],{'a':worker},tmp_path/'report',plots=False)


def receiver_manifest(worker,*,pin=None):
    value={'id':{'seal_sha256':pin or file_sha(worker/'checkpoints/id.json'),'files':3,'verified_unix_s':1}}
    (worker/'received-checkpoints.json').write_text(json.dumps(value))


def test_receiver_gate_missing_or_stale_never_reads_unpublished_raw(tmp_path):
    plan,worker,run=sealed_pilot(tmp_path)
    (run/'rt_sys_trace.jsonl').write_text('not finished mirroring')
    for pin in (None,'stale'):
        if pin:receiver_manifest(worker,pin=pin)
        r=analyze([plan],{'a':worker},tmp_path/str(pin),plots=False,require_receiver_seals=True)
        assert not r['complete_cases']
        assert r['pending_cases'][0]['reason']=='receiver_verified_checkpoint_pending'


def test_receiver_gate_preserves_measurements_and_raw_checksum_requirement(tmp_path):
    plan,worker,run=sealed_pilot(tmp_path);receiver_manifest(worker)
    plain=analyze([plan],{'a':worker},tmp_path/'plain',plots=False)
    gated=analyze([plan],{'a':worker},tmp_path/'gated',plots=False,require_receiver_seals=True)
    assert plain['complete_cases'][0]['windows']==gated['complete_cases'][0]['windows']
    assert plain['complete_cases'][0]['flows']==gated['complete_cases'][0]['flows']
    assert gated['complete_cases'][0]['receiver_admission']['checkpoint_sha256']==file_sha(worker/'checkpoints/id.json')
    (run/'rt_sys_trace.jsonl').write_text('corrupted after receiver verification')
    with pytest.raises(ValueError,match='sealed_recording_changed'):
        analyze([plan],{'a':worker},tmp_path/'corrupt',plots=False,require_receiver_seals=True)


def test_checkpoint_arriving_during_analysis_is_individually_receiver_gated(tmp_path,monkeypatch):
    import experiments.analyze_environment_validation as module
    planpath,worker,runpath=sealed_pilot(tmp_path)
    plan=json.loads(planpath.read_text());newcase={**plan['cases'][0],'id':'later'}
    plan['cases'].append(newcase);plan.pop('plan_sha256');plan['plan_sha256']=digest(plan);planpath.write_text(json.dumps(plan))
    cp=worker/'checkpoints/id.json';seal=json.loads(cp.read_text());seal['plan_sha256']=plan['plan_sha256'];cp.write_text(json.dumps(seal))
    receiver_manifest(worker)
    original=module.records
    def publish_during_read(*args,**kwargs):
        (worker/'checkpoints/later.json').write_text('unfinished checkpoint copy')
        yield from original(*args,**kwargs)
    monkeypatch.setattr(module,'records',publish_during_read)
    r=analyze([planpath],{'a':worker},tmp_path/'report',plots=False,require_receiver_seals=True)
    assert len(r['complete_cases'])==1 and len(r['pending_cases'])==1
    assert r['pending_cases'][0]['case']['id']=='later'
    assert r['pending_cases'][0]['reason']=='receiver_verified_checkpoint_pending'


def test_checkpoint_changes_during_raw_read_are_rejected(tmp_path,monkeypatch):
    import experiments.analyze_environment_validation as module
    plan,worker,run=sealed_pilot(tmp_path);receiver_manifest(worker)
    original=module.records
    def replace_checkpoint(*args,**kwargs):
        yield from original(*args,**kwargs)
        with(worker/'checkpoints/id.json').open('a')as stream:stream.write(' ')
    monkeypatch.setattr(module,'records',replace_checkpoint)
    with pytest.raises(ValueError,match='checkpoint_changed_during_analysis'):
        analyze([plan],{'a':worker},tmp_path/'report',plots=False,require_receiver_seals=True)


def test_followup_is_separate_and_does_not_repair_the_primary_window_with_future_data():
    cfg=config();cfg['configs']['phy']['rt_sys']['ue_services']['mec']['tasks_by_ue']['u']['deadline_ms']=2
    index=WindowIndex(cfg)
    birth=dict(ue_id='u',task_id=1,kind='task_generated',created_s=.002,time_s=.002,deadline_s=.002,offloaded=True)
    terminal=dict(birth,kind='task_completed',time_s=.006,latency_ms=4)
    for slot in range(1,9):
        counts=dict(generated=int(slot>=3),offloaded_generated=int(slot>=3),completed=int(slot>=7),
            offloaded_completed=int(slot>=7),failed=0,offloaded_failed=0,on_time_completed=0)
        index.observe(slot,[dict(ue_id='u')],{'u':dict(ue_task_metrics=counts)}, {},
            [birth]if slot==3 else[terminal]if slot==7 else[])
    primary=index.window(['mec_offloaded_latency_ms'],2,5,through=8)['states']['kpi:mec_offloaded_latency_ms']
    assert primary['value']is None and primary['pending']==1
    followup=full_birth_followup(index,2,5,8,offloaded_only=True)
    assert followup['value']==4 and followup['sample_count']==1 and followup['pending']==0
    assert followup['deadline_success_fraction']==0
    assert followup['changes_primary_guarded_target']is False


def test_html_target_units_are_not_overwritten_by_side_effect_units(tmp_path):
    pairs=[]
    for arm in APPS:
        off=run();on=run(arm)
        pairs.extend(pair_results([off,on])[0])
    report=dict(complete_cases=[],pending_cases=[],baselines=[],paired_results=pairs,
        environment_contrasts=[],plot_created=False)
    make_html(report,tmp_path);markup=(tmp_path/'index.html').read_text()
    assert '1.000 W → 1.000 W'in markup
    assert '1.000 Mbit/s → 1.000 Mbit/s'in markup
    assert '1.000 ms → 1.000 ms'in markup
    assert '100.00% → 100.00%'in markup
    assert '1.000 GFLOP/s → 1.000 GFLOP/s'in markup
    assert '1.000 bytes → 1.000 bytes'not in markup


def test_two_seed_zero_error_url_is_not_counted_as_a_successful_app():
    cases=[]
    for seed in (1,2):
        for arm in ('OFF','E'):
            r=run(arm,scene='selected'+str(seed));r['case']['seed']=seed
            r['case']['scene']='selected'
            r['windows']['after']['states']['kpi:urllc_failure_fraction'].update(
                value=0,failure_events=0,offered_packets=4000,sample_count=4000)
            cases.append(r)
    pairs,_=pair_results(cases);summary=singleton_summary(pairs,cases)
    assert summary[0]['counts']=={'unchanged':2}
    assert summary[0]['all_planned_pairs_numerically_positive']is False
    assert summary[0]['status']=='requires_environment_iteration'


def test_two_different_recipe_seeds_are_not_pooled_as_two_seed_validation():
    cases=[]
    for seed,scene in [(6,'recipe_a'),(7,'recipe_b')]:
        for arm in ('OFF','A'):
            r=run(arm,scene=scene);r['case']['seed']=seed
            if arm=='A':r['windows']['after']['states']['kpi:ue_power_w']['value']=.9
            cases.append(r)
    pairs,_=pair_results(cases);summary=singleton_summary(pairs,cases)
    assert len(summary)==2 and all(s['planned_pairs']==1 for s in summary)
    assert {s['scene']for s in summary}=={'recipe_a','recipe_b'}
    assessments=recipe_assessments(summary,'discovery')
    assert all(a['same_environment_independent_two_seed_validation_passed']is False for a in assessments)


def test_natural_100ms_windows_keep_partial_edges_and_backlog_occupancy():
    cfg=config();cfg['configs']['ue']['ues'][0]['slice_id']='embb';index=ExtraWindows(cfg)
    for slot in range(251):
        index.observe(dict(slot_id=slot,traffic_arrivals={'u':dict(exogenous_bytes=10,mec_result_bytes=0)}),
            [dict(ue_id='u',packet=dict(offered_bytes=slot*10,delivered_bytes=slot*8,overflow_bytes=0,pending_bytes=slot*2))],
            {'u':dict(ue_service_policy=dict(offload_fraction=.5))})
    frame=index.window(40,240);parts=frame['natural_100ms_subwindows']
    assert [p['exposure_ms']for p in parts]==[60,100,40]
    assert [p['partial_edge']for p in parts]==[True,False,True]
    assert [p['end_slot_inclusive']for p in parts]==[100,200,240]
    assert frame['service_class_flows']['embb']['backlogged_slot_fraction']==1
    assert sum(p['service_class_flows']['embb']['offered_bytes']for p in parts)==frame['service_class_flows']['embb']['offered_bytes']
    assert sum(p['service_class_flows']['embb']['pending_bytes_change']for p in parts)==frame['service_class_flows']['embb']['pending_bytes_change']


def exclusion_fixture(tmp_path):
    root=tmp_path/'worker';(root/'checkpoints').mkdir(parents=True)
    case=dict(id='unstarted',arm='C',scene='rejected',seed=6,worker_id='a',origin_key='rejected_s6')
    plan=dict(cases=[case],worker_id='a',stage='pilot');plan['plan_sha256']=digest(plan)
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
    receipt=dict(schema='oran.environment-off-causal-rejection.v1',source_sha256=SOURCE,
        models_or_actions_modified=False,plan_sha256=plan['plan_sha256'],reason='Failure predates first possible control')
    rp=tmp_path/'rejection.json';rp.write_text(json.dumps(receipt))
    entry=dict(case,case_id='unstarted',job_id='unstarted',status='not_run_environment_rejected',rejection_receipt_sha256=file_sha(rp))
    manifest=dict(schema='oran.environment-excluded-cases.v1',rejection_receipt=dict(path=str(rp),sha256=file_sha(rp)),
        plan_file=dict(path=str(pp),sha256=file_sha(pp)),plan_sha256=plan['plan_sha256'],
        excluded_cases=[entry],planned=1,actual_target=0,excluded_count=1)
    mp=tmp_path/'exclusions.json';mp.write_text(json.dumps(manifest))
    rows=[dict(status='pending',case=case,stage='pilot',worker_id='a')]
    return rows,mp,plan,root


def test_explicit_not_run_is_not_missing_or_successful_and_cannot_hide_actual_data(tmp_path):
    rows,mp,plan,root=exclusion_fixture(tmp_path)
    ref=bind_environment_exclusions(rows,mp,[plan],{'a':root})
    assert rows[0]['status']=='not_run_environment_rejected'and ref['excluded_count']==1
    summary=singleton_summary([],rows)[0]
    assert summary['status']=='not_run_environment_rejected'
    assert summary['missing_job_ids']==[]and summary['paired']==0
    assert summary['all_planned_pairs_numerically_positive']is False
    job=root/'jobs/unstarted';job.mkdir(parents=True);(job/'request.json').write_text('{}')
    rows[0]['status']='pending'
    with pytest.raises(ValueError,match='excluded_case_has_actual_run_evidence'):
        bind_environment_exclusions(rows,mp,[plan],{'a':root})


def multi_plan_exclusion_fixture(tmp_path):
    plans=[];rows=[];refs={};roots={}
    for worker,seed in [('a',10),('b',11)]:
        root=tmp_path/worker;(root/'checkpoints').mkdir(parents=True);roots[worker]=root
        case=dict(id='unstarted_'+worker,arm='C',scene='rejected',seed=seed,
            worker_id=worker,origin_key='rejected_'+worker)
        plan=dict(cases=[case],worker_id=worker,stage='validation');plan['plan_sha256']=digest(plan)
        pp=tmp_path/(worker+'-plan.json');pp.write_text(json.dumps(plan));plans.append(plan)
        refs[worker]=dict(path=str(pp),sha256=file_sha(pp),plan_sha256=plan['plan_sha256'])
        rows.append(dict(status='pending',case=case,stage='validation',worker_id=worker))
    receipt=dict(schema='oran.environment-off-causal-rejection.v1',source_sha256=SOURCE,
        models_or_actions_modified=False,plan_sha256_by_worker={p['worker_id']:p['plan_sha256']for p in plans},
        reason='Sealed OFF has persistent coverage loss and unresolved mature tasks')
    rp=tmp_path/'rejection.json';rp.write_text(json.dumps(receipt))
    entries=[dict(r['case'],case_id=r['case']['id'],job_id=r['case']['id'],
        status='not_run_environment_rejected',rejection_receipt_sha256=file_sha(rp),
        plan_sha256=refs[r['worker_id']]['plan_sha256'])for r in rows]
    manifest=dict(schema='oran.environment-excluded-cases.v1',
        rejection_receipt=dict(path=str(rp),sha256=file_sha(rp)),plan_files=refs,
        excluded_cases=entries,planned=2,actual_target=0,excluded_count=2)
    mp=tmp_path/'exclusions.json';mp.write_text(json.dumps(manifest))
    return rows,mp,plans,roots


def test_multi_plan_not_run_binds_both_original_worker_plans(tmp_path):
    rows,mp,plans,roots=multi_plan_exclusion_fixture(tmp_path)
    result=bind_environment_exclusions(rows,mp,plans,roots)
    assert result['planned']==2 and result['actual_target']==0 and result['excluded_count']==2
    assert all(r['status']=='not_run_environment_rejected'for r in rows)
    assert all(not r['all_planned_pairs_numerically_positive']for r in singleton_summary([],rows))


@pytest.mark.parametrize('corruption',['missing_worker','swapped_plan','wrong_entry_pin','wrong_receipt_pin','ambiguous_legacy_plan'])
def test_multi_plan_exclusion_rejects_cross_plan_or_ambiguous_identity(tmp_path,corruption):
    rows,mp,plans,roots=multi_plan_exclusion_fixture(tmp_path);m=json.loads(mp.read_text())
    if corruption=='missing_worker':m['plan_files'].pop('b')
    elif corruption=='swapped_plan':m['plan_files']['a'],m['plan_files']['b']=m['plan_files']['b'],m['plan_files']['a']
    elif corruption=='wrong_entry_pin':m['excluded_cases'][0]['plan_sha256']=plans[1]['plan_sha256']
    elif corruption=='ambiguous_legacy_plan':m['plan_file']=m['plan_files']['a']
    else:
        rp=tmp_path/'rejection.json';receipt=json.loads(rp.read_text());receipt['plan_sha256_by_worker']['a']='wrong'
        rp.write_text(json.dumps(receipt));m['rejection_receipt']['sha256']=file_sha(rp)
        for e in m['excluded_cases']:e['rejection_receipt_sha256']=file_sha(rp)
    mp.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='excluded_case_plan_identity_mismatch'):
        bind_environment_exclusions(rows,mp,plans,roots)


def test_multi_plan_exclusion_cannot_hide_second_host_request(tmp_path):
    rows,mp,plans,roots=multi_plan_exclusion_fixture(tmp_path)
    job=roots['b']/'jobs/unstarted_b';job.mkdir(parents=True);(job/'request.json').write_text('{}')
    with pytest.raises(ValueError,match='excluded_case_has_actual_run_evidence'):
        bind_environment_exclusions(rows,mp,plans,roots)


def test_independent_control_audit_binds_checkpoint_and_does_not_equate_applied_with_changed(tmp_path):
    row=run('A');row.update(checkpoint=dict(sha256='cp'),runner_sha256='runner',source_archive_sha256='archive')
    entry=dict(job_id=row['case']['id'],status='ready',checkpoint_sha256='cp',source_sha256='source',
        runner_sha256='runner',source_archive_sha256='archive',worker_id='a',activation_verified=True,frozen_verified=True,
        consumer_verified=True,trace_consumer_mismatches=[],
        actual_changed=False,own_applied_receipt_count=1,changed_receipt_count=0,noop_receipt_count=1,
        consumer_fields_unresolved=0,consumer_mismatches=[],effect_provenance_status='deployed_noop')
    p=tmp_path/'audit.json';p.write_text(json.dumps(dict(schema='oran.environment-activation-audit.v1',entries=[entry])))
    bind_activation_audit([row],p)
    assert row['control_provenance']['status']=='deployed_noop'
    assert row['control_provenance']['qualified']is False
    entry.update(actual_changed=True,effect_provenance_status='changed_control')
    p.write_text(json.dumps(dict(schema='oran.environment-activation-audit.v1',entries=[entry])))
    bind_activation_audit([row],p);assert row['control_provenance']['qualified']is True
    entry['checkpoint_sha256']='wrong';p.write_text(json.dumps(dict(schema='oran.environment-activation-audit.v1',entries=[entry])))
    with pytest.raises(ValueError,match='activation_audit_recording_identity_mismatch'):bind_activation_audit([row],p)


def test_f_quota_starts_at_first_applied_even_if_noop_or_infeasible_not_first_good_change(tmp_path):
    row=run('F');row['case'].update(baseline_slot=2,observe_slots=6)
    row['network_config']=dict(slot_duration_ms=.5,task_profiles={'ue_4':dict(interval_ms=50,compute_gflop=.05)})
    row['flows']['after']['ue4_budget']=dict(budget_gflops=.75,mean_compliant=False,maximum_compliant=False)
    row['run_path']=str(tmp_path);before=deepcopy(row['flows'])
    rows=[dict(slot_id=slot,ue_services={'ue_4':dict(ue_service_policy=dict(offload_fraction=1 if slot<=4 else .5))})for slot in range(1,9)]
    (tmp_path/'rt_sys_trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    entry=dict(first_binding_slot=3,first_applied_receipt_slot=4,first_changed_receipt_slot=5)
    q=post_first_apply_quota(row,entry)
    assert q['maximum_gflops']==1 and q['maximum_compliant']is False
    assert q['post_apply_exposure_slots']==5 and q['post_apply_exposure_ms']==2.5
    assert q['activation_to_first_apply_ms']==.5
    assert row['flows']==before
    entry['first_applied_receipt_slot']=None
    assert post_first_apply_quota(row,entry)['status']=='unavailable'


def test_f_post_apply_quota_keeps_startup_whole_window_diagnostic_separate(tmp_path):
    row=run('F');row['case'].update(baseline_slot=2,observe_slots=6)
    row['network_config']=dict(slot_duration_ms=.5,task_profiles={'ue_4':dict(interval_ms=50,compute_gflop=.05)})
    row['flows']['after']['ue4_budget']=dict(budget_gflops=.75,mean_compliant=False,maximum_compliant=False)
    row['run_path']=str(tmp_path)
    rows=[dict(slot_id=slot,ue_services={'ue_4':dict(ue_service_policy=dict(offload_fraction=1 if slot<4 else .5))})for slot in range(1,9)]
    (tmp_path/'rt_sys_trace.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    q=post_first_apply_quota(row,dict(first_binding_slot=3,first_applied_receipt_slot=4))
    assert q['mean_gflops']==.5 and q['maximum_gflops']==.5 and q['maximum_compliant']is True
    assert row['flows']['after']['ue4_budget']['maximum_compliant']is False
