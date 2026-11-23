"""Exact label-reuse equivalence on synthetic sealed raw scalar traces."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil

import pytest

from experiments import study_evaluation_cache as cache
from experiments import study_quantification as study


def fixture(tmp_path,*,sealed=True):
    run=tmp_path/'workers/w/jobs/job/data/run';run.mkdir(parents=True)
    config=dict(source_sha256='source',manifest=dict(rng_seed=7,slot_duration_ms=1,total_slots=5,ues=[{'ue_id':'u'}],cells=[{'cell_id':'c'}]),
        configs={'ue':{'ues':[{'ue_id':'u','slice_id':'urllc'}]}})
    study.write(run/'config_snapshot.json',config);study.write(run/'run_status.json',dict(completed=True,source_sha256='source',replay={'audits_passed':True}))
    guard=dict(origin_id='origin',after_kpis_used=False,**{k:True for k in('verified_before_activation','observed_prefix_equal','configuration_equal','models_equal','network_dynamics_code_equal')})
    study.write(run/'evaluation_origin_verified.json',guard)
    with gzip.open(run/'rt_sys_trace.jsonl.gz','wt')as stream:
        for slot in range(1,6):
            row=dict(slot_id=slot,ues=[dict(ue_id='u',cell_id='c',packet_delivered_bytes=100*slot if slot<=3 else 300+50*(slot-3),
                packet_admitted_packets=slot,packet_overflow_packets=0,packet_deadline_failed_packets=max(0,slot-4))],
                ue_services={'u':dict(ue_energy_total_j=.001*slot if slot<=3 else.003+.002*(slot-3),ue_active_fraction=1)},
                ap_services={'cells':{'c':dict(ap_energy_total_j=.01*slot,ap_available_slots_total=slot)}},ue_service_events=[])
            stream.write(json.dumps(row)+'\n')
    checkpoint=tmp_path/'workers/w/checkpoints/job.json'
    def seal_run():
        pins={str(p.relative_to(run.parents[1])):study.file_sha(p)for p in run.iterdir()if p.is_file()}
        study.write(checkpoint,dict(completed=True,case={'id':'job','app_ids':['a']},physical_plan_sha256='physical',
            source_sha256='source',origin_id='origin',files_sha256=pins))
    if sealed:seal_run()
    provenance=dict(job_id='job',seed=7,physical_plan_sha256='physical',after_kpis_used=False)
    ranges=[dict(kpi=k,unit=unit,label=k,scope='All UEs',current=current,domain=[0,1 if k=='urllc_failure_fraction'else None],
        lower=current if up else 0,upper=None if up else current,lower_inclusive=not up,upper_inclusive=False if k=='ue_power_w'else True)
        for k,unit,current,up in(('network_goodput_mbps','Mbit/s',.8,True),('ue_power_w','W',1.,False),('urllc_failure_fraction','ratio',0.,False))]
    frame=dict(slot=3,time_ms=3,states={'kpi:'+r['kpi']:{'value':r['current']}for r in ranges})
    case=dict(case_id='case',intent_id='intent',group_id='g',app_ids=['a'],scene='native',ranges=ranges,frame=frame,preview={'slot':3,'apps':[]},
        options={'normalization':study.normalization(ranges,frame)},protocol=dict(primary_validation_eligible=True,actual_arm_origin_guard_pending=True),
        candidate={'eligible':True,'pool':'primary'},cohort_id='cohort',independence_group='job',provenance=provenance,timing={'observe_slots':2})
    source={k:deepcopy(case[k])for k in('ranges','intent_id','cohort_id','independence_group','provenance','scene','protocol','candidate')}
    source.update(run_path=str(run),seal_path=str(checkpoint),expected_origin_id='origin',requires_actual_origin_guard=True,
        before_window={'window_start_slot':1,'slot':3,'window_slots':2},after_window={'window_start_slot':3,'slot':5,'window_slots':2})
    specs=[]
    methods=('goal_obstruction_v2','net_intent_obstruction_v1','temporal_net_intent_obstruction_v1','net_profile_gain_obstruction_v1')
    for i,method in enumerate(methods):
        folder=tmp_path/'predictions'/method;folder.mkdir(parents=True);inp=folder/'scoring-inputs.json'
        study.write(inp,dict(stage='predeployment_inputs',after_values_included=False,cases=[deepcopy(case)]));study.write(folder/'label-sources.json',{'sources':{'case':source}})
        engine=folder/'synthetic-frozen-engine.js';engine.write_text('// Analytic prediction fixture only; not a real trained result.\n')
        group=dict(group_id='g',app_ids=['a'],score=float(i*10),raw_obstruction_strength=i/10,score_status='complete',reason=None,
            reference_effect=-i/20,reference_effect_score=-i*5)
        item={k:deepcopy(case[k])for k in('case_id','intent_id','cohort_id','independence_group','protocol','provenance','candidate','scene')};item['result']={'groups':[group]}
        scored=dict(stage='scores_before_label_join',method=method,after_values_used=False,input_path=str(inp),input_sha256=study.file_sha(inp),
            module_sha256={'fixture':study.file_sha(engine)},archived_modules={'fixture':str(engine)},label_sources_sha256=study.file_sha(folder/'label-sources.json'),scores=[item])
        if i==3:scored.update(graph_variant={'id':method,'synthetic_fixture':True},engine_method='net_intent_obstruction_v1')
        score_path=folder/'scores.json';study.write(score_path,scored)
        seal_path=folder/'prediction-seal.json';study.write(seal_path,dict(schema='oran.prospective-score-seal.v1',
            after_values_used=False,nonempty_arm_started_before_scoring=False,algorithm_tuned_to_test_labels=False,gate_opened=False,
            input_path=str(inp),input_sha256=study.file_sha(inp),scores={method:dict(path=str(score_path),sha256=study.file_sha(score_path),count=1)}))
        specs.append(dict(score_path=str(score_path),seal_path=str(seal_path)))
    return specs,run,seal_run


def test_four_methods_reuse_one_parse_and_match_full_join_every_original_field(tmp_path,monkeypatch):
    specs,run,_=fixture(tmp_path);full=[];real=study._observed_windows;calls=[]
    def observe(source):calls.append(source['run_path']);return real(source)
    monkeypatch.setattr(study,'_observed_windows',observe)
    for i,spec in enumerate(specs):
        source=Path(study.read(spec['score_path'])['input_path']).with_name('label-sources.json')
        path=tmp_path/f'full-{i}.json';study.join_evaluation(spec['score_path'],source,path);full.append(study.read(path))
    assert len(calls)==4;calls.clear()
    result=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='cold')
    assert len(calls)==1 and result['persistent_cache_created']is True
    for expected,ref in zip(full,result['outputs']):
        reused=study.read(ref['path'])
        assert all(reused[key]==value for key,value in expected.items())
    saved=study.read(next((tmp_path/'cache/windows').glob('*.json')))
    assert all(not set(frame)&set(cache.RISK_FIELDS)for frame in saved['windows'])
    assert saved['physical_run_sealed_complete']is True
    calls.clear();warm=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='warm')
    assert warm['persistent_cache_hit']is True and calls==[]
    assert [study.read(ref['path'])['rows'][0]['raw_risk']for ref in warm['outputs']]==[0,.1,.2,.3]


def test_pending_refresh_is_not_cached_and_does_not_hide_later_completion(tmp_path,monkeypatch):
    specs,_,seal_run=fixture(tmp_path,sealed=False);real=study._observed_windows;calls=[]
    monkeypatch.setattr(study,'_observed_windows',lambda source:(calls.append(1),real(source))[1])
    pending=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='pending')
    assert pending['persistent_cache_created']is False and calls==[]
    assert study.read(pending['outputs'][0]['path'])['rows'][0]['label_status']=='awaiting_completed_sealed_run'
    seal_run();complete=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='complete')
    assert complete['persistent_cache_hit']is False and complete['persistent_cache_created']is True and len(calls)==1
    assert study.read(complete['outputs'][0]['path'])['rows'][0]['observed_metric']is not None
    assert study.read(pending['outputs'][0]['path'])['rows'][0]['observed_metric']is None


def test_changed_actual_config_cannot_be_masked_by_warm_label_cache(tmp_path):
    specs,run,_=fixture(tmp_path);cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='initial')
    config=study.read(run/'config_snapshot.json');config['manifest']['rng_seed']=8;study.write(run/'config_snapshot.json',config)
    with pytest.raises(ValueError,match='certified_physical_data_changed'):
        cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='changed')


def test_evaluation_code_pin_change_invalidates_cache_without_tuning_labels(tmp_path,monkeypatch):
    specs,_,_=fixture(tmp_path);first=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='v1')
    real=cache.evaluation_code_pins;monkeypatch.setattr(cache,'evaluation_code_pins',lambda:real()|{'synthetic_code_version_test':'new'})
    second=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='v2')
    assert first['cache_key']!=second['cache_key']and second['persistent_cache_hit']is False
    assert [study.read(x['path'])['rows'][0]['observed_metric']for x in first['outputs']]==[study.read(x['path'])['rows'][0]['observed_metric']for x in second['outputs']]


def test_score_input_unit_or_range_mismatch_cannot_share_actual_labels(tmp_path):
    specs,_,_=fixture(tmp_path);spec=specs[1];score=study.read(spec['score_path']);inp=Path(score['input_path']);data=study.read(inp)
    data['cases'][0]['ranges'][0]['unit']='kbit/s';study.write(inp,data);score['input_sha256']=study.file_sha(inp);study.write(spec['score_path'],score)
    seal=study.read(spec['seal_path']);seal['input_sha256']=study.file_sha(inp);seal['scores'][score['method']]['sha256']=study.file_sha(spec['score_path']);study.write(spec['seal_path'],seal)
    with pytest.raises(ValueError,match='geometry_differ'):
        cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='bad')


def test_completed_window_stays_cached_while_another_arm_is_pending(tmp_path,monkeypatch):
    specs,run,_=fixture(tmp_path);pending=run.parents[4]/'w/jobs/pending/data/run';checkpoint=tmp_path/'workers/w/checkpoints/pending.json'
    for spec in specs:
        score=study.read(spec['score_path']);inp=Path(score['input_path']);data=study.read(inp)
        second=deepcopy(data['cases'][0]);second.update(case_id='case_pending',group_id='g2',independence_group='pending');second['provenance']['job_id']='pending'
        data['cases'].append(second);study.write(inp,data)
        sourcepath=inp.with_name('label-sources.json');source=study.read(sourcepath);new=deepcopy(source['sources']['case'])
        new.update(run_path=str(pending),seal_path=str(checkpoint),independence_group='pending');new['provenance']['job_id']='pending'
        source['sources']['case_pending']=new;study.write(sourcepath,source)
        item=deepcopy(score['scores'][0]);item.update(case_id='case_pending',independence_group='pending');item['provenance']['job_id']='pending';item['result']['groups'][0]['group_id']='g2'
        score['scores'].append(item);score.update(input_sha256=study.file_sha(inp),label_sources_sha256=study.file_sha(sourcepath));study.write(spec['score_path'],score)
        seal=study.read(spec['seal_path']);seal['input_sha256']=study.file_sha(inp);seal['scores'][score['method']].update(sha256=study.file_sha(spec['score_path']),count=2);study.write(spec['seal_path'],seal)
    real=study._observed_windows;calls=[]
    monkeypatch.setattr(study,'_observed_windows',lambda source:(calls.append(source['run_path']),real(source))[1])
    first=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='partial-cold')
    assert len(calls)==1 and first['persistent_cache_created']is True and first['all_physical_runs_sealed_complete']is False
    calls.clear();warm=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='partial-warm')
    assert calls==[]and warm['persistent_cache_hit']is True
    assert study.read(warm['outputs'][0]['path'])['rows'][1]['observed_metric']is None
    shutil.copytree(run,pending)
    pins={str(p.relative_to(pending.parents[1])):study.file_sha(p)for p in pending.iterdir()if p.is_file()}
    study.write(checkpoint,dict(completed=True,case={'id':'pending','app_ids':['a']},physical_plan_sha256='physical',
        source_sha256='source',origin_id='origin',files_sha256=pins))
    calls.clear();done=cache.join_with_label_reuse(specs,tmp_path/'outputs',tmp_path/'cache',refresh_id='completed-second')
    assert calls==[str(pending)]and done['all_physical_runs_sealed_complete']is True
    assert study.read(done['outputs'][0]['path'])['rows'][1]['observed_metric']is not None
