"""Synthetic six-origin supplement gates, without any simulation or labels."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import supplement_admission as gate
from experiments.quantification_supplement import assignment, digest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=gate.sha(path))


def fixture(tmp_path, monkeypatch):
    root=tmp_path/'v2';artifact=root/'supplement';artifact.mkdir(parents=True)
    physical='physical';source='source';archive='archive'
    module=root/'archive'/'engine.js';module.parent.mkdir();module.write_text('// synthetic immutable code')
    module_key='source/engine.js';module_pins={module_key:gate.sha(module)}
    roots=[];blocks=[];parent_origins=[];templates={};old_sources={}
    for i in range(6):
        key=f'scene{i}_s{i}';worker='agx1'if i%2 else'agx2';oid=f'origin{i}';off=f'OFF{i}'
        origin=dict(origin_key=key,scene=f'scene{i}',seed=i,worker_id=worker,origin_id=oid,off_job_id=off,
            origin_proof=put(root/'workers'/worker/'origins'/(key+'.json'),dict(origin_id=oid)),
            before_preview=put(root/'workers'/worker/'origins'/(key+'-before-preview.json'),dict(slot=2040)))
        roots.append(origin)
        cases=[]
        for arm in range(20):
            for intent in range(15):
                intent_id=f'intent{intent}';job=off if arm==0 else f'old-arm{arm}'
                cases.append(dict(case_id=f'{key}:{job}:{intent_id}',intent_id=intent_id,
                    group_id='study-arm:OFF'if arm==0 else f'study-arm:{arm}',app_ids=[]if arm==0 else['oldapp'],
                    scene=origin['scene'],graph_id=origin['scene'],context_id='context',
                    ranges=[dict(kpi='power',lower=0,upper=1,current=1,unit='W',anchors=[dict(current=1)])],
                    options=dict(method='goal_obstruction_v2',normalization={'kpis':[dict(kpi='power',scale=1)]}),
                    candidate=dict(pool='off_diagnostic'if arm==0 else'primary',eligible=bool(arm)),
                    protocol=dict(name='prospective_certified_OFF_prefix',primary_validation_eligible=bool(arm),
                        actual_arm_origin_guard_required=bool(arm),actual_arm_origin_guard_pending=bool(arm),
                        reasons=['OFF_diagnostic']if arm==0 else[],origin_checks={'controllers_off':True}),
                    timing=dict(decision_slot=2040,baseline_slot=2040,observe_slots=2000),
                    provenance=dict(job_id=job,batch_id='oldbatch',physical_plan_sha256=physical,
                        source_archive_sha256=archive,scoring_origin_id=oid,scoring_origin_job_id=off,
                        worker_id=worker,seed=i,after_kpis_used=False),cohort_id='cohort'+key,
                    independence_group='old-independence'+job))
        context=dict(frame=dict(slot=2040,states={'rcp:p':{'value':0}}),preview=dict(slot=2040,apps=[]),
            options=dict(sourceIdentity=oid,graphContextSha256='oldgraph',arbitration={'mode':'uniform'},
                history=[dict(frame={'slot':2000},preview={'slot':2000},source_identity=oid,graph_context_sha256='oldgraph')]))
        data=dict(stage='predeployment_inputs',after_values_included=False,cases=cases,
            graphs={origin['scene']:{'version':'original'}},contexts={'context':context})
        old_sources[key]={c['case_id']:dict(run_path='old/'+c['provenance']['job_id'],
            seal_path='old-checkpoint/'+c['provenance']['job_id'],expected_origin_id=oid,
            requires_actual_origin_guard=bool(c['app_ids']),ranges=deepcopy(c['ranges']),
            before_window={'start':40,'end':2040},after_window={'start':2040,'end':4040},
            **{k:deepcopy(c[k])for k in ('protocol','candidate','provenance','independence_group')})for c in cases}
        templates[key]={}
        refs={}
        for variant in (False,True):
            directory=root/('profile'if variant else'original')/key
            d=deepcopy(data)
            if variant:
                d['graphs'][origin['scene']]['version']='profile'
                d['contexts']['context']['options']['graphContextSha256']='profilegraph'
                d['contexts']['context']['options']['history'][0]['graph_context_sha256']='profilegraph'
            inp=put(directory/'inputs.json',d)
            labels=put(directory/'label-sources.json',dict(sources=old_sources[key],after_values_included=False))
            score_refs={}
            for method in ({gate.PROFILE_METHOD}if variant else gate.ORIGINAL_METHODS):
                score_refs[method]=put(directory/(method+'.json'),dict(module_sha256=module_pins))
            seal=dict(schema='oran.prospective-score-seal.v1',origin_key=key,
                input_path=inp['path'],input_sha256=inp['sha256'],scores=score_refs,label_sources_sha256=labels['sha256'])
            refs[variant]=put(directory/'seal.json',seal);templates[key][variant]=d
        blocks.append(dict(origin_key=key,**refs[False]))
        parent_origins.append(dict(origin_key=key,original_seal=refs[False],profile_seal=refs[True]))
    original_set=put(root/'predictions-v2'/'six-block-score-seal.json',dict(blocks=blocks))
    monkeypatch.setattr(gate,'ORIGINAL_SET_SHA256',original_set['sha256'])
    put(root/'precontrol-admission.json',dict(schema='oran.precontrol-admission.v1',
        physical_plan_sha256=physical,after_values_used=False,origins=parent_origins))
    plan=dict(schema='oran.quantification-supplement-plan.v1',source_version='v2',
        physical_run_count=12,origins=roots,baseline_slot=2040,observe_slots=2000,total_slots=4440,
        heldout_after_kpis_used=False,original_plan_or_seals_changed=False,
        apps={k:'xapp:'+k for k in 'ABCDEF'},group_id_prefix='study-supplement-arm:',
        parent_physical_plan_sha256=physical,source_sha256=source,source_archive_sha256=archive,
        original_score_set=original_set)
    plan['arms']=[dict(id=a,app_ids=[plan['apps'][c]for c in a])for a in('BCE','BCDE')]
    plan['plan_sha256']=digest(plan);pin=plan['plan_sha256'];put(artifact/'supplement-plan.json',plan)
    runner=artifact/'runner.py';runner.write_text('# synthetic independent runner')
    runner_pin=gate.sha(runner)
    publication=dict(supplement_plan_sha256=pin,parent_physical_plan_sha256=physical,
        source_sha256=source,source_archive_sha256=archive,runner_sha256=runner_pin,original_source_modified=False)
    put(artifact/'source-publication.json',publication)
    schedule=assignment(plan,pin);jobs=[]
    for worker,cases in schedule.items():
        put(artifact/'workers'/worker/'plan.json',dict(worker_id=worker,supplement=plan,
            supplement_plan_sha256=pin,runner_sha256=runner_pin,cases=cases))
        jobs.extend({**c,'worker_id':worker}for c in cases)
    physical_cases=[{**j, 'run_path':str(artifact/'workers'/j['worker_id']/'jobs'/j['id']/'data/run'),
        'remote_run_path':'/dev/shm/oran_quantification_20261122/source-v2/outputs/prospective_supplement/jobs/'+j['id']+'/data/run',
        'source_sha256':source,'source_archive_sha256':archive,'runner_sha256':runner_pin}for j in jobs]
    put(artifact/'cases-manifest.json',dict(schema='oran.prospective-supplement-cases.v1',source_version='v2',
        supplement_plan_sha256=pin,parent_physical_plan_sha256=physical,after_values_used=False,cases=physical_cases))
    admissions=[];new_refs={};new_inputs={};new_seals={}
    for origin in roots:
        key=origin['origin_key'];method_inputs={};score_refs={};label_refs={}
        for variant in (False,True):
            parent=templates[key][variant];d=deepcopy(parent);d['cases']=[]
            for job in [j for j in jobs if j['origin_key']==key]:
                for old in [c for c in parent['cases']if not c['app_ids']]:
                    c=deepcopy(old);c.update(case_id=key+':'+job['id']+':'+old['intent_id'],
                        group_id=job['group_id'],app_ids=job['app_ids'],candidate={'pool':'primary','eligible':True},
                        independence_group=digest([job['app_ids'],c['cohort_id']]))
                    c['provenance'].update(job_id=job['id'],batch_id='newbatch',supplement_plan_sha256=pin,grid_partition='prospective_supplement')
                    c['protocol'].update(primary_validation_eligible=True,actual_arm_origin_guard_required=True,
                        actual_arm_origin_guard_pending=True,reasons=[])
                    d['cases'].append(c)
            directory=artifact/'predictions'/key/('profile'if variant else'original')
            inp=put(directory/'inputs.json',d);new_inputs[key,variant]=inp['path']
            sources={}
            for c in d['cases']:
                job=next(j for j in jobs if j['id']==c['provenance']['job_id'])
                old_id=f"{key}:{origin['off_job_id']}:{c['intent_id']}"
                source_meta=deepcopy(old_sources[key][old_id])
                run=artifact/'workers'/job['worker_id']/'jobs'/job['id']/'data/run'
                source_meta.update(run_path=str(run),seal_path=str(run.parents[3]/'checkpoints'/(job['id']+'.json')),
                    requires_actual_origin_guard=True,**{k:deepcopy(c[k])for k in ('protocol','candidate','provenance','independence_group')})
                sources[c['case_id']]=source_meta
            label_refs[variant]=put(directory/'label-sources.json',dict(sources=sources,after_values_included=False))
            for method in ({gate.PROFILE_METHOD}if variant else gate.ORIGINAL_METHODS):
                method_inputs[method]=inp
                rows=[dict(**{k:deepcopy(c[k])for k in ('case_id','intent_id','candidate','protocol','provenance','cohort_id','independence_group','scene')},
                    result={'groups':[dict(group_id=c['group_id'],app_ids=c['app_ids'],score=0)]})for c in d['cases']]
                score=dict(stage='scores_before_label_join',method=method,input_path=inp['path'],input_sha256=inp['sha256'],
                    scores=rows,module_sha256=module_pins,archived_modules={module_key:str(module)},after_values_used=False,
                    label_sources_sha256=label_refs[variant]['sha256'])
                score_refs[method]={**put(directory/(method+'.json'),score),'count':30}
        original_ref=next(p for p in parent_origins if p['origin_key']==key)
        main=method_inputs[next(iter(gate.ORIGINAL_METHODS))]
        seal=dict(schema='oran.prospective-supplement-score-seal.v1',source_version='v2',origin_key=key,
            supplement_plan_sha256=pin,parent_physical_plan_sha256=physical,cases=30,physical_arms=2,intents=15,
            source_sha256=source,source_archive_sha256=archive,cases_manifest=gate._ref(artifact/'cases-manifest.json'),
            after_values_used=False,nonempty_arm_started_before_scoring=False,gate_opened=False,algorithm_tuned_to_test_labels=False,
            base_seal=original_ref['original_seal'],profile_parent_seal=original_ref['profile_seal'],
            blocks=[dict(**{k:origin[k]for k in ('origin_key','origin_id','off_job_id')},arms=2,intents=15)],
            input_path=main['path'],input_sha256=main['sha256'],method_inputs=method_inputs,scores=score_refs,
            label_sources=label_refs[False],profile_label_sources=label_refs[True])
        ref=put(artifact/'predictions'/key/'seal.json',seal);new_refs[key]=ref;new_seals[key]=seal
        admissions.append({**{k:origin[k]for k in ('origin_key','origin_id','worker_id','off_job_id')},'prediction_seal':ref})
    certificate=dict(schema='oran.supplement-precontrol-admission.v1',source_version='v2',
        supplement_plan_sha256=pin,parent_physical_plan_sha256=physical,source_sha256=source,
        source_archive_sha256=archive,runner_sha256=runner_pin,after_values_used=False,origins=admissions)
    put(artifact/'precontrol-admission.json',certificate)
    return artifact,pin,certificate,new_seals,new_inputs


def reseal(artifact, certificate, key, seal):
    ref=put(artifact/'predictions'/key/'seal.json',seal)
    next(o for o in certificate['origins']if o['origin_key']==key)['prediction_seal']=ref
    put(artifact/'precontrol-admission.json',certificate)


def test_six_origins_four_methods_two_inputs_pass_without_opening_labels(tmp_path,monkeypatch):
    artifact,pin,_,_,_=fixture(tmp_path,monkeypatch)
    result=gate.validate_full_admission(artifact,pin)
    assert result['origins']==6 and set(result['methods'])==gate.METHODS
    assert result['runner_sha256']==gate.sha(artifact/'runner.py')


@pytest.mark.parametrize('mutation', ['range','normalization','frame','preview','history','off_protocol','job','origin','candidate','case_id'])
def test_repinned_input_cannot_change_frozen_geometry_or_arm_identity(tmp_path,monkeypatch,mutation):
    artifact,pin,certificate,seals,inputs=fixture(tmp_path,monkeypatch)
    key=next(iter(seals));path=Path(inputs[key,False]);data=gate.read(path);case=data['cases'][0]
    if mutation=='range':case['ranges'][0]['upper']=2
    elif mutation=='normalization':case['options']['normalization']['kpis'][0]['scale']=2
    elif mutation=='frame':data['contexts']['context']['frame']['slot']=2000
    elif mutation=='preview':data['contexts']['context']['preview']['slot']=2000
    elif mutation=='history':data['contexts']['context']['options']['history'][0]['frame']['slot']=1900
    elif mutation=='off_protocol':case['protocol'].update(primary_validation_eligible=False,reasons=['OFF_diagnostic'])
    elif mutation=='job':case['provenance']['job_id']='OFF0'
    elif mutation=='origin':case['provenance']['scoring_origin_id']='another_origin'
    elif mutation=='candidate':case['candidate']['pool']='off_diagnostic'
    else:case['case_id']='copied-original-case'
    ref=put(path,data);seal=seals[key];seal['input_sha256']=ref['sha256']
    for method in gate.ORIGINAL_METHODS:seal['method_inputs'][method]=ref
    reseal(artifact,certificate,key,seal)
    with pytest.raises(ValueError,match='Supplement'):gate.validate_full_admission(artifact,pin)


@pytest.mark.parametrize('mutation',['missing_method','module','score_case','score_metadata','result_arm','after_flag'])
def test_score_or_seal_tampering_is_fail_closed(tmp_path,monkeypatch,mutation):
    artifact,pin,certificate,seals,_=fixture(tmp_path,monkeypatch)
    key=next(iter(seals));seal=seals[key];method=next(iter(gate.ORIGINAL_METHODS))
    if mutation=='missing_method':seal['scores'].pop(method)
    else:
        ref=seal['scores'][method];score=gate.read(ref['path'])
        if mutation=='module':score['module_sha256']={'source/engine.js':'f'*64}
        elif mutation=='score_case':score['scores'][0]['case_id']='unregistered'
        elif mutation=='score_metadata':score['scores'][0]['candidate']['pool']='off_diagnostic'
        elif mutation=='result_arm':score['scores'][0]['result']['groups'][0]['app_ids']=[]
        else:score['after_values_used']=True
        seal['scores'][method]={**put(Path(ref['path']),score),'count':30}
    reseal(artifact,certificate,key,seal)
    with pytest.raises(ValueError,match='Supplement|preregistered'):gate.validate_full_admission(artifact,pin)


def test_explicit_unknown_score_is_retained_in_the_complete_grid(tmp_path,monkeypatch):
    artifact,pin,certificate,seals,_=fixture(tmp_path,monkeypatch)
    key=next(iter(seals));seal=seals[key];method=next(iter(gate.ORIGINAL_METHODS))
    ref=seal['scores'][method];score=gate.read(ref['path'])
    score['scores'][0]['result']=dict(status='unavailable',reason='synthetic_missing_gain',groups=[])
    seal['scores'][method]={**put(Path(ref['path']),score),'count':30}
    reseal(artifact,certificate,key,seal)
    assert gate.validate_full_admission(artifact,pin)['origins']==6


@pytest.mark.parametrize('mutation',['runner','source','worker_cases','physical_path','origin_certificate','old_set','outside_root'])
def test_publication_plan_paths_and_original_pins_are_checked(tmp_path,monkeypatch,mutation):
    artifact,pin,certificate,seals,_=fixture(tmp_path,monkeypatch)
    if mutation=='runner':(artifact/'runner.py').write_text('# changed')
    elif mutation=='source':certificate['source_sha256']='wrong';put(artifact/'precontrol-admission.json',certificate)
    elif mutation=='worker_cases':
        p=artifact/'workers/agx1/plan.json';d=gate.read(p);d['cases'][0]['app_ids']=[];put(p,d)
    elif mutation=='physical_path':
        p=artifact/'cases-manifest.json';d=gate.read(p);d['cases'][0]['run_path']=str(artifact.parent/'original/OFF/data/run');put(p,d)
    elif mutation=='origin_certificate':
        certificate['origins'][0]['origin_id']='wrong';put(artifact/'precontrol-admission.json',certificate)
    elif mutation=='old_set':
        (artifact.parent/'predictions-v2/six-block-score-seal.json').write_text('{}')
    else:
        key=next(iter(seals));ref=put(tmp_path/'outside.json',seals[key]);certificate['origins'][0]['prediction_seal']=ref
        put(artifact/'precontrol-admission.json',certificate)
    with pytest.raises(ValueError):gate.validate_full_admission(artifact,pin)


@pytest.mark.parametrize('mutation',['run_path','seal_path','expected_origin','before_window','guard'])
def test_repinned_planned_label_metadata_cannot_point_to_off_or_another_window(tmp_path,monkeypatch,mutation):
    artifact,pin,certificate,seals,_=fixture(tmp_path,monkeypatch)
    key=next(iter(seals));seal=seals[key];labels=gate.read(seal['label_sources']['path'])
    source=next(iter(labels['sources'].values()))
    if mutation=='run_path':source['run_path']='old/OFF'
    elif mutation=='seal_path':source['seal_path']='old-checkpoint/OFF'
    elif mutation=='expected_origin':source['expected_origin_id']='another-origin'
    elif mutation=='before_window':source['before_window']['end']=2000
    else:source['requires_actual_origin_guard']=False
    for ref_name in ('label_sources','profile_label_sources'):
        seal[ref_name]=put(Path(seal[ref_name]['path']),labels)
    for method in gate.METHODS:
        score_ref=seal['scores'][method];score=gate.read(score_ref['path'])
        score['label_sources_sha256']=seal['label_sources']['sha256']
        seal['scores'][method]={**put(Path(score_ref['path']),score),'count':30}
    reseal(artifact,certificate,key,seal)
    with pytest.raises(ValueError,match='planned label mapping'):gate.validate_full_admission(artifact,pin)
