"""Real prediction producer and strict supplement reader share one contract.

All network values here are analytic fixtures, not run outcomes. No simulator
or raw AFTER trace is accessed.
"""
from copy import deepcopy
import io
from pathlib import Path
import tarfile

import pytest

from experiments import study_quantification as study
from experiments import study_supplement as supplement
from experiments import supplement_admission as admission
from experiments.quantification_supplement import assignment
from smo.influence_units import align_graph


def fixture(tmp_path):
    from test_smo_goal_obstruction import fixture as physics_fixture
    root=tmp_path/'v2';artifact=root/'supplement';artifact.mkdir(parents=True)
    graph,frame,catalog,preview,_=physics_fixture();graph=align_graph(graph)
    graph['nodes'][0].update(key='k',label='Power',scope='All UEs')
    graph['nodes']+=[dict(id=x,kind='xapp',target_kpis={'k':'maximize'})for x in('a','b','c','d')]
    frame.update(slot=2040,time_ms=1020);preview['slot']=2040
    for app in preview['apps']:app['mode']='cold_activation'
    preview['apps']+=[dict(app_id=x,mode='cold_activation',status='proposed',actions=[])for x in('c','d')]
    origins=[]
    for scene in('native_1x','steady_2x','burst_2x'):
        for seed in(20261201,20261202):
            key=f'{scene}_s{seed}';oid='origin:'+key;off='OFF:'+key
            proof=root/'workers/agx1/origins'/(key+'.json');shadow=proof.with_name(key+'-before-preview.json')
            study.write(proof,{'origin_id':oid});study.write(shadow,dict(origin_id=oid,after_kpis_used=False))
            origins.append(dict(origin_key=key,scene=scene,seed=seed,worker_id='agx1'if seed==20261201 else'agx2',origin_id=oid,off_job_id=off,
                origin_proof={'path':str(proof),'sha256':study.file_sha(proof)},before_preview={'path':str(shadow),'sha256':study.file_sha(shadow)}))
    origin=origins[0];key=origin['origin_key'];basepath=root/'predictions-v2'/key/'prediction-seal.json';basepath.parent.mkdir(parents=True)
    archive=root/'source.tgz'
    with tarfile.open(archive,'w:gz')as tar:
        for name in('smo/graph_combinations.py','smo/xapp_capabilities.py','smo/direct_action_filter.py'):
            payload=(study.ROOT/name).read_bytes();item=tarfile.TarInfo(name);item.size=len(payload);tar.addfile(item,io.BytesIO(payload))
    archivepin=study.file_sha(archive);prefix='prefix'
    prov=dict(job_id=origin['off_job_id'],batch_id='prospective:'+key,seed=origin['seed'],worker_id=origin['worker_id'],
        configuration_sha256='configuration',prefix_exogenous_sha256=prefix,physical_plan_sha256='parent',source_archive_sha256=archivepin,
        scoring_origin_id=origin['origin_id'],scoring_origin_job_id=origin['off_job_id'],after_kpis_used=False)
    protocol=dict(name='prospective_certified_OFF_prefix',primary_validation_eligible=False,actual_arm_origin_guard_required=False,
        actual_arm_origin_guard_pending=False,origin_checks={'exact_baseline':True},reasons=['OFF_diagnostic'])
    cases=[];sources={};graphpin=study.digest(graph)
    for arm in range(20):
        for intent in range(15):
            ident=f'{key}:original{arm}:i{intent}';p=deepcopy(prov)
            if arm:p['job_id']=f'oldjob{arm}'
            case=dict(case_id=ident,intent_id=f'i{intent}',group_id=f'arm{arm}',app_ids=[]if arm==0 else['a'],scene=origin['scene'],graph_id=origin['scene'],
                frame=deepcopy(frame),preview=deepcopy(preview),ranges=deepcopy(catalog['ranges']),protocol=deepcopy(protocol),
                options=dict(method='goal_obstruction_v2',normalization=study.normalization(catalog['ranges'],frame),
                    sourceIdentity='origin',graphContextSha256=graphpin,arbitration={'mode':'uniform_random'},
                    history=[dict(frame=deepcopy(frame),preview=deepcopy(preview),source_identity='origin',graph_context_sha256=graphpin)]),
                timing=dict(baseline_slot=2040,observe_slots=2000),provenance=p,candidate={'pool':'off_diagnostic','eligible':False},
                cohort_id='cohort',independence_group=f'oldindependent{arm}')
            cases.append(case);sources[ident]=dict(run_path=str(root/'oldjobs'/p['job_id']/'data/run'),
                seal_path=str(root/'oldchecks'/(p['job_id']+'.json')),expected_origin_id=origin['origin_id'],requires_actual_origin_guard=False,
                ranges=case['ranges'],protocol=case['protocol'],candidate=case['candidate'],provenance=p,independence_group=case['independence_group'],
                before_window={'window_start_slot':40,'slot':2040,'window_slots':2000},after_window={'window_start_slot':2040,'slot':4040,'window_slots':2000})
    block=dict(origin_key=key,origin_id=origin['origin_id'],off_job_id=origin['off_job_id'],arms=20,intents=15,after_values_used=False)
    original=study.compact_contexts(dict(stage='predeployment_inputs',graphs={origin['scene']:graph},cases=cases,prospective_blocks=[block],
        after_values_included=False,declared_experimental_intents=[{'id':f'i{i}'}for i in range(15)]))
    inp=basepath.with_name('scoring-inputs.json');study.write(inp,original);study.write(inp.with_name('label-sources.json'),{'sources':sources})
    scores={}
    for method in supplement.METHODS[:3]:
        dest=basepath.with_name('scores-'+method+'.json')
        study.score_cases(inp,dest,method=method,module=None if method==supplement.METHODS[0]else study.ROOT/'smo/web/intent_obstruction.js',
            engine='SmoGoalObstruction'if method==supplement.METHODS[0]else'SmoIntentObstruction')
        scores[method]=dict(path=str(dest),sha256=study.file_sha(dest),count=300)
    study.freeze_pipeline(basepath.parent/'pipeline-sources')
    base=dict(schema='oran.prospective-score-seal.v1',origin_key=key,blocks=[block],physical_arms=20,intents=15,cases=300,
        input_path=str(inp),input_sha256=study.file_sha(inp),label_sources_sha256=study.file_sha(inp.with_name('label-sources.json')),
        scores=scores,after_values_used=False,nonempty_arm_started_before_scoring=False,gate_opened=False,algorithm_tuned_to_test_labels=False)
    study.write(basepath,base);originalset=root/'predictions-v2/six-block-score-seal.json'
    study.write(originalset,{'blocks':[dict(origin_key=o['origin_key'],path=str(basepath),sha256=study.file_sha(basepath))for o in origins]})
    plan=dict(schema='oran.quantification-supplement-plan.v1',parent_physical_plan_sha256='parent',source_version='v2',source_sha256='source',
        source_archive_sha256=archivepin,physical_run_count=12,baseline_slot=2040,observe_slots=2000,total_slots=4440,intents=15,
        heldout_after_kpis_used=False,original_plan_or_seals_changed=False,methods=list(supplement.METHODS),
        apps={'A':'z','B':'a','C':'b','D':'c','E':'d','F':'f'},arms=[dict(id='BCE',app_ids=['a','b','d']),dict(id='BCDE',app_ids=['a','b','c','d'])],
        origins=origins,group_id_prefix='study-supplement-arm:',original_score_set={'path':str(originalset),'sha256':study.file_sha(originalset)})
    plan['plan_sha256']=study.digest(plan);planpath=artifact/'supplement-plan.json';study.write(planpath,plan)
    jobs=[]
    for worker,items in assignment(plan,plan['plan_sha256']).items():
        for job in items:jobs.append(dict(**job,worker_id=worker,source_sha256=plan['source_sha256'],source_archive_sha256=archivepin,
            run_path=str(artifact/'workers'/worker/'jobs'/job['id']/'data/run')))
    manifest=artifact/'cases-manifest.json';study.write(manifest,dict(schema='oran.prospective-supplement-cases.v1',
        supplement_plan_sha256=plan['plan_sha256'],parent_physical_plan_sha256='parent',source_version='v2',cases=jobs,after_values_used=False))
    profileplan=root/'profile-plan.json';p={'extra_variant':{'id':supplement.METHODS[-1]},'validation_seeds_excluded':[origin['seed']]};p['plan_sha256']=study.digest(p);study.write(profileplan,p)
    graphpath=root/'profile-graph.json';study.write(graphpath,graph)
    profileparent=root/'predictions-v3'/key;study.seal_graph_variant(basepath,{origin['scene']:graphpath},profileparent,profile_plan_path=profileplan)
    return root,planpath,manifest,basepath,profileparent/'prediction-seal.json',origin,jobs


def test_actual_supplement_producer_preserves_off_geometry_and_passes_strict_gate(tmp_path):
    root,planpath,manifest,base,profile,origin,jobs=fixture(tmp_path);output=root/'supplement-predictions'/origin['origin_key']
    originalpin=study.file_sha(base);seal=supplement.seal_supplement_origin(base,profile,planpath,manifest,output,origin_key=origin['origin_key'])
    references=lambda p:{'path':str(p),'sha256':study.file_sha(p)}
    ids=admission._seal(references(output/'prediction-seal.json'),root.resolve(),origin,
        [j for j in jobs if j['origin_key']==origin['origin_key']],study.read(planpath),references(base),references(profile))
    assert len(ids)==30 and set(seal['scores'])==set(supplement.METHODS)
    assert seal['blocks'][0]['arms']==2 and seal['physical_arms']==2
    original=study.read(study.read(base)['input_path']);off=next(c for c in original['cases']if not c['app_ids'])
    data=study.read(seal['input_path']);new=study.materialize_case(data,data['cases'][0]);old=study.materialize_case(original,off)
    assert new['frame']==old['frame']and new['preview']==old['preview']and new['ranges']==old['ranges']
    assert new['options']==old['options']and new['timing']==old['timing']
    assert new['protocol']['primary_validation_eligible']is True and 'OFF_diagnostic'not in new['protocol']['reasons']
    assert new['candidate']['pool']!='off_diagnostic'
    for source in study.read(output/'label-sources.json')['sources'].values():
        assert source['requires_actual_origin_guard']is True and '/supplement/workers/'in source['run_path']
        assert '/checkpoints/'in source['seal_path']
    assert study.file_sha(base)==originalpin


def test_supplement_rejects_started_future_arm_before_writing_inputs(tmp_path):
    root,planpath,manifest,base,profile,origin,jobs=fixture(tmp_path)
    job=next(j for j in jobs if j['origin_key']==origin['origin_key']);study.write(Path(job['run_path']).parent/'request.json',{})
    output=root/'supplement-predictions'/origin['origin_key']
    with pytest.raises(ValueError,match='started_before_predictions'):
        supplement.prepare_origin_inputs(base,planpath,manifest,output,origin_key=origin['origin_key'])
    assert not output.exists()
