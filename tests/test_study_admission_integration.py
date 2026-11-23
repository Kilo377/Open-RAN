"""Actual variant-seal producer interoperates with the strict admission reader.

The 20×15 inputs are an analytic test grid, not simulated measurements. Both
the production JS scorer and the real wrapper/validator execute unchanged.
"""
from copy import deepcopy

from experiments import precontrol_admission as admission
from experiments import study_quantification as study
from smo.influence_units import align_graph


def test_real_graph_variant_seal_passes_strict_precontrol_validator(tmp_path):
    from test_smo_goal_obstruction import fixture
    graph,frame,catalog,preview,_=fixture();graph=align_graph(graph)
    root=tmp_path/'v2';original=root/'predictions-v2/test_origin';original.mkdir(parents=True)
    expected=dict(origin_key='test_origin',origin_id='origin',off_job_id='OFF',worker_id='agx1',seed=2,
        physical_plan_sha256='plan',source_archive_sha256='archive')
    provenance=dict(physical_plan_sha256='plan',source_archive_sha256='archive',scoring_origin_id='origin',
        scoring_origin_job_id='OFF',worker_id='agx1',seed=2,after_kpis_used=False)
    cases=[];sources={}
    for arm in range(20):
        for intent in range(15):
            ident=f'test_origin:arm{arm}:i{intent}'
            cases.append(dict(case_id=ident,intent_id=f'i{intent}',group_id=f'arm{arm}',app_ids=['a']if arm else[],
                graph_id='native',frame=deepcopy(frame),preview=deepcopy(preview),ranges=deepcopy(catalog['ranges']),
                options=dict(sourceIdentity='origin',graphContextSha256=study.digest(graph),arbitration={'mode':'uniform_random'},history=[]),
                protocol={'name':'synthetic_test_grid'},timing={},provenance=deepcopy(provenance),
                candidate={'eligible':bool(arm),'pool':'primary'if arm else'off_diagnostic'},cohort_id='cohort',independence_group=f'arm{arm}'))
            sources[ident]=dict(run_path=str(root/'workers/agx1/jobs'/str(arm)/'data/run'))
    data=study.compact_contexts(dict(stage='predeployment_inputs',after_values_included=False,graphs={'native':graph},cases=cases))
    inp=original/'scoring-inputs.json';study.write(inp,data);study.write(original/'label-sources.json',{'sources':sources})
    original_score=original/'scores-goal_obstruction_v2.json'
    study.score_cases(inp,original_score)
    base=dict(schema='oran.prospective-score-seal.v1',origin_key='test_origin',blocks=[{k:expected[k]for k in('origin_key','origin_id','off_job_id')}],
        input_path=str(inp),input_sha256=study.file_sha(inp),label_sources_sha256=study.file_sha(original/'label-sources.json'),
        scores={'goal_obstruction_v2':dict(path=str(original_score),sha256=study.file_sha(original_score),count=300)},
        after_values_used=False,nonempty_arm_started_before_scoring=False,gate_opened=False,algorithm_tuned_to_test_labels=False,
        cases=300,physical_arms=20,intents=15)
    basepath=original/'prediction-seal.json';study.write(basepath,base);basepin=study.file_sha(basepath);inputpin=study.file_sha(inp)
    graphpath=root/'overlay-graph.json';study.write(graphpath,graph)
    plan=dict(extra_variant={'id':'net_profile_gain_obstruction_v1'},validation_seeds_excluded=[2]);plan['plan_sha256']=study.digest(plan)
    planpath=root/'profile-plan.json';study.write(planpath,plan)
    output=root/'predictions-v3/test_origin'
    produced=study.seal_graph_variant(basepath,{'native':graphpath},output,profile_plan_path=planpath)
    sealpath=output/'prediction-seal.json';reference=dict(path=str(sealpath),sha256=study.file_sha(sealpath))
    identifiers=admission._seal(reference,root.resolve(),expected,{admission.PROFILE_METHOD})
    assert len(identifiers)==300
    assert produced['blocks']==base['blocks']
    assert produced['gate_opened']is False and produced['algorithm_tuned_to_test_labels']is False
    scored=study.read(produced['scores'][admission.PROFILE_METHOD]['path'])
    assert scored['method']==admission.PROFILE_METHOD
    assert scored['engine_method']==scored['dispatch_method']=='net_intent_obstruction_v1'
    assert study.file_sha(basepath)==basepin and study.file_sha(inp)==inputpin
