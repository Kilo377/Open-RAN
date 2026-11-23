"""Synthetic132-checkpoint orchestration; no simulator or production labels."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from experiments import analyze_quantification_study as driver
from experiments import study_quantification as study
from test_study_evaluation_cache import fixture as one_run_fixture


def fixture(tmp_path,*,pending=False):
    template_specs,template_run,_=one_run_fixture(tmp_path/'template')
    templates=[(study.read(s['score_path']),study.read(study.read(s['score_path'])['input_path']),
        study.read(Path(study.read(s['score_path'])['input_path']).with_name('label-sources.json')))for s in template_specs]
    blocks=[];pending_seal=None
    for scene in('native_1x','steady_2x','burst_2x'):
        for seed in(20261201,20261202):
            origin=f'{scene}_s{seed}'
            for partition,n in driver.PARTITIONS.items():
                count=20 if partition=='original_matrix'else 2;jobs=[]
                for arm in range(count):
                    job=f'{partition}_{origin}_{arm}';run=tmp_path/'runs'/job/'data/run';shutil.copytree(template_run,run)
                    off=partition=='original_matrix'and arm==0
                    config=study.read(run/'config_snapshot.json');config['manifest']['rng_seed']=seed;study.write(run/'config_snapshot.json',config)
                    pins={str(p.relative_to(run.parents[1])):study.file_sha(p)for p in run.iterdir()if p.is_file()}
                    checkpoint=tmp_path/'checkpoints'/(job+'.json');seal=dict(completed=True,case={'id':job,'app_ids':[]if off else['a']},
                        physical_plan_sha256='physical',source_sha256='source',origin_id='origin',files_sha256=pins)
                    if pending and pending_seal is None and partition=='prospective_supplement':pending_seal=(checkpoint,seal)
                    else:study.write(checkpoint,seal)
                    jobs.append(dict(job=job,run=run,checkpoint=checkpoint,off=off))
                specs=[]
                for index,(score_template,input_template,source_template)in enumerate(templates):
                    method=driver.METHODS[index];folder=tmp_path/'predictions'/partition/origin/method;folder.mkdir(parents=True)
                    cases=[];sources={};items=[]
                    for arm,job in enumerate(jobs):
                        case=deepcopy(input_template['cases'][0]);ident=job['job']+':intent'
                        case.update(case_id=ident,group_id=f'g{arm}',app_ids=[]if job['off']else['a'],scene=scene,
                            independence_group=job['job'],cohort_id=origin)
                        case['provenance'].update(job_id=job['job'],seed=seed,grid_partition=partition)
                        case['protocol'].update(primary_validation_eligible=not job['off'],actual_arm_origin_guard_pending=not job['off'])
                        case['candidate']=dict(eligible=not job['off'],pool='off_diagnostic'if job['off']else'primary')
                        source=deepcopy(source_template['sources']['case']);source.update({k:deepcopy(case[k])for k in('ranges','intent_id','cohort_id','independence_group','provenance','scene','protocol','candidate')})
                        source.update(run_path=str(job['run']),seal_path=str(job['checkpoint']),requires_actual_origin_guard=not job['off'])
                        item={k:deepcopy(case[k])for k in('case_id','intent_id','cohort_id','independence_group','protocol','provenance','candidate','scene')}
                        group=deepcopy(score_template['scores'][0]['result']['groups'][0]);group.update(group_id=case['group_id'],app_ids=case['app_ids'])
                        item['result']={'groups':[group]};cases.append(case);sources[ident]=source;items.append(item)
                    inp=folder/'scoring-inputs.json';study.write(inp,dict(stage='predeployment_inputs',after_values_included=False,cases=cases))
                    source_path=folder/'label-sources.json';study.write(source_path,{'sources':sources})
                    score=deepcopy(score_template);score.update(input_path=str(inp),input_sha256=study.file_sha(inp),label_sources_sha256=study.file_sha(source_path),scores=items)
                    score_path=folder/'scores.json';study.write(score_path,score);seal_path=folder/'prediction-seal.json'
                    study.write(seal_path,dict(schema='oran.prospective-score-seal.v1',after_values_used=False,nonempty_arm_started_before_scoring=False,
                        algorithm_tuned_to_test_labels=False,gate_opened=False,input_path=str(inp),input_sha256=study.file_sha(inp),
                        scores={method:dict(path=str(score_path),sha256=study.file_sha(score_path),count=len(cases))}))
                    specs.append(dict(method=method,score_path=str(score_path),score_sha256=study.file_sha(score_path),seal_path=str(seal_path),seal_sha256=study.file_sha(seal_path)))
                blocks.append(dict(id=partition+':'+origin,partition=partition,origin_key=origin,methods=specs))
    inventory=tmp_path/'coverage.json';study.write(inventory,dict(schema='oran.before-action-coverage-inventory.v1',heldout_nonempty_after_data_read=False,
        input_file_sha256={},method='net_intent_obstruction_v1',summary={},score_reasons={},missing_projection_buckets=[]))
    sets={}
    for name in('original_matrix','profile_variant','prospective_supplement'):
        p=tmp_path/(name+'-set.json');study.write(p,{'after_values_used':False,'synthetic_fixture':True});sets[name]=driver._ref(p)
    manifest=dict(schema='oran.quantification-analysis-plan.v1',source_version='v2',expected_physical_runs=driver.PARTITIONS,
        expected_unique_seeds=[20261201,20261202],expected_scene_seed_blocks=6,methods=list(driver.METHODS),blocks=blocks,
        source_sets=sets,before_action_coverage_inventory=driver._ref(inventory),synthetic_test_fixture=True)
    manifest['manifest_sha256']=study.digest(manifest);path=tmp_path/'analysis-manifest.json';study.write(path,manifest)
    return path,pending_seal


def test_default_final_refuses_one_pending_checkpoint_without_writing_output(tmp_path):
    path,_=fixture(tmp_path,pending=True);out=tmp_path/'final'
    result=driver.run_analysis(path,out,tmp_path/'cache')
    assert result['stage']=='waiting_for_completed_physical_matrix'
    assert result['status']['verified_physical_runs']==131 and not out.exists()
    assert result['status']['unique_seed_count']==2 and result['status']['scene_seed_blocks']==6


def test_interim_is_explicit_and_partition_manifests_keep_original_primary_separate(tmp_path,monkeypatch):
    path,_=fixture(tmp_path,pending=True);captured=[]
    def render(manifest,output):
        data=study.read(manifest);captured.append(data);output.mkdir(parents=True);(output/'index.html').write_text('Synthetic rendering fixture')
        study.write(output/'report-summary.json',{'study_partition':data['study_partition'],'analysis_status':data['analysis_status']})
    monkeypatch.setattr(driver,'build_report',render)
    result=driver.run_analysis(path,tmp_path/'interim-v1',tmp_path/'cache',interim=True)
    assert result['analysis_status']=='interim'and result['status']['verified_physical_runs']==131
    assert [len(m['methods'][0]['blocks'])for m in captured]==[6,6,12]
    assert [m['study_partition']for m in captured]==['original_matrix','prospective_supplement','combined_diagnostic']
    assert all('No method conclusion'in m['status_note']for m in captured)
    assert 'same two seeds'in captured[1]['population_note']
    assert result['original_primary_results_reported_separately']is True


def test_complete_one_command_writes_three_scientific_reports_without_extra_seeds(tmp_path):
    pytest.importorskip('matplotlib');path,_=fixture(tmp_path)
    result=driver.run_analysis(path,tmp_path/'final-v1',tmp_path/'cache')
    assert result['analysis_status']=='final'and result['status']['verified_physical_runs']==132
    reports={r['partition']:study.read(r['summary']['path'])for r in result['reports']}
    assert reports['original_matrix']['availability'][0]['unique_primary_physical_runs']==114
    assert reports['prospective_supplement']['availability'][0]['unique_primary_physical_runs']==12
    for report in reports.values():
        assert report['comparison']['matched_primary'][driver.METHODS[0]]['unique_seed_count']==2
        assert all(Path(p).is_file()for formats in report['figures'].values()for p in formats.values())
    assert (tmp_path/'final-v1/index.html').is_file()
