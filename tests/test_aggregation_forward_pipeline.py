"""Synthetic metadata/engine pipeline; no real AFTER values or SSH access."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from experiments import aggregation_aware_study as agg
from experiments import forward_method_analysis as forward
from experiments import prospective_request_inventory as inventory
from experiments import study_quantification as study
from experiments.study_evaluation_cache import verify_prediction
from test_quantification_analysis_driver import fixture as analysis_fixture


def fixture(tmp_path):
    manifest_path,_=analysis_fixture(tmp_path/'analysis');manifest=study.read(manifest_path)
    for block in manifest['blocks']:
        for spec in block['methods']:
            score=study.read(spec['score_path']);inp=Path(score['input_path']);data=study.read(inp)
            sourcepath=inp.with_name('label-sources.json');source=study.read(sourcepath);cases=[];rows=[];sources={}
            oldrows={r['case_id']:r for r in score['scores']}
            for old in data['cases']:
                for i in range(15):
                    case=deepcopy(old);case['case_id']=old['case_id']+f':i{i}';case['intent_id']=f'i{i}'
                    case['provenance']['worker_id']='agx1'if case['provenance']['seed']==20261201 else'agx2'
                    row=deepcopy(oldrows[old['case_id']]);row.update(case_id=case['case_id'],intent_id=case['intent_id'],provenance=deepcopy(case['provenance']))
                    src=deepcopy(source['sources'][old['case_id']]);src.update(intent_id=case['intent_id'],provenance=deepcopy(case['provenance']))
                    cases.append(case);rows.append(row);sources[case['case_id']]=src
            data.update(cases=cases,graphs={'fixture':{'nodes':[],'edges':[]}})
            if spec['method']==agg.PROFILE_METHOD:data['graph_variant']={'id':agg.PROFILE_METHOD,'synthetic_fixture':True}
            for case in cases:case['graph_id']='fixture'
            study.write(inp,data);source['sources']=sources;study.write(sourcepath,source)
            score.update(scores=rows,input_sha256=study.file_sha(inp),label_sources_sha256=study.file_sha(sourcepath))
            for name in('app.js','goal_obstruction.js'):
                path=study.ROOT/'smo/web'/name;score['module_sha256'][str(path)]=study.file_sha(path);score['archived_modules'][str(path)]=str(path)
            study.write(spec['score_path'],score);seal=study.read(spec['seal_path']);seal['input_sha256']=study.file_sha(inp)
            seal['scores'][spec['method']].update(sha256=study.file_sha(spec['score_path']),count=len(cases));study.write(spec['seal_path'],seal)
            spec.update(score_sha256=study.file_sha(spec['score_path']),seal_sha256=study.file_sha(spec['seal_path']))
    manifest['manifest_sha256']=study.digest({k:v for k,v in manifest.items()if k!='manifest_sha256'});study.write(manifest_path,manifest)
    plan=dict(schema='oran.aggregation-aware-method-addendum.v1',method=agg.METHOD,scoring_parameters_fit_to_after_labels=False,additional_physical_runs=0)
    plan['plan_sha256']=study.digest(plan);planpath=tmp_path/'addendum.json';study.write(planpath,plan)
    root=tmp_path/'v2/fifth';ledger=agg.prepare_ledger(manifest_path,planpath,root);code=agg.freeze_bundle(root/'before-ledger.json',root/'engine')
    return root,ledger,code


def mocked_scanner(ledger):
    calls=[];jobs=ledger['jobs']
    def scan(v2,path,*,code_bundle):
        after=bool(calls);now=time.time();workers=[]
        for wid in('agx1','agx2'):
            rows=[]
            for i,job in enumerate(jobs):
                if job['worker_id']!=wid:continue
                exists=i==0 or(after and i==2);partials=['preserved']if i==1 else[]
                rows.append(dict(job,exists=exists,historical_partial_directories=partials,worker_status_memberships=[],
                    worker_status_known=i!=3,ever_started_evidence=bool(exists or partials)))
            worker=dict(worker_id=wid,jobs=rows,source_sha256='synthetic');worker['response_sha256']=inventory._digest(worker);workers.append(worker)
        value=dict(schema='oran.prospective-request-scan.v1',local_started_unix_s=now,local_finished_unix_s=now,
            frozen_code_bundle=code_bundle,jobs_grid_sha256='same132',expected_source_pins={'agx1':'synthetic','agx2':'synthetic'},
            inventory_module_sha256='same_module',workers=workers,code_bundle_persisted_before_scan=True,after_kpi_values_read=False,remote_writes_performed=False)
        value['scan_sha256']=inventory._digest(value);study.write(path,value);calls.append(path)
        return dict(path=str(path),sha256=study.file_sha(path),scan_sha256=value['scan_sha256'])
    return scan,calls


def test_fifth_ledger_keeps_all1980_cases_and_same_graph_options_and_method_id(tmp_path):
    root,ledger,code=fixture(tmp_path)
    assert len(ledger['cases'])==1980 and len(ledger['jobs'])==132
    for block in ledger['blocks']:
        data=study.read(block['input']['path']);parent=study.read(block['profile_input']['path'])
        assert 'graph_variant'not in data and data['gain_model_provenance']['profile_graph_variant']['id']==agg.PROFILE_METHOD
        assert data['graphs']==parent['graphs']
        assert all(study.materialize_case(data,a)==study.materialize_case(parent,b)for a,b in zip(data['cases'],parent['cases']))
        assert Path(block['label_sources']['path']).read_bytes()==Path(block['profile_input']['path']).with_name('label-sources.json').read_bytes()
    first=ledger['blocks'][0];result=study.score_cases(first['input']['path'],root/'test-score.json',method=agg.METHOD,module=code['code_bundle']['path'],engine=agg.ENGINE)
    assert result['method']==agg.METHOD and 'graph_variant'not in result
    assert len(result['scores'])==300


def test_real_frozen_engine_and_remote_scan_shapes_seal_future_scope_then_filter_without_labels(tmp_path,monkeypatch):
    root,ledger,code=fixture(tmp_path);scanner,calls=mocked_scanner(ledger);monkeypatch.setattr(inventory,'scan_requests',scanner)
    result=agg.score_between_scans(root/'before-ledger.json',root/'engine/code-freeze.json',root.parent,root/'attempt')
    eligibility=study.read(result['forward_eligibility']['path'])
    assert len(calls)==2 and result['all_case_count']==1980 and eligibility['eligible_physical_jobs']==128
    assert all(ledger['jobs'][i]['job_id']not in eligibility['eligible_job_ids']for i in range(4))
    checked=forward.protocol(root/'attempt/method-seal.json');block=result['blocks'][0]
    spec=dict(score_path=block['score']['path'],seal_path=str(root/'attempt/method-seal.json'))
    view=forward.write_subset_view(spec,root/'attempt/method-seal.json',root/'view',verified_protocol=checked)
    verified=verify_prediction(view)
    assert all(s['provenance']['job_id']in checked['cohort']for s in verified['sources'].values())
    assert len(verified['case_ids'])<300
    parent=study.read(spec['score_path']);view_score=study.read(view['score_path'])
    assert view_score['scores']==[r for r in parent['scores']if r['case_id']in verified['case_ids']]
    # Altering a retained frozen risk, even with a refreshed local view pin,
    # cannot turn a view into a newly fitted prediction.
    view_score['scores'][0]['result']['groups'][0]['score']=73;study.write(view['score_path'],view_score)
    metadata=study.read(view['seal_path']);metadata['score']['sha256']=study.file_sha(view['score_path']);study.write(view['seal_path'],metadata)
    with pytest.raises(ValueError,match='changed_frozen_risks'):verify_prediction(view)


def test_forward_driver_preserves_both_preregistered_pools_and_never_joins_started_jobs(tmp_path,monkeypatch):
    from experiments import report_quantification_study as report
    root,ledger,_=fixture(tmp_path);scanner,_=mocked_scanner(ledger);monkeypatch.setattr(inventory,'scan_requests',scanner)
    result=agg.score_between_scans(root/'before-ledger.json',root/'engine/code-freeze.json',root.parent,root/'attempt')
    plan=dict(schema='oran.aggregation-forward-analysis-plan.v1',addendum=result['addendum'],before_ledger=result['ledger'],
        comparisons=[dict(id='joint_five_matched',methods=list(forward.OLD_METHODS)+[agg.METHOD],rule='All five common numeric cases.'),
            dict(id='profileG_pairwise_ablation',methods=[agg.PROFILE_METHOD,agg.METHOD],rule='Only methods4/5 common numeric cases.')],
        partitions=['original_matrix','prospective_supplement','combined_diagnostic'],after_KPIs_used_to_choose_analysis=False,new_physical_runs=0)
    plan['plan_sha256']=study.digest(plan);plan_path=root/'forward-plan.json';study.write(plan_path,plan)
    # Evidence pins include Python source bytes, which must be hashed as bytes
    # rather than parsed as JSON alongside runtime/protocol metadata.
    archived_driver=root/'archived-driver.py';archived_driver.write_text('print("Synthetic archival source")\n')
    runtime_path=root/'runtime.json';study.write(runtime_path,{'synthetic':True})
    study.write(root/'attempt/evidence-binding.json',dict(method_seal=forward.ref(root/'attempt/method-seal.json'),
        local_forward_analysis=forward.ref(plan_path),root_forward_analysis=forward.ref(plan_path),
        runtime=forward.ref(runtime_path),driver_source=forward.ref(archived_driver)))
    manifest=forward.prepare_views(root/'attempt/method-seal.json',plan_path,root/'views');path=root/'views/forward-analysis-inputs.json'
    assert manifest['eligible_physical_jobs']==128 and manifest['future_case_count']==1920 and len(manifest['declared_strata'])==90
    before_loader=study._observed_windows
    def forbidden(_):raise AssertionError('No actual windows may be parsed during eligibility/preflight.')
    monkeypatch.setattr(study,'_observed_windows',forbidden)
    status=forward.run_forward_analysis(path,root/'unwritten',root/'cache',check_only=True)
    assert status['stage']=='ready_for_forward_report'and not(root/'unwritten').exists()
    # Default final admission waits for every eligible checkpoint without parsing
    # labels. Previously-started diagnostic jobs are outside this condition.
    spec=manifest['blocks'][0]['methods'][0];verified=verify_prediction(spec)
    checkpoint=Path(next(iter(verified['sources'].values()))['seal_path']);saved=checkpoint.read_bytes();checkpoint.unlink()
    pending=forward.run_forward_analysis(path,root/'pending',root/'cache')
    assert pending['stage']=='waiting_for_forward_physical_cohort'and not(root/'pending').exists()
    checkpoint.write_bytes(saved);monkeypatch.setattr(study,'_observed_windows',before_loader)
    rendered=[]
    def render(input_path,output):
        data=study.read(input_path);rendered.append(data);output.mkdir(parents=True)
        (output/'index.html').write_text('Synthetic forward report')
        study.write(output/'report-summary.json',dict(forward_protocol=data['forward_protocol']))
    monkeypatch.setattr(report,'build_report',render)
    joined=forward.run_forward_analysis(path,root/'complete',root/'cache')
    assert joined['status']['verified_physical_jobs']==128 and len(joined['reports'])==6
    assert [len(m['methods'])for m in rendered]==[5,5,5,2,2,2]
    assert [len(m['methods'][0]['blocks'])for m in rendered]==[6,6,12,6,6,12]
    assert all(len(m['declared_strata'])==90 for m in rendered)
    assert all(m['forward_protocol']['scope']['diagnostic_jobs_excluded']==4 for m in rendered)
    eligible=set(manifest['eligible_job_ids'])
    for evaluation in(root/'complete/evaluations').rglob('evaluation-*.json'):
        assert all(row['provenance']['job_id']in eligible for row in study.read(evaluation)['rows'])
    # Each frozen method keeps its own risk values, while actual-label reuse is
    # exactly the unchanged direct join on the identical future view.
    selected=manifest['blocks'][0]['methods'][-1]
    direct_path=root/'direct.json';source=Path(study.read(selected['score_path'])['input_path']).with_name('label-sources.json')
    study.join_evaluation(selected['score_path'],source,direct_path);direct=study.read(direct_path)
    reused=study.read(root/'complete/evaluations'/manifest['blocks'][0]['partition']/manifest['blocks'][0]['origin_key']/'final'/('evaluation-'+agg.METHOD+'.json'))
    assert direct['method']==agg.METHOD and all(reused[k]==v for k,v in direct.items())
    assert(root/'complete/index.html').is_file()
