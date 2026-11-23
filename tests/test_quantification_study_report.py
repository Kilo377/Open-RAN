"""The publication report is tested with explicitly synthetic analysis fixtures."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments import report_quantification_study as report
from experiments import study_quantification as study


def fixture(tmp_path):
    methods=[]
    for method,risks in(('method_a',[0.,0.]),('method_b',[0.,.1])):
        base=tmp_path/method;base.mkdir();engine=base/'synthetic-engine.js';engine.write_text('// Synthetic fixture only.\n')
        inp=base/'inputs.json'
        rows=[];scores=[]
        for i in range(4):
            risk=risks[i]if i<2 else 0. if i==3 else None
            row=dict(case_id=str(i),intent_id='unsupported'if i==2 else'improve',cohort_id='c',independence_group=str(i),scene='native_1x',
                provenance={'seed':2,'job_id':str(i),'evaluation_partition':'synthetic_test_fixture'},
                protocol={'primary_validation_eligible':i!=3},candidate={'pool':'primary'if i<2 else'off_diagnostic'if i==3 else'not_candidate','eligible':i<2},
                risk=100*risk/(1+risk)if risk is not None else None,raw_risk=risk,risk_status='complete'if i<2 else'unavailable',
                risk_reason=None if i<2 else'not_candidate',label_status='complete'if i!=2 else'incomplete_requested_kpis',
                observed_metric=10. if i==0 else-10. if i==1 else-2. if i==3 else None,
                observed_deterioration=0. if i==0 else.1 if i==1 else.02 if i==3 else None,
                all_satisfied=False if i!=2 else None)
            if i==3:row['before_after']=[dict(kpi='network_goodput_mbps',label='Throughput',unit='Mbit/s',before={'value':10},after={'value':9.8})]
            rows.append(row);scores.append(dict(case_id=str(i),result={'groups':[dict(score=row['risk'],raw_obstruction_strength=risk)]}))
        study.write(inp,{'after_values_included':False,'synthetic_test_fixture':True,
            'cases':[dict(case_id=str(i),app_ids=[]if i==3 else['a'],timing={'baseline_slot':2040,'observe_slots':2000})for i in range(4)]})
        score=base/'scores.json';study.write(score,dict(stage='scores_before_label_join',after_values_used=False,method=method,
            input_path=str(inp),input_sha256=study.file_sha(inp),module_sha256={'fixture.js':study.file_sha(engine)},
            archived_modules={'fixture.js':str(engine)},scores=scores))
        seal=base/'prediction-seal.json';study.write(seal,dict(schema='oran.prospective-score-seal.v1',scores={method:{'sha256':study.file_sha(score)}},
            input_path=str(inp),input_sha256=study.file_sha(inp),after_values_used=False,nonempty_arm_started_before_scoring=False))
        evaluation=base/'evaluation.json';study.write(evaluation,dict(stage='labels_joined_after_scoring',method=method,rows=rows,
            scores_sha256=study.file_sha(score),scoring_input_sha256=study.file_sha(inp),observed_results_used_for_scoring=False,score_label_separation=True))
        methods.append(dict(id=method,blocks=[dict(score_path=str(score),seal_path=str(seal),evaluation_path=str(evaluation))]))
    manifest=tmp_path/'manifest.json';study.write(manifest,dict(schema='oran.quantification-report-inputs.v1',methods=methods,synthetic_test_fixture=True))
    return manifest,methods


def test_standalone_report_retains_undefined_strata_coverage_and_zero_risk_regression(tmp_path):
    pytest.importorskip('matplotlib');manifest,_=fixture(tmp_path)
    summary=report.build_report(manifest,tmp_path/'report')
    a=summary['availability'][0]
    assert a['numeric_primary_rows']==2 and a['zero_risk_unmet_rows']==2
    assert a['zero_risk_unmet_with_regression']==1 and a['zero_risk_unmet_without_regression']==1
    comparison=summary['comparison']
    assert comparison['macro_spearman_same_defined_strata']=={'method_a':None,'method_b':None}
    assert comparison['matched_primary']['method_a']['macro_pair_concordance']==.5
    page=(tmp_path/'report/index.html').read_text()
    assert 'data:image/png;base64,'in page and '<script'not in page
    assert 'unsupported|native_1x|2'in page and 'constant_risk'in page
    assert 'OFF natural drift'in page and '10.000 → 9.800 Mbit/s'in page
    drift=summary['OFF_natural_drift']
    assert drift['physical_blocks']==1 and drift['derived_intent_rows']==1
    assert drift['rows'][0]['actual_metric']==-2 and drift['rows'][0]['regression']is True
    assert drift['subtracted_from_candidate_evaluation']is False
    assert comparison['common_numeric_primary_cases']==2
    assert summary['raw_traces_read']is False
    assert all(Path(p).is_file()for formats in summary['figures'].values()for p in formats.values())


def test_bad_prediction_seal_cannot_open_evaluation_labels(tmp_path,monkeypatch):
    _,methods=fixture(tmp_path);spec=methods[0]['blocks'][0];seal=study.read(spec['seal_path']);seal['scores']={};study.write(spec['seal_path'],seal)
    real=report.read;opened=[]
    def tracked(path):
        opened.append(str(path))
        if str(path)==spec['evaluation_path']:pytest.fail('Unsealed predictions must not open Evaluation labels')
        return real(path)
    monkeypatch.setattr(report,'read',tracked)
    with pytest.raises(ValueError,match='does_not_certify'):report.load_sealed_block(spec)
    assert spec['evaluation_path']not in opened


def test_separate_evaluation_must_not_rewrite_prediction_value(tmp_path):
    _,methods=fixture(tmp_path);spec=methods[0]['blocks'][0];data=study.read(spec['evaluation_path']);data['rows'][0]['raw_risk']=.2;study.write(spec['evaluation_path'],data)
    with pytest.raises(ValueError,match='rewrote_frozen_prediction'):report.load_sealed_block(spec)


def test_unresolved_off_is_recognized_by_frozen_arm_not_candidate_pool(tmp_path):
    _,methods=fixture(tmp_path);spec=methods[0]['blocks'][0];evaluation=study.read(spec['evaluation_path'])
    off=evaluation['rows'][3];off['candidate']={'pool':'unresolved_intent','eligible':False}
    off.update(observed_metric=None,observed_deterioration=None,all_satisfied=None,label_status='intent_reference_unavailable')
    off['before_after']=[];study.write(spec['evaluation_path'],evaluation)
    loaded,_=report.load_sealed_block(spec)
    assert loaded['rows'][3]['diagnostic_arm']=='OFF'
    drift=report.natural_off_drift([loaded])
    assert drift['derived_intent_rows']==1 and drift['rows'][0]['actual_metric']is None


def test_before_coverage_reports_existing_relation_gains_without_inventing_ue_applicability(tmp_path):
    source=tmp_path/'BEFORE-input.json';study.write(source,{'only':'before'})
    path=tmp_path/'coverage.json';study.write(path,dict(schema='oran.before-action-coverage-inventory.v1',
        heldout_nonempty_after_data_read=False,method='net_intent_obstruction_v1',summary={'all_cases':3},score_reasons={'numeric':2},
        input_file_sha256={str(source):study.file_sha(source)},missing_projection_buckets=[dict(parameter='ue_scheduler_weights',kpi='network_goodput_mbps',
            all_case_count=1,eligible_primary_case_count=1,classification='unmeasured_member',explanation='UE8 is unmeasured.',
            unmeasured_members=['ue_8'],calibration_contexts=[{'members':['ue_0'],'estimate':1}])]))
    coverage=report.before_action_coverage(path)
    assert coverage['buckets'][0]['recorded_relation_gain_contexts']==1
    assert coverage['buckets'][0]['unmeasured_members']==['ue_8']
    assert coverage['baseline_graph_inventory']is True


def test_raw_natural_boundary_does_not_round_small_positive_risk_to_zero():
    rows=[dict(case_id=str(i),intent_id='i',scene='s',cohort_id='c',independence_group=str(i),provenance={'job_id':str(i),'seed':1},
        protocol={'primary_validation_eligible':True},candidate={'eligible':True},raw_risk=risk,observed_deterioration=actual,
        observed_metric=-actual*100,all_satisfied=False)for i,(risk,actual)in enumerate(((1e-15,0.),(0.,.1)))]
    value=dict(method='test',rows=rows,label_status_counts={'complete':2},all_arms_diagnostics={'macro_pair_concordance':None})
    assert report.availability(value)['zero_risk_unmet_rows']==1
    diagnostic=report.natural_boundary_diagnostic(value)
    assert diagnostic['counts']==dict(zero_risk_no_regression=0,false_safe=1,false_alarm=1,positive_risk_with_regression=0)
    assert diagnostic['rows'][0]['raw_risk']==1e-15
    assert diagnostic['rounding_applied']is False and diagnostic['threshold_fitted']is False


def test_supplement_report_uses_each_method_own_frozen_input_reference(tmp_path):
    _,methods=fixture(tmp_path);spec=methods[0]['blocks'][0]
    scores=study.read(spec['score_path']);seal=study.read(spec['seal_path'])
    seal.update(schema='oran.prospective-supplement-score-seal.v1',input_path='main-input-unused-for-this-method',input_sha256='different',
        method_inputs={scores['method']:dict(path=scores['input_path'],sha256=scores['input_sha256'])})
    study.write(spec['seal_path'],seal)
    loaded,_=report.load_sealed_block(spec)
    assert loaded['method']==scores['method']
    seal['method_inputs'][scores['method']]['sha256']='changed';study.write(spec['seal_path'],seal)
    with pytest.raises(ValueError,match='input_changed'):report.load_sealed_block(spec)


def test_forward_report_retains_declared_strata_without_future_jobs_and_protocol_hashes(tmp_path):
    pytest.importorskip('matplotlib');path,_=fixture(tmp_path);manifest=study.read(path)
    manifest.update(declared_strata=['no-future|steady_2x|3'],forward_protocol={'code_bundle':{'sha256':'synthetic'},'diagnostic_jobs_excluded':4})
    study.write(path,manifest);result=report.build_report(path,tmp_path/'forward')
    assert result['forward_protocol']==manifest['forward_protocol']
    page=(tmp_path/'forward/index.html').read_text()
    assert 'no-future|steady_2x|3'in page and 'no prospective physical jobs'in page


def test_exported_interim_badge_and_native_not_matched_scatter_are_presentation_only(tmp_path,monkeypatch):
    pytest.importorskip('matplotlib');from matplotlib.figure import Figure
    path,methods=fixture(tmp_path);spec=methods[1]['blocks'][0]
    # Method A has two native numeric rows; B has one. Their shared pool stays
    # one, independently of the diagnostic scatter's native availability.
    score=study.read(spec['score_path']);score['scores'][1]['result']['groups'][0].update(score=None,raw_obstruction_strength=None)
    study.write(spec['score_path'],score);seal=study.read(spec['seal_path']);seal['scores']['method_b']['sha256']=study.file_sha(spec['score_path']);study.write(spec['seal_path'],seal)
    evaluation=study.read(spec['evaluation_path']);evaluation['rows'][1].update(risk=None,raw_risk=None,risk_status='unavailable',risk_reason='synthetic_missing_gain')
    evaluation['scores_sha256']=study.file_sha(spec['score_path']);study.write(spec['evaluation_path'],evaluation)
    immutable={s[k]:study.file_sha(s[k])for m in methods for s in m['blocks']for k in('score_path','seal_path','evaluation_path')}
    captures=[];savefig=Figure.savefig
    def capture(figure,path,*args,**kwargs):
        captures.append((Path(path).suffix,[text.get_text()for text in figure.texts]));return savefig(figure,path,*args,**kwargs)
    monkeypatch.setattr(Figure,'savefig',capture)
    manifest=study.read(path);manifest['analysis_status']='interim';study.write(path,manifest)
    interim=report.build_report(path,tmp_path/'interim')
    assert len(captures)==9 and {suffix for suffix,_ in captures}=={'.png','.pdf','.svg'}
    assert all(any('INTERIM'in text for text in texts)for _,texts in captures)
    svg=(tmp_path/'interim/figures/risk-outcome-primary.svg').read_text()
    assert 'Native available cases (not matched) · n=2'in svg and 'Native available cases (not matched) · n=1'in svg
    assert interim['comparison']['common_numeric_primary_cases']==1
    captures.clear();manifest['analysis_status']='final';study.write(path,manifest)
    final=report.build_report(path,tmp_path/'final')
    assert len(captures)==9 and all(not any('INTERIM'in text for text in texts)for _,texts in captures)
    assert {k:v for k,v in final['comparison'].items()if k!='report_files_sha256'}=={k:v for k,v in interim['comparison'].items()if k!='report_files_sha256'}
    assert sorted(final['comparison']['report_files_sha256'].values())==sorted(interim['comparison']['report_files_sha256'].values())
    assert final['availability']==interim['availability']
    assert all(study.file_sha(p)==pin for p,pin in immutable.items())
