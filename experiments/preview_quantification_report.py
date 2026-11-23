"""Generate unmistakably synthetic report previews without opening study data."""
import argparse
from copy import deepcopy
from pathlib import Path

from experiments.study_quantification import write,file_sha
from experiments.report_quantification_study import build_report
from experiments.report_study_guide import write_landing_page


METHODS=('goal_obstruction_v2','net_intent_obstruction_v1','temporal_net_intent_obstruction_v1',
    'net_profile_gain_obstruction_v1','aggregation_aware_obstruction_v1')


def generate(output):
    output=Path(output)
    if output.exists():raise ValueError('synthetic_preview_requires_a_new_directory')
    output.mkdir(parents=True);specs=[]
    for index,method in enumerate(METHODS):
        folder=output/'fixtures'/method;folder.mkdir(parents=True);older=index<3
        archive=output/'synthetic-code'/('earlier-projector'if older else'later-projector');archive.mkdir(parents=True,exist_ok=True)
        modules={};copies={}
        for name in('app.js','goal_obstruction.js'):
            path=archive/name
            if not path.exists():path.write_text('// SYNTHETIC TEST FIXTURE ONLY · '+archive.name+' · '+name+'\n')
            modules[name]=file_sha(path);copies[name]=str(path)
        rows=[];scores=[];cases=[]
        for i in range(4):
            raw=0. if index==0 or i in(0,3)else .05*index if i==1 else None
            if index==2 and i==1:raw=None
            intent='reduce_urllc_failure_fraction'if i==0 else'throughput'if i==1 else'unsupported_reference'if i==2 else'OFF_rare_failure'
            metric=-.01 if i in(0,3)else-10. if i==1 else None
            row=dict(case_id=str(i),intent_id=intent,cohort_id='synthetic',independence_group=str(i),scene='synthetic_native',
                provenance={'seed':'synthetic','job_id':str(i),'evaluation_partition':'synthetic_test_fixture'},
                protocol={'primary_validation_eligible':i<2},candidate={'eligible':i<2,'pool':'primary'if i<2 else'off_diagnostic'if i==3 else'not_candidate'},
                risk=100*raw/(1+raw)if raw is not None else None,raw_risk=raw,risk_status='complete'if raw is not None else'unavailable',
                risk_reason=None if raw is not None else'synthetic_missing_evidence',label_status='complete'if i!=2 else'incomplete_requested_kpis',
                observed_metric=metric,observed_deterioration=max(0.,-metric/100)if metric is not None else None,all_satisfied=False if i!=2 else None)
            if i==3:row['before_after']=[dict(kpi='urllc_failure_fraction',label='URLLC failure fraction',unit='ratio',before={'value':0.},after={'value':.0001})]
            rows.append(row);scores.append(dict(case_id=str(i),result={'groups':[dict(score=row['risk'],raw_obstruction_strength=raw)]}))
            cases.append(dict(case_id=str(i),app_ids=[]if i==3 else['synthetic-app'],timing={'baseline_slot':2040,'observe_slots':2000}))
        inp=folder/'scoring-inputs.json';write(inp,dict(after_values_included=False,synthetic_test_fixture=True,cases=cases,
            synthetic_gain_model='same synthetic G for methods four/five; no physical graph is evaluated'))
        score_path=folder/'scores.json';write(score_path,dict(stage='scores_before_label_join',after_values_used=False,method=method,
            input_path=str(inp),input_sha256=file_sha(inp),module_sha256=modules,archived_modules=copies,scores=scores))
        seal_path=folder/'prediction-seal.json';write(seal_path,dict(schema='oran.prospective-score-seal.v1',after_values_used=False,
            nonempty_arm_started_before_scoring=False,input_path=str(inp),input_sha256=file_sha(inp),scores={method:{'sha256':file_sha(score_path)}}))
        evaluation_path=folder/'evaluation.json';write(evaluation_path,dict(stage='labels_joined_after_scoring',method=method,rows=rows,
            scores_sha256=file_sha(score_path),scoring_input_sha256=file_sha(inp),observed_results_used_for_scoring=False,score_label_separation=True))
        specs.append(dict(id=method,blocks=[dict(score_path=str(score_path),seal_path=str(seal_path),evaluation_path=str(evaluation_path))]))
    pages=[]
    for comparison,selected in(('joint_five_matched',specs),('profileG_pairwise_ablation',specs[-2:])):
        path=output/'manifests'/(comparison+'.json');write(path,dict(schema='oran.quantification-report-inputs.v1',synthetic_test_fixture=True,
            methods=deepcopy(selected),title='SYNTHETIC PREVIEW · '+comparison.replace('_',' '),study_partition='synthetic_fixture',analysis_status='synthetic',
            declared_strata=['no-future|synthetic_other|synthetic'],
            population_note='All numbers and archives are analytic display fixtures. No simulator, AGX output, trained model or physical result is read.',
            status_note='Rare-ratio example: 0/10,000 → 1/10,000 interval events means 0% → 0.01%, a +0.01 percentage-point change. A relative percentage increase from zero is undefined.'))
        report=output/'reports'/comparison/'original_matrix';build_report(path,report)
        pages.append(dict(comparison=comparison,partition='original_matrix',html=str(report/'index.html'),
            label='Synthetic report example',description='Four deliberately synthetic rows exercise constant scores, ties, missing evidence and a rare URLLC event. These are not physical measurements.'))
    write_landing_page(output,title='SYNTHETIC PREVIEW · O-RAN report',status='DISPLAY FIXTURES ONLY · not a physical study, trained-model result or effect estimate',pages=pages,forward=True)
    return dict(index=str(output/'index.html'),reports=pages,synthetic_only=True,real_study_data_read=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(__import__('json').dumps(generate(args.output)))


if __name__=='__main__':main()
