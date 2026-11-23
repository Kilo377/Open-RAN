"""Standalone, auditable AFTER analysis of sealed quantification predictions.

The manifest lists methods and one or more independently sealed scene/seed
blocks per method. Prediction code and input hashes are verified before the
separate joined Evaluation file is opened. No raw simulator trace is read.
"""
import argparse
import base64
from collections import Counter,defaultdict
from copy import deepcopy
import html
import json
import math
from pathlib import Path

from experiments.study_quantification import compare_evaluation_reports,file_sha,read,summarize,write
from experiments.report_study_guide import INTERPRETATION,interpretation_html

METHOD_LABELS={'goal_obstruction_v2':'Goal v2 (original G)','net_intent_obstruction_v1':'Net (original G)',
    'temporal_net_intent_obstruction_v1':'Temporal net (original G)','net_profile_gain_obstruction_v1':'Net (profile G)',
    'aggregation_aware_obstruction_v1':'Aggregation-aware (profile G)'}


def finite(value):
    return type(value)in(int,float)and math.isfinite(value)


def method_release_summary(reports):
    """Display archived implementation identities, without merging versions."""
    rows=[]
    for report in reports:
        modules=defaultdict(set)
        for block in report.get('sealed_source_blocks',[]):
            for path,pin in block['module_sha256'].items():modules[Path(path).name].add(pin)
        rows.append(dict(method=report['method'],modules_by_name={k:sorted(v)for k,v in sorted(modules.items())},
            app_sha256=sorted(modules.get('app.js',[])),goal_sha256=sorted(modules.get('goal_obstruction.js',[]))))
    return rows


def certified(row):
    return row.get('candidate',{}).get('eligible')and row.get('protocol',{}).get('primary_validation_eligible')


def complete(row):
    return finite(row.get('raw_risk'))and finite(row.get('observed_deterioration'))


def load_sealed_block(spec):
    """No Evaluation read is permitted before this prediction seal passes."""
    scores_path=Path(spec['score_path']);seal_path=Path(spec['seal_path'])
    scores=read(scores_path);seal=read(seal_path);pin=file_sha(scores_path)
    if scores.get('stage')!='scores_before_label_join'or scores.get('after_values_used')is not False:
        raise ValueError('sealed_predeployment_scores_required')
    input_ref=dict(path=seal.get('input_path'),sha256=seal.get('input_sha256'));predeployment=seal.get('nonempty_arm_started_before_scoring')is False
    if seal.get('schema')=='oran.prospective-score-seal.v1':
        allowed=seal.get('scores',{}).get(scores['method'],{}).get('sha256')==pin
        clean=seal.get('after_values_used')is False
    elif seal.get('schema')=='oran.prospective-supplement-score-seal.v1':
        allowed=seal.get('scores',{}).get(scores['method'],{}).get('sha256')==pin
        input_ref=seal.get('method_inputs',{}).get(scores['method'],{})
        clean=seal.get('after_values_used')is False
    elif seal.get('schema')=='oran.prospective-graph-variant-seal.v1':
        allowed=seal.get('score_sha256')==pin;clean=seal.get('observed_validation_after_kpis_used')is False
    elif seal.get('schema')=='oran.frozen-prediction-subset-view.v1':
        from experiments.study_evaluation_cache import verify_prediction
        verify_prediction(spec)
        allowed=seal.get('score',{}).get('sha256')==pin;clean=True;predeployment=True;input_ref=seal['input']
    else:allowed=False;clean=False
    if not allowed or not clean or not predeployment:
        raise ValueError('prediction_seal_does_not_certify_scores')
    if (not isinstance(input_ref.get('path'),str)or file_sha(scores['input_path'])!=scores['input_sha256']or input_ref.get('sha256')!=scores['input_sha256']
        or Path(input_ref.get('path','')).resolve()!=Path(scores['input_path']).resolve()):
        raise ValueError('sealed_prediction_input_changed')
    before_inputs=read(scores['input_path']);input_cases={r['case_id']:r for r in before_inputs['cases']}
    if not scores.get('archived_modules'):raise ValueError('archived_prediction_engines_required')
    for original,archive in scores.get('archived_modules',{}).items():
        if file_sha(archive)!=scores['module_sha256'][original]:raise ValueError('archived_prediction_engine_changed')
    evaluation=read(spec['evaluation_path'])
    if (evaluation.get('stage')!='labels_joined_after_scoring'or evaluation.get('scores_sha256')!=pin
        or evaluation.get('scoring_input_sha256')!=scores['input_sha256']
        or evaluation.get('observed_results_used_for_scoring')is not False
        or evaluation.get('score_label_separation')is not True):
        raise ValueError('separate_evaluation_does_not_match_frozen_predictions')
    by_id={r['case_id']:r for r in scores['scores']}
    if len(evaluation['rows'])!=len(by_id)or {r['case_id']for r in evaluation['rows']}!=set(by_id):
        raise ValueError('evaluation_case_grid_changed')
    for row in evaluation['rows']:
        group=(by_id[row['case_id']]['result'].get('groups')or[{}])[0]
        if row.get('risk')!=group.get('score')or row.get('raw_risk')!=group.get('raw_obstruction_strength'):
            raise ValueError('evaluation_rewrote_frozen_prediction:'+row['case_id'])
        case=input_cases[row['case_id']]
        # Arm identity comes from the frozen plan, including an OFF intent
        # whose missing reference prevented candidate-range resolution.
        row['diagnostic_arm']='OFF'if not case['app_ids']else'NONEMPTY'
        row['frozen_window_protocol']=deepcopy(case.get('timing',{}))
        if row['diagnostic_arm']=='OFF'and row.get('candidate',{}).get('eligible'):
            raise ValueError('OFF_control_was_incorrectly_promoted_to_candidate')
    expected=scores.get('graph_variant',{}).get('id',scores['method'])
    if evaluation['method']!=expected:raise ValueError('evaluation_method_identity_changed')
    return evaluation,dict(score_path=str(scores_path),score_sha256=pin,seal_path=str(seal_path),seal_sha256=file_sha(seal_path),
        evaluation_path=str(spec['evaluation_path']),evaluation_sha256=file_sha(spec['evaluation_path']),
        input_sha256=scores['input_sha256'],module_sha256=scores['module_sha256'])


def aggregate_method(spec,output):
    rows=[];pins=[];case_ids=set()
    fields=('case_id','intent_id','cohort_id','independence_group','provenance','protocol','candidate','scene',
        'risk','raw_risk','risk_status','risk_reason','reference_effect','reference_effect_score','label_status',
        'observed_metric','observed_deterioration','all_satisfied','diagnostic_arm','frozen_window_protocol')
    for block in spec['blocks']:
        report,pin=load_sealed_block(block)
        if report['method']!=spec['id']:raise ValueError('manifest_method_does_not_match_sealed_engine')
        if any(r['case_id']in case_ids for r in report['rows']):raise ValueError('duplicate_physical_block_case_in_method')
        case_ids.update(r['case_id']for r in report['rows']);pins.append(pin)
        for row in report['rows']:
            reduced={k:deepcopy(row[k])for k in fields if k in row}
            if row['diagnostic_arm']=='OFF':reduced['before_after']=deepcopy(row.get('before_after',[]))
            rows.append(reduced)
    primary=[r for r in rows if certified(r)]
    value=dict(stage='labels_joined_after_scoring',method=spec['id'],rows=rows,
        primary=summarize(primary),all_arms_diagnostics=summarize(rows),
        label_status_counts=dict(Counter(r['label_status']for r in rows)),
        observed_results_used_for_scoring=False,score_label_separation=True,sealed_source_blocks=pins)
    write(output,value);return value


def availability(report):
    rows=report['rows'];eligible=[r for r in rows if r.get('candidate',{}).get('eligible')];primary=[r for r in rows if certified(r)]
    numeric=[r for r in primary if complete(r)]
    # This diagnostic separates hard target attainment from directional
    # improvement. A zero risk is not a guarantee of a numeric target's reach.
    zero=[r for r in numeric if r['raw_risk']==0 and r.get('all_satisfied')is False]
    return dict(method=report['method'],all_case_rows=len(rows),preselected_rows=len(eligible),certified_primary_rows=len(primary),
        grid_partition_counts=dict(Counter(r['provenance'].get('grid_partition','original_matrix')for r in rows)),
        numeric_primary_rows=len(numeric),numeric_primary_coverage=len(numeric)/len(primary)if primary else None,
        unique_primary_physical_runs=len({r['provenance']['job_id']for r in primary}),
        unavailable_primary_reasons=dict(Counter(r.get('risk_reason')or r.get('label_status')or'unknown'for r in primary if not complete(r))),
        zero_risk_unmet_rows=len(zero),zero_risk_unmet_with_regression=sum(r['observed_deterioration']>0 for r in zero),
        zero_risk_unmet_without_regression=sum(r['observed_deterioration']==0 for r in zero),
        zero_risk_unmet_case_ids=[r['case_id']for r in zero],
        diagnostic_numeric_rows=sum(complete(r)for r in rows),
        diagnostic_macro_concordance=report['all_arms_diagnostics']['macro_pair_concordance'],label_status_counts=report['label_status_counts'])


def natural_off_drift(reports):
    """OFF is diagnostic only; never subtract it or alter any primary row."""
    rows={}
    for report in reports:
        for row in report['rows']:
            if row.get('diagnostic_arm')!='OFF':continue
            key=(row.get('scene'),row['provenance'].get('seed'),row['provenance']['job_id'],row['intent_id'])
            value=dict(scene=key[0],seed=key[1],physical_run_id=key[2],intent_id=key[3],
                frozen_window_protocol=deepcopy(row.get('frozen_window_protocol',{})),
                actual_metric=row.get('observed_metric'),observed_deterioration=row.get('observed_deterioration'),
                regression=(row['observed_deterioration']>0 if finite(row.get('observed_deterioration'))else None),
                all_satisfied=row.get('all_satisfied'),label_status=row.get('label_status'),
                before_after=deepcopy(row.get('before_after',[])))
            if key in rows:
                previous={k:v for k,v in rows[key].items()if k!='methods_observed'}
                if value!=previous:raise ValueError('OFF_natural_drift_labels_differ_between_methods')
                rows[key]['methods_observed'].append(report['method'])
            else:rows[key]={**value,'methods_observed':[report['method']]}
    values=[rows[k]for k in sorted(rows,key=lambda x:tuple(str(v)for v in x))]
    return dict(schema='oran.OFF-natural-drift-diagnostic.v1',rows=values,
        physical_blocks=len({(r['scene'],r['seed'],r['physical_run_id'])for r in values}),derived_intent_rows=len(values),
        regression_rows=sum(r['regression']is True for r in values),unmet_rows=sum(r['all_satisfied']is False for r in values),
        primary_evaluation_adjusted=False,subtracted_from_candidate_evaluation=False,used_for_prediction=False,
        interpretation='Actual BEFORE/AFTER Evaluation includes natural evolution and stochastic drift. OFF is a separate diagnostic, not a difference-in-differences correction. Multiple intents on one OFF run are not independent repetitions.')


def natural_boundary_diagnostic(report):
    """Unfitted raw D>0 × observed D>0 table, separate from ranking statistics."""
    counts=dict(zero_risk_no_regression=0,false_safe=0,false_alarm=0,positive_risk_with_regression=0)
    values=[];missing=0;strata={}
    for row in report['rows']:
        if not certified(row):continue
        if not complete(row):missing+=1;continue
        risk,actual=row['raw_risk'],row['observed_deterioration']
        if risk<0 or actual<0:raise ValueError('natural_obstruction_boundary_requires_nonnegative_D')
        category=('positive_risk_with_regression'if actual>0 else'false_alarm')if risk>0 else('false_safe'if actual>0 else'zero_risk_no_regression')
        counts[category]+=1
        key='|'.join((row['intent_id'],row.get('scene','unknown'),str(row['provenance'].get('seed'))))
        strata.setdefault(key,{k:0 for k in counts})[category]+=1
        values.append(dict(case_id=row['case_id'],intent_id=row['intent_id'],scene=row.get('scene'),seed=row['provenance'].get('seed'),
            raw_risk=risk,actual_metric=row.get('observed_metric'),observed_deterioration=actual,
            all_satisfied=row.get('all_satisfied'),category=category))
    return dict(method=report['method'],counts=counts,numeric_primary_rows=len(values),unavailable_primary_rows=missing,
        per_intent_scene_seed=strata,rows=values,
        prediction_boundary='raw_obstruction_strength > 0',outcome_boundary='observed_deterioration > 0 (raw Evaluation < 0)',
        rounding_applied=False,threshold_fitted=False,primary_statistics_modified=False,
        interpretation='A descriptive natural-boundary table, not a trained classifier or target-attainment guarantee. Exact positive values stay positive, including floating-point-small values; rounding is reserved for preregistered ranking ties.')


def before_action_coverage(path):
    inventory=read(path)
    if inventory.get('schema')!='oran.before-action-coverage-inventory.v1'or inventory.get('heldout_nonempty_after_data_read')is not False:
        raise ValueError('BEFORE_only_coverage_inventory_required')
    for source,pin in inventory['input_file_sha256'].items():
        if file_sha(source)!=pin:raise ValueError('BEFORE_coverage_source_changed')
    buckets=[{k:deepcopy(row.get(k))for k in('parameter','kpi','all_case_count','eligible_primary_case_count','classification','explanation','unmeasured_members')}
        |{'recorded_relation_gain_contexts':len(row.get('calibration_contexts',[]))}for row in inventory['missing_projection_buckets']]
    return dict(path=str(path),sha256=file_sha(path),method=inventory['method'],summary=inventory['summary'],buckets=buckets,
        score_reasons=inventory['score_reasons'],baseline_graph_inventory=True,
        interpretation='A measured parameter–KPI relation does not imply that its gain applies to the proposed entity, profile, coordinate or operating context. Missing action-specific gains remain unknown; neither0 nor another UE gain is substituted.')


def scientific_figures(reports,comparison,output,*,declared_strata=(),synthetic=False,analysis_status=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.dpi':180,'pdf.fonttype':42,'svg.fonttype':'none'})
    output=Path(output);output.mkdir(parents=True,exist_ok=True);paths={}
    def save(fig,name):
        badges=(['INTERIM']if analysis_status=='interim'else[])+(['SYNTHETIC PREVIEW · test fixtures, not network results']if synthetic else[])
        if badges:fig.suptitle(' · '.join(badges),fontsize=11,y=1.02)
        paths[name]={}
        for suffix in('png','pdf','svg'):
            p=output/(name+'.'+suffix);fig.savefig(p,bbox_inches='tight');paths[name][suffix]=str(p)
        plt.close(fig)
    n=len(reports);fig,axes=plt.subplots(math.ceil(n/2),2,figsize=(10,3.5*math.ceil(n/2)),squeeze=False)
    colors={'native_1x':'#176b75','steady_2x':'#a46b26','burst_2x':'#8c5696'}
    for ax,report in zip(axes.flat,reports):
        rows=[r for r in report['rows']if certified(r)and complete(r)]
        scenes=sorted({r.get('scene','unknown')for r in rows})
        for scene in scenes:
            group=[r for r in rows if r.get('scene','unknown')==scene]
            ax.scatter([r['risk']for r in group],[100*r['observed_deterioration']for r in group],s=15,alpha=.55,
                color=colors.get(scene,'#667078'),label=scene.replace('_',' '))
        ax.set(title=METHOD_LABELS.get(report['method'],report['method'])+'\n'+f'Native available cases (not matched) · n={len(rows)}',xlabel='Risk R (dimensionless, 0–100)',ylabel='100 D_obs (normalized points)')
        ax.axhline(0,color='#919899',lw=.6);ax.set_xlim(-2,102)
        if scenes:ax.legend(loc='best',frameon=False,fontsize=7)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    fig.tight_layout();save(fig,'risk-outcome-primary')
    summary=[availability(r)for r in reports];fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    names=[r['method']for r in reports];x=np.arange(n)
    vals=[r['numeric_primary_coverage']for r in summary]
    axes[0].bar(x,[v or 0 for v in vals],color='#c2d8df',edgecolor='#53676d',linewidth=.6)
    axes[0].set(ylim=(0,1.1),ylabel='Numeric scores / certified eligible cases',title='Prediction coverage')
    conc=[comparison['matched_primary'][m]['macro_pair_concordance']for m in names]
    axes[1].bar(x,[v or 0 for v in conc],color='#edd6ba',edgecolor='#816b4d',linewidth=.6)
    axes[1].axhline(.5,color='#888',ls='--',lw=.8);axes[1].set(ylim=(0,1.1),ylabel='Macro pair concordance (ties = 0.5)',title='Same cases; equal intent × scene × seed weight')
    for ax,values in zip(axes,(vals,conc)):
        ax.set_xticks(x,[METHOD_LABELS.get(m,m).replace(' (','\n(')for m in names],fontsize=7)
        for i,value in enumerate(values):ax.text(i,(value or 0)+.025,'NA'if value is None else f'{value:.3f}',ha='center',fontsize=8)
    fig.tight_layout();save(fig,'method-summary')
    # Every requested intent/scene/seed stratum stays in this figure, including
    # no eligible candidates, unavailable gains and constant risk/target.
    strata=sorted(set(declared_strata)|{'|'.join((r['intent_id'],r.get('scene','unknown'),str(r.get('provenance',{}).get('seed'))))for report in reports for r in report['rows']})
    values=np.full((len(strata),n),np.nan)
    for j,report in enumerate(reports):
        table=report['primary']['per_intent_scene_seed']
        for i,key in enumerate(strata):
            value=table.get(key,{}).get('spearman')
            if value is not None:values[i,j]=value
    fig,ax=plt.subplots(figsize=(max(8,n*2.2),max(4,len(strata)*.22)))
    cmap=plt.get_cmap('RdBu_r').with_extremes(bad='#e6e9e9')
    im=ax.imshow(np.ma.masked_invalid(values),vmin=-1,vmax=1,cmap=cmap,aspect='auto')
    ax.set_xticks(range(n),[METHOD_LABELS.get(m,m).replace(' (','\n(')for m in names],fontsize=7);ax.set_yticks(range(len(strata)),strata,fontsize=6)
    ax.set_title('All requested strata: Spearman correlation; grey = undefined or unavailable')
    for i in range(len(strata)):
        for j in range(n):ax.text(j,i,'—'if np.isnan(values[i,j])else f'{values[i,j]:.2f}',ha='center',va='center',fontsize=6)
    fig.colorbar(im,ax=ax,shrink=.4,label='Descriptive within-stratum Spearman');fig.tight_layout();save(fig,'all-strata-correlations')
    return paths


def build_report(manifest_path,output):
    manifest=read(manifest_path);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if manifest.get('schema')!='oran.quantification-report-inputs.v1':raise ValueError('report_manifest_schema_required')
    if len(manifest.get('methods',[]))<2 or any(not m.get('blocks')for m in manifest['methods']):
        raise ValueError('multiple_nonempty_method_reports_required')
    reports=[];aggregate_paths=[]
    for method in manifest['methods']:
        if not method.get('id')or any(c not in'abcdefghijklmnopqrstuvwxyz0123456789_-'for c in method['id']):raise ValueError('invalid_method_id')
        path=output/('joined-'+method['id']+'.json');reports.append(aggregate_method(method,path));aggregate_paths.append(path)
    comparison=compare_evaluation_reports(aggregate_paths,output/'method-comparison.json')
    drift=natural_off_drift(reports);write(output/'OFF-natural-drift.json',drift)
    coverage=before_action_coverage(manifest['before_action_coverage_inventory'])if manifest.get('before_action_coverage_inventory')else None
    boundaries=[natural_boundary_diagnostic(r)for r in reports]
    stats=[availability(r)for r in reports];figures=scientific_figures(reports,comparison,output/'figures',declared_strata=manifest.get('declared_strata',()),synthetic=manifest.get('synthetic_test_fixture',False),analysis_status=manifest.get('analysis_status'))
    releases=method_release_summary(reports)
    from experiments.report_execution_provenance import collect
    execution=collect(manifest.get('execution_label_sources',[]))
    if manifest.get('execution_label_sources'):
        expected_jobs={r['provenance']['job_id']for report in reports for r in report['rows']}
        if {r['job_id']for r in execution['rows']}!=expected_jobs:raise ValueError('execution_provenance_is_not_the_reported_physical_pool')
    summary=dict(schema='oran.quantification-study-report.v1',manifest_path=str(manifest_path),manifest_sha256=file_sha(manifest_path),
        study_partition=manifest.get('study_partition'),analysis_status=manifest.get('analysis_status'),status_note=manifest.get('status_note'),
        forward_protocol=manifest.get('forward_protocol'),
        interpretation_contract=INTERPRETATION,archived_method_releases=releases,execution_provenance=execution,
        availability=stats,comparison=comparison,figures=figures,OFF_natural_drift=drift,before_action_coverage=coverage,natural_boundary_diagnostics=boundaries,
        predeployment_scores_modified=False,raw_traces_read=False,confidence_intervals_claimed=False)
    write(output/'report-summary.json',summary)
    esc=lambda v:html.escape(str(v))
    fmt=lambda v:'—'if v is None else'0'if v==0 else f'{v:.3g}'if abs(v)<.001 or abs(v)>=1e6 else f'{v:.3f}'
    cells=lambda values:'<tr>'+''.join('<td>'+esc(v)+'</td>'for v in values)+'</tr>'
    body=''.join(cells([s['method'],s['preselected_rows'],s['certified_primary_rows'],s['numeric_primary_rows'],fmt(s['numeric_primary_coverage']),s['zero_risk_unmet_rows'],s['zero_risk_unmet_with_regression']])for s in stats)
    row=lambda m:comparison['matched_primary'][m]
    matches=''.join(cells([m,row(m)['rows'],fmt(row(m)['macro_pair_concordance']),fmt(comparison['macro_spearman_same_defined_strata'][m]),len(row(m)['valid_spearman_strata']),len(row(m)['undefined_spearman_strata'])])for m in comparison['methods'])
    diagnostic_rows=''.join(cells([s['method'],s['all_case_rows'],s['diagnostic_numeric_rows'],fmt(s['diagnostic_macro_concordance'])])for s in stats)
    images=''.join('<figure><img alt="'+esc(name)+'" src="data:image/png;base64,'+base64.b64encode(Path(paths['png']).read_bytes()).decode()+'"><figcaption>'+esc(name.replace('-',' '))+' · '+''.join('<a href="figures/'+esc(Path(paths[s]).name)+'">'+s.upper()+'</a> 'for s in('pdf','svg'))+'</figcaption></figure>'for name,paths in figures.items())
    details=''.join('<details><summary>'+esc(s['method'])+' · unavailable and zero-risk cases</summary><pre>'+esc(json.dumps(s,indent=2))+'</pre></details>'for s in stats)
    tables=[]
    for report in reports:
        rows=report['primary']['per_intent_scene_seed'];present={'|'.join((r['intent_id'],r.get('scene','unknown'),str(r.get('provenance',{}).get('seed'))))for r in report['rows']}
        requested=sorted(set(manifest.get('declared_strata',()))|present)
        entries=''.join(cells([key,rows.get(key,{}).get('rows',0),fmt(rows.get(key,{}).get('spearman')),fmt(rows.get(key,{}).get('pair_concordance')),rows.get(key,{}).get('spearman_unavailable_reason','no eligible candidates'if key in present else'no prospective physical jobs')])for key in requested)
        tables.append('<details><summary>'+esc(report['method'])+' · every requested stratum</summary><table><thead>'+cells(['Intent | scene | seed','Eligible rows','Spearman','Concordance','Undefined reason'])+'</thead><tbody>'+entries+'</tbody></table></details>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quantification study</title>
<style>body{font:15px/1.55 system-ui,sans-serif;color:#203338;background:#f3f6f5;margin:0}main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:29px;margin:0 0 10px}h2{font-size:19px;margin-top:30px}p{max-width:1000px;color:#485d62}table{border-collapse:collapse;width:100%;background:white;font-size:12px}td,th{padding:10px;text-align:left;border-bottom:1px solid #d7e0de}thead{font-weight:600;background:#e8efed}figure{margin:22px 0;padding:18px;background:white;border:1px solid #dbe4e1;border-radius:8px}img{max-width:100%;height:auto}figcaption{color:#617278;font-size:12px}a{color:#176b75}details{background:white;padding:14px;margin:12px 0;border:1px solid #dbe4e1;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}</style><main><h1>Quantification study</h1>
<p>Predictions use frozen predeployment state, policy proposals and knowledge-graph gains. Actual intent Evaluation is joined afterwards. All methods are descriptive comparisons; no deployment outcome trains or changes a score.</p>
<h2>Coverage and zero-risk failures</h2><p>Primary cases must pass the frozen candidate selector and common-origin protocol. A hard target can remain unmet despite beneficial progress: zero-risk unmet cases with actual regression are listed separately.</p><table><thead>'''+cells(['Method','Preselected','Certified primary','Numeric primary','Coverage','Zero-risk unmet','With regression'])+'</thead><tbody>'+body+'</tbody></table><h2>Matched method comparison</h2><p>Same numeric primary cases across every method. Macro statistics weight each intent × scene × seed stratum equally. Reassessing one physical run against several intents does not create independent experiments. Constant risks retain concordance ties; Spearman remains undefined.</p><table><thead>'+cells(['Method','Matched rows','Macro concordance','Same-defined macro ρ','Defined ρ strata','Undefined ρ strata'])+'</thead><tbody>'+matches+'</tbody></table><p>Common strata with defined Spearman in every method: '+str(len(comparison['same_defined_spearman_strata']))+'. All requested strata, including undefined and unavailable cases, remain below.</p>'+images+'<h2>Complete availability</h2>'+details+'<h2>Every requested stratum</h2>'+''.join(tables)+'<p><a href="report-summary.json">Full reproducible summary and artifact hashes</a> · <a href="method-comparison.json">Matched comparison JSON</a></p></main></html>'
    diagnostic_table='<h2>All-arm diagnostics</h2><p>Includes OFF controls, reserve and rejected/noncandidate arms. This population is reported separately and does not enter the primary ranking comparison.</p><table><thead>'+cells(['Method','All derived rows','Numeric score + label','Diagnostic macro concordance'])+'</thead><tbody>'+diagnostic_rows+'</tbody></table>'
    boundary_rows=''.join(cells([b['method'],b['counts']['zero_risk_no_regression'],b['counts']['false_safe'],b['counts']['false_alarm'],b['counts']['positive_risk_with_regression'],b['unavailable_primary_rows']])for b in boundaries)
    boundary_table='<h2>Raw natural-boundary diagnostic</h2><p>Certified primary cases only. Strict raw D &gt; 0 versus actual deterioration &gt; 0; no fitted threshold, rounding or change to the main statistics. “False safe” means zero predicted obstruction with actual regression; “false alarm” means positive obstruction without actual regression. These descriptive labels do not imply target attainment.</p><table><thead>'+cells(['Method','Zero / no regression','False safe','False alarm','Positive / regression','Unavailable'])+'</thead><tbody>'+boundary_rows+'</tbody></table>'
    off_blocks=defaultdict(list)
    for off in drift['rows']:off_blocks[(off['scene'],off['seed'])].append(off)
    off_tables=[]
    for (scene,seed),rows in sorted(off_blocks.items()):
        entries=[]
        for off in rows:
            def measurement(k):
                before,after=k.get('before',{}).get('value'),k.get('after',{}).get('value');unit=k.get('unit','')
                multiplier=100 if k['kpi']=='urllc_failure_fraction'and unit in('ratio','fraction')else 1
                shown='%'if multiplier==100 else unit
                show=lambda value:fmt(multiplier*value)if value is not None else'—'
                return str(k.get('label',k['kpi']))+': '+show(before)+' → '+show(after)+' '+str(shown)
            values='; '.join(measurement(k)for k in off['before_after'])
            yn=lambda x:'unavailable'if x is None else'yes'if x else'no'
            entries.append(cells([off['intent_id'],values or'unavailable',fmt(off['actual_metric']),yn(off['regression']),yn(off['all_satisfied']),off['label_status']]))
        protocols={(r['frozen_window_protocol'].get('baseline_slot'),r['frozen_window_protocol'].get('observe_slots'))for r in rows}
        window_note=''
        if len(protocols)==1:
            baseline,slots=next(iter(protocols))
            if type(baseline)is int and type(slots)is int:
                window_note=f'<p>BEFORE ({baseline-slots}, {baseline}]; AFTER ({baseline}, {baseline+slots}] slots. Same windows as candidate Evaluation.</p>'
        off_tables.append('<details><summary>'+esc(scene)+' · seed '+esc(seed)+'</summary>'+window_note+'<table><thead>'+cells(['Intent','Requested KPIs: BEFORE → AFTER','Actual Metric','Regression','Target attained','Status'])+'</thead><tbody>'+''.join(entries)+'</tbody></table></details>')
    off_section='<h2>OFF natural drift</h2><p>'+esc(drift['interpretation'])+'</p><p>'+str(drift['physical_blocks'])+' physical scene/seed blocks; '+str(drift['derived_intent_rows'])+' derived intent rows. OFF is outside the primary candidate pool. No candidate Metric or risk score is adjusted by these values.</p>'+''.join(off_tables)
    coverage_section=''
    if coverage:
        gain_rows=''.join(cells([b['parameter'],b['kpi'],b['recorded_relation_gain_contexts'],b['all_case_count'],b['eligible_primary_case_count'],b['explanation']])for b in coverage['buckets'])
        coverage_section='<h2>Relation gains versus action applicability</h2><p>Baseline graph inventory: '+esc(coverage['interpretation'])+'</p><table><thead>'+cells(['Parameter','KPI','Existing gain contexts','Unknown rows','Primary unknown','Why the proposed action is uncovered'])+'</thead><tbody>'+gain_rows+'</tbody></table>'
    page=page.replace('<h2>Complete availability</h2>',diagnostic_table+boundary_table+off_section+coverage_section+'<h2>Complete availability</h2>')
    release_rows=''.join(cells([METHOD_LABELS.get(r['method'],r['method']),', '.join(p[:12]for p in r['app_sha256'])or'not attached',
        ', '.join(p[:12]for p in r['goal_sha256'])or'not attached'])for r in releases)
    release_section='<h2>Frozen implementation versions</h2><p>'+esc(INTERPRETATION['versions'])+'</p><table><thead>'+cells(['Method','Archived app SHA-256','Archived projector SHA-256'])+'</thead><tbody>'+release_rows+'</tbody></table>'
    execution_rows=''.join(cells([r['job_id'],r['planned_worker']or'not recorded',r['actual_execution_host']or'not recorded',
        'completed'if r['checkpoint_completed']else'planned / pending',r['host_evidence'],
        r.get('execution_kind','original planned execution'),(r.get('original_protocol_runner_sha256')or'')[:12],(r.get('actual_execution_wrapper_sha256')or'')[:12]])for r in execution['rows'])
    execution_section='<details><summary>Execution provenance · '+str(execution['physical_jobs'])+' physical jobs</summary><p>'+esc(execution['interpretation'])+'</p><table><thead>'+cells(['Job','Logical / planned worker','Actual compute host','Checkpoint','Host evidence','Execution mode','Original runner SHA','Actual wrapper SHA'])+'</thead><tbody>'+execution_rows+'</tbody></table></details>'
    page=page.replace('<h2>Coverage and zero-risk failures</h2>',interpretation_html()+release_section+execution_section+'<h2>Coverage and zero-risk failures</h2>')
    title=manifest.get('title','Quantification study');page=page.replace('<title>Quantification study</title>','<title>'+esc(title)+'</title>').replace('<h1>Quantification study</h1>','<h1>'+esc(title)+'</h1>')
    notes=(['SYNTHETIC PREVIEW — numerical test fixtures, not network results.']if manifest.get('synthetic_test_fixture')else[])+[manifest.get('population_note'),manifest.get('status_note')]
    banner=''.join('<p><strong>'+esc(note)+'</strong></p>'for note in notes if note)
    page=page.replace('<h1>'+esc(title)+'</h1>','<h1>'+esc(title)+'</h1>'+banner)
    (output/'index.html').write_text(page);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=build_report(args.manifest,args.output)
    print(json.dumps(dict(html=str(args.output/'index.html'),methods=[x['method']for x in summary['availability']],raw_traces_read=False)))


if __name__=='__main__':main()
