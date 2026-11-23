"""Audited multi-point first-order fits with independent seed blocks."""
from pathlib import Path
from copy import deepcopy
import argparse
import csv
import json
import math
import statistics

from smo.identify import audit_case,load,write,sha,numeric,response_delta,paired_summary,fit_contrasts
from smo.contracts import KPIS,build_graph
from smo.population import SCOPE
from experiments.rcp_perturbation import vector


def verify_bridge(study,reference_dataset):
    study=Path(study);plan=load(study/'plan.json');old=load(reference_dataset)['cases']
    lookup={(r['seed'],r['arm']):r for r in old};results=[]
    bridge=[c for c in plan['cases']if c.get('role')=='bridge']
    if len(bridge)!=3:raise ValueError('Three native/scheduler bridge conditions are required')
    for c in bridge:
        new=audit_case(study,c,plan,population=True);ref=lookup[c['seed'],c['arm']];diff={}
        for k,v in new['metrics'].items():
            previous=ref['metrics'][k]
            same=(v is None and previous is None)or(type(v)in(int,float)and type(previous)in(int,float)and math.isclose(v,previous,rel_tol=1e-8,abs_tol=1e-10))
            if not same:diff[k]=dict(previous=previous,current=v)
        for k in('exogenous_arrivals_sha256','task_births_sha256'):
            if new['proof'][k]!=ref['proof'][k]:diff[k]='mismatch'
        if new['task_counts']!=ref['task_counts']:diff['task_counts']='mismatch'
        results.append(dict(case=c['name'],differences=diff,proof=new['proof']))
    return dict(source_sha256=plan['source_sha256'],passed=not any(r['differences']for r in results),
        metric_tolerance=dict(relative=1e-8,absolute=1e-10),conditions=results)


def line_fit(xs,ys):
    if len(xs)!=len(ys)or len(xs)<3 or len(set(xs))!=len(xs)or not all(numeric(v)for v in xs+ys):raise ValueError('At least three distinct finite paired grid points required')
    xm=statistics.mean(xs);ym=statistics.mean(ys)
    beta=math.fsum((x-xm)*(y-ym)for x,y in zip(xs,ys))/math.fsum((x-xm)**2 for x in xs)
    if response_delta(min(ys),max(ys))==0:beta=0.
    return dict(beta=beta,intercept=ym-beta*xm,mean=ym)


def fit_grid(g,seeds,lookup,kpi,data_sha256):
    raw=[dict(x=x,values=[dict(seed=s,value=lookup[s,arm]['metrics'][kpi])for s in seeds])for x,arm in zip(g['levels'],g['arms'])]
    kept=[r for r in raw if all(numeric(v['value'])for v in r['values'])]
    if len(kept)<4:return None
    xs=[r['x']for r in kept];ys={s:[r['values'][i]['value']for r in kept]for i,s in enumerate(seeds)}
    fits={s:line_fit(xs,ys[s])for s in seeds};summary=paired_summary([fits[s]['beta']for s in seeds]);beta=summary['estimate'];xm=statistics.mean(xs)
    residuals=[y-(fits[s]['mean']+beta*(x-xm))for s in seeds for x,y in zip(xs,ys[s])]
    sse=math.fsum(e*e for e in residuals);sst=math.fsum((y-fits[s]['mean'])**2 for s in seeds for y in ys[s])
    constant=all(response_delta(min(ys[s]),max(ys[s]))==0 for s in seeds)
    r2=None if constant else max(0.,1-sse/sst)if sst>0 else None
    rmse=math.sqrt(sse/len(residuals));cv=[]
    for i,x in enumerate(xs):
        trainx=[v for j,v in enumerate(xs)if j!=i]
        models={s:line_fit(trainx,[v for j,v in enumerate(ys[s])if j!=i])for s in seeds}
        slope=statistics.mean(m['beta']for m in models.values());center=statistics.mean(trainx)
        cv.extend(ys[s][i]-(models[s]['mean']+slope*(x-center))for s in seeds)
    cv_rmse=math.sqrt(statistics.mean(e*e for e in cv));span=statistics.mean(max(v)-min(v)for v in ys.values())
    limited=not constant and(r2 is None or r2<.7 or cv_rmse>.3*span)
    ci=summary['confidence_interval'];direction_resolved=ci is not None and(ci[0]>0 or ci[1]<0)
    assessment='constant_response_observed'if constant else'limited_first_order_approximation'if limited else'first_order_approximation_supported'
    reference_arm=g['arms'][g['levels'].index(g['reference'])]
    refs=[lookup[s,reference_arm]['metrics'][kpi]for s in seeds]
    points=[]
    for r in raw:
        valid=[v['value']for v in r['values']if numeric(v['value'])]
        points.append(dict(x=r['x'],mean=statistics.mean(valid)if valid else None,valid_seed_count=len(valid),
            used_in_fit=len(valid)==len(seeds),by_seed=r['values']))
    row=dict(parameter=g['parameter'],kpi=kpi,contrast_id=g['id'],kind='range_linear_fit',
        parameter_component=g['component'],scalar_component=g.get('scalar_component'),
        unit=KPIS[kpi][1]+' / '+g['x_unit'],x_unit=g['x_unit'],sample_count=len(seeds),independent_unit='seed_block',
        population_control_eligible=True,metric_scope=SCOPE,horizon_ms=600,scoring_start_ms=200,scoring_end_ms=700,scoring_window_ms=500,
        scope=g['scope'],context='Fixed '+g['mode']+'; all other RCPs held at their native values',
        method='Multi-point seed-blocked affine least squares; interval across independent seed slopes',
        design='controlled_perturbation',data_sha256=data_sha256,confidence_level=.95,
        sampled_domain=[min(g['levels']),max(g['levels'])],fit_domain=[min(xs),max(xs)],
        sampled_levels=g['levels'],fit_levels=xs,excluded_levels=[r['x']for r in raw if r not in kept],
        grid_point_count=len(g['levels']),fit_point_count=len(xs),observed_run_count=len(seeds)*len(g['levels']),
        reference_parameter=g['reference'],reference_kpi_mean=statistics.mean(refs)if all(numeric(v)for v in refs)else None,
        intercept_at_reference=statistics.mean(fits[s]['mean']+beta*(g['reference']-xm)for s in seeds),
        seed_fits=[dict(seed=s,**fits[s])for s in seeds],grid_points=points,
        fit_diagnostics=dict(within_seed_r2=r2,rmse=rmse,leave_level_out_rmse=cv_rmse,max_abs_residual=max(map(abs,residuals)),
            assessment=assessment,average_direction_resolved=direction_resolved,
            usable_for_linear_control=not limited and not constant and direction_resolved,
            heuristic_thresholds=dict(min_r2=.7,max_cv_rmse_over_within_seed_response_span=.3)),
        domain_note=g.get('domain_note'),
        interpretation='Range-wide average first-order approximation, not an exact derivative at every operating point. Missing outcomes stay missing; nonlinear residuals are retained.',**summary)
    return row


def analyze(studies,reuse_study,output):
    studies=[Path(s)for s in studies];reuse=Path(reuse_study);output=Path(output)
    plans=[load(s/'plan.json')for s in studies];old=load(reuse/'plan.json')
    if len({p['source_sha256']for p in plans})!=1:raise ValueError('Grid workers must execute the same source files')
    if any(p['grids']!=plans[0]['grids']or p['contrasts']!=plans[0]['contrasts']for p in plans):raise ValueError('Grid designs differ')
    cases=[];origin_cache={};origins=[];environments=[]
    for study,plan in zip(studies,plans):
        status=load(study/'study_status.json')
        if not status.get('completed')or status.get('failed'):raise ValueError('Incomplete grid worker')
        if plan.get('requires_bridge'):
            bridge=load(study/'bridge_verified.json')
            if not bridge.get('passed')or bridge['source_sha256']!=plan['source_sha256']:raise ValueError('Worker bridge not validated')
        if(study/'environment.json').exists():environments.append(dict(study=str(study),**load(study/'environment.json')))
        for case in plan['cases']:
            if case.get('role')=='bridge':continue
            row=audit_case(study,case,plan,population=True);row['origin']=dict(study=str(study),case=case['name'],source_sha256=plan['source_sha256'],reused=False)
            cases.append(row)
        for r in plan['reused']:
            if r['origin_plan_sha256']!=sha(reuse/'plan.json')or plan['reuse_source_sha256']!=old['source_sha256']:
                raise ValueError('Reused origin plan changed')
            c=r['origin_case'];key=c['name']
            if vector(old['specs'][c['arm']],perturbed=True)!=vector(plan['specs'][r['arm']],perturbed=True):
                raise ValueError('Reused control vectors differ from the grid point')
            if key not in origin_cache:origin_cache[key]=audit_case(reuse,c,old,population=True)
            row=deepcopy(origin_cache[key]);row['name']=r['name'];row['arm']=r['arm']
            row['origin']=dict(study=str(reuse),case=c['name'],source_sha256=old['source_sha256'],reused=True)
            cases.append(row)
        origins.append(dict(study=str(study),plan_sha256=sha(study/'plan.json'),source_sha256=plan['source_sha256'],
            dynamics_sha256=plan['dynamics_sha256'],retention=load(study/'retention-verification.json')))
    seeds=sorted({c['seed']for c in cases});lookup={(c['seed'],c['arm']):c for c in cases}
    if len(lookup)!=len(cases)or len(seeds)!=3:raise ValueError('This preregistered grid requires three complete unique seed blocks')
    expected={(s,a)for s in seeds for a in plans[0]['specs']}
    if set(lookup)!=expected:raise ValueError('Grid has missing or extra conditions')
    for seed in seeds:
        same=[c['proof']for c in cases if c['seed']==seed]
        for k in('exogenous_environment_sha256','cache_key','native_prefix_sha256','exogenous_arrivals_sha256','task_births_sha256'):
            if len({p[k]for p in same})!=1:raise ValueError('Paired grid environment differs: '+str(seed)+':'+k)
    dataset=dict(schema='oran.rcp-grid-measurements.v1',metric_scope=SCOPE,seeds=seeds,grids=plans[0]['grids'],cases=cases,
        origins=origins,reuse_plan_sha256=sha(reuse/'plan.json'),environments=environments,
        notes=['Each seed block stays on one hardware/software environment. Reused points have matching runtime dynamics and exact executed vectors.',
            'Energy sums all UEs; compute demand sums all task sources. Completed-task delay remains conditional on completion.',
            'At offload probability zero, offloaded-task latency is unavailable, never zero. Missing levels are excluded consistently across seeds.',
            'Range fits use one affine intercept per seed and a shared slope. Intervals use independent seed slopes, not slots/grid points as independent replications.'])
    output.mkdir(parents=True,exist_ok=True);write(output/'measurement-dataset.json',dataset);digest=sha(output/'measurement-dataset.json')
    estimates=[];unavailable=[]
    for g in plans[0]['grids']:
        for k in KPIS:
            row=fit_grid(g,seeds,lookup,k,digest)
            if row:estimates.append(row)
            else:unavailable.append(dict(contrast_id=g['id'],kpi=k,reason='Fewer than four complete multi-seed levels'))
    contrast_plan={**plans[0],'seeds':seeds}
    categorical=fit_contrasts(contrast_plan,cases,digest,population=True)
    for row in categorical:
        row['interpretation']='Three-seed paired categorical contrast, conditional on the recorded load, context and horizon; not a numerical derivative.'
    estimates+=categorical
    write(output/'all-effects.json',dict(schema='oran.intervention-edges.v1',metric_scope=SCOPE,estimates=estimates,unavailable=unavailable))
    candidates={(e['source'][4:],e['target'][4:])for e in build_graph()['edges']if e['kind']=='influences'}
    def resolved(r):
        ci=r.get('confidence_interval');return ci is not None and(ci[0]>0 or ci[1]<0)
    discovered={(r['parameter'],r['kpi'])for r in estimates if resolved(r)}-candidates
    published=[r for r in estimates if(r['parameter'],r['kpi'])in candidates|discovered]
    write(output/'influences.json',dict(schema='oran.intervention-edges.v1',metric_scope=SCOPE,estimates=published,
        measurement_dataset_sha256=digest,complete_matrix='all-effects.json',study='Multi-point seed-blocked range fits; inspect residuals and tested domains. Intervals are descriptive; exploratory links are not multiplicity-corrected.'))
    with(output/'measurements.csv').open('w',newline='')as f:
        fields=['case','seed','arm','reused']+list(KPIS)
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cases:w.writerow(dict(case=c['name'],seed=c['seed'],arm=c['arm'],reused=c['origin']['reused'],**c['metrics']))
    with(output/'sampled-response-points.csv').open('w',newline='')as f:
        fields=['parameter','contrast','kpi','x','x_unit','seed','kpi_value','used_in_fit']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in estimates:
            for point in r.get('grid_points',[]):
                for v in point['by_seed']:w.writerow(dict(parameter=r['parameter'],contrast=r['contrast_id'],kpi=r['kpi'],
                    x=point['x'],x_unit=r['x_unit'],seed=v['seed'],kpi_value=v['value'],used_in_fit=point['used_in_fit']))
    with(output/'response-coefficients.csv').open('w',newline='')as f:
        fields=['parameter','kpi','contrast','kind','estimate','ci95_low','ci95_high','unit','seeds','fit_levels','domain','r2','rmse','assessment']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in estimates:
            ci=r.get('confidence_interval')or[None,None];d=r.get('fit_diagnostics',{})
            w.writerow(dict(parameter=r['parameter'],kpi=r['kpi'],contrast=r['contrast_id'],kind=r['kind'],estimate=r['estimate'],
                ci95_low=ci[0],ci95_high=ci[1],unit=r['unit'],seeds=r['sample_count'],fit_levels=r.get('fit_point_count'),
                domain=json.dumps(r.get('fit_domain')),r2=d.get('within_seed_r2'),rmse=d.get('rmse'),assessment=d.get('assessment')))
    report=dict(independent_seeds=len(seeds),conditions_per_seed=len(plans[0]['specs']),population_observations=len(cases),
        reused_observations=sum(c['origin']['reused']for c in cases),new_grid_observations=sum(not c['origin']['reused']for c in cases),
        estimated_responses=len(estimates),unavailable=unavailable,published_estimates=len(published),
        measured_links=len({(r['parameter'],r['kpi'])for r in published}),
        limited_range_fits=sum(r.get('fit_diagnostics',{}).get('assessment')=='limited_first_order_approximation'for r in estimates),
        raw_files_verified=sum(len(c['proof']['files_sha256'])for c in cases),
        audited_scoring_slots=sum(c['proof']['verified_held_parameter_slots']for c in cases),measurements_sha256=digest)
    write(output/'identification-report.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--studies',type=Path,nargs='+',required=True);p.add_argument('--reuse',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(analyze(a.studies,a.reuse,a.output),indent=2))


if __name__=='__main__':main()
