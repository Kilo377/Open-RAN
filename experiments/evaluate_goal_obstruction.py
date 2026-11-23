"""Predeployment-only risk replay; measured outcomes are joined after scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from smo.contracts import build_graph,validate_influences
from smo.population import graph_view
from smo.runtime import records,replay
from smo.xapp_preview import PreviewStore
from smo.influence_units import align_graph, normalization
from smo.result_evaluation import enrich_batch

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(a,b):
    import math
    if len(a)<2:return None
    ma,mb=sum(a)/len(a),sum(b)/len(b)
    denominator=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return sum((x-ma)*(y-mb)for x,y in zip(a,b))/denominator if denominator else None


def ordering(rows,key):
    valid=[r for r in rows if r.get(key)is not None and r.get('observed_deterioration')is not None]
    a=[round(r[key],6)for r in valid];b=[round(r['observed_deterioration'],6)for r in valid]
    def ranks(v):return [1+sum(x<y for x in v)+(sum(x==y for x in v)-1)/2 for y in v]
    pairs=[(i,j)for i in range(len(a))for j in range(i)if b[i]!=b[j]]
    return dict(n=len(valid),spearman=correlation(ranks(a),ranks(b)),pearson=correlation(a,b),
        pair_order_accuracy=sum(1 if (a[i]-a[j])*(b[i]-b[j])>0 else .5 if a[i]==a[j] else 0 for i,j in pairs)/len(pairs)if pairs else None,
        comparable_pairs=len(pairs),tie_rounding_digits=6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--influences',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--node',default=shutil.which('node'))
    p.add_argument('--method',choices=['goal_obstruction_v1','goal_obstruction_v2'],default='goal_obstruction_v2')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not a.node:p.error('Node is needed to execute the production risk modules')
    batch=json.loads(a.batch.read_text());graph=align_graph(validate_influences(graph_view(build_graph()),a.influences))
    assert batch['context']['contracts']==graph['contract_sha256'],'Frozen controllers changed; do not compare across versions'
    assert batch['context']['metric_scope']==graph['metric_scope']
    jobs=a.batch.parents[2];cases=[];labels=[];hashes={str(a.batch):sha(a.batch),str(a.influences):sha(a.influences)}
    for member in batch['members']:
        if member['phase']!='complete' or not member.get('comparison'):continue
        run=jobs/member['job_id']/'data/run';comparison=member['comparison'];baseline=comparison['baseline']['slot'];apply=comparison['application_slot']
        previews=[r for r in records(run,'xapp_decision_previews')if r['slot']<=baseline and r['slot']<apply]
        if not previews:raise ValueError('No genuinely predeployment preview; refusing to use future decisions')
        at=max(r['slot']for r in previews);preview=PreviewStore(run).read(at)
        source=replay(run,180,population=True);frame=next(f for f in source['frames']if f['slot']==at)
        assert all(not frame['states'].get(app,{}).get('enabled') for app in member['app_ids'])
        config=json.loads((run/'config_snapshot.json').read_text());ric=config['configs'].get('ric',{})
        cases.append(dict(group_id=member['group_id'],app_ids=member['app_ids'],frame=frame,preview=preview,ranges=batch['ranges'],
            options=dict(method=a.method,arbitration=dict(mode=ric.get('direct_conflict_mitigation','uniform_random')),
                normalization=normalization(batch['ranges'],frame,batch=batch)),
            timing=dict(preview_slot=at,baseline_slot=baseline,application_slot=apply,
                lag_to_baseline_ms=(baseline-at)*source['slot_duration_ms'],lag_to_application_ms=(apply-at)*source['slot_duration_ms'])))
        labels.append(dict(group_id=member['group_id'],job_id=member['job_id'],
            before_after=[dict(kpi=r['kpi'],before=r['before'],after=r['after'])for r in comparison['rows']],
            valid=comparison['after']['valid']==comparison['after']['required'],
            passed=comparison['after']['all_satisfied'],observe_slots=comparison['observe_slots']))
        for path in (run/'config_snapshot.json',run/'xapp_decision_previews.jsonl'):
            hashes[str(path)]=sha(path)
    inputs=dict(graph=graph,cases=cases)
    (a.output/'scoring-inputs.json').write_text(json.dumps(inputs,indent=2,allow_nan=False))
    js="""const fs=require('fs'),vm=require('vm'),c={structuredClone};vm.createContext(c);
vm.runInContext(fs.readFileSync('smo/web/app.js','utf8'),c);
vm.runInContext(fs.readFileSync('smo/web/goal_obstruction.js','utf8'),c);
const d=JSON.parse(fs.readFileSync(0,'utf8'));
const output=d.cases.map(x=>({group_id:x.group_id,timing:x.timing,
  result:c.SmoGoalObstruction.review(d.graph,x.frame,[{id:x.group_id,app_ids:x.app_ids}],x.ranges,x.preview,x.options)}));
process.stdout.write(JSON.stringify(output));"""
    scored=json.loads(subprocess.check_output([a.node,'-e',js],cwd=ROOT,text=True,input=json.dumps(inputs)))
    (a.output/'scores-before-label-join.json').write_text(json.dumps(scored,indent=2,allow_nan=False))
    observed={m['group_id']:m['comparison']['result_evaluation']for m in enrich_batch(batch)['members']if m.get('comparison')}
    rows=[]
    for case,label in zip(scored,labels):
        assert case['group_id']==label['group_id'];q=case['result']['groups'][0]
        e=observed[case['group_id']]
        rows.append(dict(**label,**case['timing'],score=q['score'],fractional_score=q.get('fractional_score'),
            raw_obstruction_strength=q.get('raw_obstruction_strength'),legacy_aligned_score=q.get('legacy_score'),
            observed_before_score=e['before']['score'],observed_after_score=e['after']['score'],
            observed_deterioration=-e['change']if e['change']is not None else None,
            score_status=q['score_status'],reason=q.get('reason'),
            goal_components=q['goal_components'],outcome_count=len(q['actual_outcomes']),
            reference_outcome_count=q['reference_outcome_count']))
    pairs=[(x,y)for x in rows if x['valid']and x['passed']for y in rows if y['valid']and not y['passed']]
    auc=lambda key:sum(1 if x[key]<y[key]else .5 if x[key]==y[key]else 0 for x,y in pairs)/len(pairs)if pairs and all(x[key]is not None and y[key]is not None for x,y in pairs)else None
    result=dict(batch_id=batch['id'],new_simulations=0,scoring_outcomes_used=False,
        method=a.method,rows=rows,
        continuous_ordering={key:ordering(rows,key)for key in ('score','raw_obstruction_strength','fractional_score','legacy_aligned_score')},
        valid_score_count=sum(r['score']is not None for r in rows),
        passing_groups=sum(r['valid']and r['passed']for r in rows),
        exploratory_pairwise_order_accuracy=dict(new=auc('score'),legacy_aligned=auc('legacy_aligned_score')),
        limits=['One seed and overlapping combinations; no independent generalization claim.',
            'Latest recorded preview before baseline is used; timing gaps are reported, not called exact deployment alignment.',
            'Reference gains are importance weights, not a deployed KPI-response predictor.',
            'Fresh random arbitration is integrated exactly for frozen requests; held leases and future closed-loop decisions are not reconstructed.',
            'Counterfactual envelope uses complete controller subsets, not invented parameter values.'])
    (a.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    lines=['# Frozen-proposal goal obstruction trial','',
        'All score inputs are from recorded predeployment states. Actual after-KPI labels are joined only after scores are saved. No new simulation.','',
        '| Combination | Fractional loss v1 | Selected score ('+a.method+') | Raw B | Observed result G | Observed deterioration | Intent met |','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        fmt=lambda x:'unavailable'if x is None else f'{x:.2f}'
        lines.append(f"| {' + '.join(x.removeprefix('xapp:')for x in next(c['app_ids']for c in cases if c['group_id']==r['group_id']))} | {fmt(r['fractional_score'])} | {fmt(r['score'])} | {fmt(r['raw_obstruction_strength'])} | {fmt(r['observed_after_score'])} | {fmt(r['observed_deterioration'])} | {r['passed']} |")
    lines+=['',f"Valid scores: {result['valid_score_count']}/{len(rows)}; actual passing groups: {result['passing_groups']}.",
        '', 'Exploratory pairwise ordering (lower risk for passing groups): '+str(result['exploratory_pairwise_order_accuracy']),
        '', 'Continuous ordering against G_before - G_after (higher = worse):',
        '```json',json.dumps(result['continuous_ordering'],indent=2),'```',
        '', '## Limits','']+['- '+s for s in result['limits']]
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    (a.output/'provenance.json').write_text(json.dumps(dict(input_sha256=hashes,
        scoring_input_sha256=sha(a.output/'scoring-inputs.json'),
        engine_sha256=sha(ROOT/'smo/web/goal_obstruction.js'),reference_engine_sha256=sha(ROOT/'smo/web/app.js'),
        benchmark_sha256=sha(__file__),new_simulations=0,
        score_label_separation=True),indent=2))
    print(json.dumps({k:v for k,v in result.items()if k not in ('rows','limits')}))


if __name__=='__main__':main()
