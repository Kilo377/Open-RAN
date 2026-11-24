"""Fit measured same-origin paired latency contrasts; never reads held-out rows."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parents[1]
SOURCE=OUT/'sources/source-v3';sys.path.insert(0,str(SOURCE))
from ric.near_rt.pools.chongyu.latency_scheduler_adaptive import ACTIONS,SCHEMA,FEATURES,CONTEXT_FEATURES,feature_row,predict

def dataset():
    rows=[];excluded=[]
    cases=[json.loads(p.read_text()) for p in sorted(OUT.glob('workers/*/*/metrics.json'))+sorted(OUT.glob('analysis/derived-native-references/*.json'))]
    cases=[r for r in cases if r['case']['split']=='training']
    references={}
    for row in cases:
        c=row['case']
        key=c['worker'],c['seed'],c.get('origin_slot',2040)
        if c['arm']=='OFF' and (key not in references or c['slots']>references[key]['case']['slots']):references[key]=row
    native=set()
    for row in cases:
        c=row['case'];key=(c['worker'],c['seed'],c.get('origin_slot',2040))
        if c['arm']!='FIXED' or key not in references:continue
        if not row.get('execution_audit',{}).get('all_proposed_actions_verified'):
            excluded.append(dict(case=c['name'],reason='normal_control_execution_unverified'));continue
        off=references[key]
        if row['network_prefix_sha256']!=off['network_prefix_sha256']:
            raise ValueError('A paired cold prefix changed: '+c['name'])
        first={}
        for d in row['decisions']:
            if d.get('kind')=='decision' and 'context' in d:first.setdefault(d['cell_id'],d)
        for cid,decision in first.items():
            if decision['slot_id']!=key[2]+40:raise ValueError('First actual decision timing differs')
            if decision.get('context_names')!=CONTEXT_FEATURES:
                excluded.append(dict(case=c['name'],cell_id=cid,reason='exploratory_feature_schema_not_used_for_final_fit'))
                continue
            uid=decision['ue_ids'][0]
            before=off['windows']['500']['per_ue'][uid];after=row['windows']['500']['per_ue'][uid]
            identities=lambda r:{(t['task_key'],t['created_s'],t['deadline_s'],t['offloaded']) for t in r['tasks']}
            if identities(before)!=identities(after):raise ValueError('Task populations differ')
            if before['value'] is None or before['sample_count']<1 or before['failed'] or not after['generated']:
                excluded.append(dict(case=c['name'],uid=uid,before={k:v for k,v in before.items() if k!='tasks'},after={k:v for k,v in after.items() if k!='tasks'}));continue
            # A task still pending after its mature deadline has an observed
            # latency lower bound. Keep that adverse observation for training;
            # actual evaluation continues to report missing/failed data as such.
            completed_sum=sum(t['terminal']['latency_ms'] for t in after['tasks']
                if t.get('terminal',{}).get('kind')=='task_completed'
                and t['terminal']['recorded_slot']<=key[2]+500)
            end=key[2]+500
            pending=[t for t in after['tasks'] if not t.get('terminal') or t['terminal']['recorded_slot']>end]
            pending_lower_bound=sum(max(100.,end*.5-t['created_s']*1000.) for t in pending)
            observed_cost=(completed_sum+100.*after['failed']+pending_lower_bound)/after['generated']
            if after['unresolved'] and observed_cost<before['value']:
                excluded.append(dict(case=c['name'],uid=uid,reason='censored_label_cannot_establish_an_improvement'));continue
            contrast=(observed_cost-before['value'])/max(8.,before['value'])
            public=dict(zip(decision['context_names'],decision['context']))
            public['cell_queue_kb']=public['target_queue_kb']+public['other_queue_kb']
            ctx=[public.get(name,0.) for name in CONTEXT_FEATURES]
            record=dict(seed=c['seed'],origin=key[2],cell_id=cid,ue_id=uid,action=decision['action'],
                context=ctx,features=feature_row(ctx,decision['action']),target=contrast,
                imputed_feature_names=[n for n in CONTEXT_FEATURES if n not in public],
                before_ms=before['value'],after_ms=after['value'],samples=before['sample_count'],
                observed_training_cost_ms=observed_cost,label_is_lower_bound=after['unresolved']>0,
                source_case=c['name'],off_case=off['case']['name'],prefix_sha256=row['network_prefix_sha256'])
            if decision['action']!=0:rows.append(record)
            if (*key,cid) not in native:
                native.add((*key,cid));rows.append(dict(record,action=0,features=feature_row(ctx,0),target=0.,after_ms=before['value'],
                    observed_training_cost_ms=before['value'],label_is_lower_bound=False,source_case=off['case']['name']))
    return rows,excluded

def export(model):
    trees=[]
    for est in model.estimators_:
        t=est.tree_;trees.append(dict(children_left=t.children_left.tolist(),children_right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value.ravel().tolist()))
    return trees

def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--minimum-contexts',type=int,default=6);a=p.parse_args()
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    rows,excluded=dataset();contexts={(r['seed'],r['origin'],r['cell_id']) for r in rows}
    if len(contexts)<a.minimum_contexts:raise ValueError('Insufficient independent training contexts')
    seeds=sorted({r['seed'] for r in rows});allowed=sorted({r['action'] for r in rows})
    x=np.asarray([r['features'] for r in rows]);y=np.asarray([r['target'] for r in rows]);groups=np.asarray([r['seed'] for r in rows])
    trials=[]
    specs=[(kind,depth,leaf,feat) for kind in ('rf','extra') for depth in (3,5,7) for leaf in (1,2) for feat in (.7,1.)]
    for kind,depth,leaf,feat in specs:
        params=dict(n_estimators=96,max_depth=depth,min_samples_leaf=leaf,max_features=feat,random_state=20261124,n_jobs=2)
        cls=RandomForestRegressor if kind=='rf' else ExtraTreesRegressor
        losses=[];regrets=[]
        if len(seeds)>1:
            for train,test in LeaveOneGroupOut().split(x,y,groups):
                fit=cls(**params).fit(x[train],y[train]);prediction=fit.predict(x[test]);losses.extend((prediction-y[test])**2)
                # Only compare actually recorded alternatives in each held-out training context.
                by_context={}
                for offset,index in enumerate(test):
                    r=rows[index];by_context.setdefault((r['seed'],r['origin'],r['cell_id']),[]).append((r,prediction[offset]))
                for alternatives in by_context.values():
                    chosen=min(alternatives,key=lambda item:0. if item[0]['action']==0 else item[1]+.02)
                    best=min(r['target'] for r,_ in alternatives)
                    regrets.append(chosen[0]['target']-best+2.*max(0.,chosen[0]['target']))
        objective=float(np.mean(regrets)) if regrets else 0.
        trials.append(dict(kind=kind,params=params,group_cv_mse=float(np.mean(losses)) if losses else None,group_cv_regret=objective))
    selected=min(trials,key=lambda r:(r['group_cv_regret'],r['group_cv_mse'] or 0.))
    cls=RandomForestRegressor if selected['kind']=='rf' else ExtraTreesRegressor
    model=cls(**selected['params']).fit(x,y);forest=export(model)
    actual=np.asarray([predict(forest,row)[0] for row in x.tolist()]);assert np.max(np.abs(actual-model.predict(x)))<1e-9
    path=OUT/'checkpoints'/a.version;path.mkdir(exist_ok=False)
    metadata=dict(training_rows=len(rows),independent_contexts=len(contexts),training_seeds=seeds,
        estimator=type(model).__name__,fit_parameters=selected['params'],
        excluded=len(excluded),target='(candidate task latency - paired native latency) / max(native latency,8ms)',
        selection='leave-one-training-seed-out regret with regression penalty; no development or heldout results',
        created_utc=datetime.now(timezone.utc).isoformat(),real_network_samples_only=True)
    policy=dict(schema=SCHEMA,kind='rf',actions=ACTIONS,allowed_actions=allowed,feature_names=FEATURES,forest=forest,
        startup_hold_ms=260,decision_ms=100,minimum_predicted_gain=.02,uncertainty_weight=0.,training=metadata)
    # Training metadata is separate from the boolean enabling PPO learning.
    policy['training_metadata']=policy.pop('training')
    (path/'policy.json').write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
    (path/'dataset.json').write_text(json.dumps(dict(rows=rows,excluded=excluded),indent=2,allow_nan=False)+'\n')
    (path/'selection.json').write_text(json.dumps(dict(selected=selected,trials=trials,metadata=metadata),indent=2,allow_nan=False)+'\n')
    for guard in (0.,.2,.5):
        variant={**policy,'uncertainty_weight':guard}
        (path/f'policy-guard-{guard:g}.json').write_text(json.dumps(variant,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(checkpoint=str(path),metadata=metadata,selected=selected,allowed_actions=allowed),allow_nan=False),flush=True)

if __name__=='__main__':main()
