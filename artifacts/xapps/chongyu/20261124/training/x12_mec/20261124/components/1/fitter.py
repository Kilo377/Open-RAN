"""Train a whole-xApp RF: one candidate is the complete two-cell action vector."""
import argparse
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'sources/source-v4'))
from ric.near_rt.pools.chongyu.latency_scheduler_adaptive import (
    ACTIONS,CONTEXT_FEATURES,JOINT_FEATURES,SCHEMA,joint_feature_row,predict)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fit_contextual import export
from summarize_comparisons import followed_cohort

def supported_vectors(rows):
    """Alternatives need matched, positive benefit on two training seeds.

    Native and the jointly probed relative-priority action are retained.
    Unequal arm coverage must not create a pooled-average advantage for an
    asymmetric action that never beats the joint action at the same state.
    """
    from collections import defaultdict
    contexts=defaultdict(dict);wins=defaultdict(set)
    for r in rows:contexts[r['seed'],r['origin_slot']][tuple(r['vector'])]=r['target']
    for (seed,_),values in contexts.items():
        if (1,1) not in values:continue
        for vector,value in values.items():
            if value<-.02 and value<values[1,1]-.01:wins[vector].add(seed)
    return {(0,0),(1,1)}|{vector for vector,seeds in wins.items() if len(seeds)>=2}

def dataset(followup_labels=False,label_slots=500,include_training_diagnostics=False):
    if label_slots not in (500,1000,2000,6000):raise ValueError('Unsupported measured label horizon')
    paths=sorted(OUT.glob('workers/*/*/metrics.json'))+sorted(OUT.glob('analysis/derived-native-references/*.json'))
    cases=[json.loads(p.read_text()) for p in paths]
    for r in cases:
        if r['case'].get('record_kind')=='derived_native_reference' and not (r.get('new_simulations')==0 and r.get('native_reuse_proof',{}).get('all_shorter_native_histories_equal')):
            raise ValueError('Unverified native observation reuse')
    diagnostic_seeds={20261214,2026112402}
    cases=[r for r in cases if r['case']['split']=='training' or
           include_training_diagnostics and r['case']['split']=='diagnostic' and r['case']['seed'] in diagnostic_seeds]
    off={}
    for r in cases:
        c=r['case']
        key=c['worker'],c['seed'],c['origin_slot']
        if c['arm']=='OFF' and (key not in off or c['slots']>off[key]['case']['slots']):off[key]=r
    rows=[];excluded=[];native=set();contexts={}
    for r in cases:
        c=r['case'];key=(c['worker'],c['seed'],c['origin_slot'])
        if c['arm']!='FIXED' or key not in off:continue
        if str(label_slots) not in r['windows'] or str(label_slots) not in off[key]['windows']:continue
        if not r.get('execution_audit',{}).get('all_proposed_actions_verified'):
            excluded.append(dict(case=c['name'],reason='normal_control_execution_unverified'));continue
        baseline=off[key]
        if r['network_prefix_sha256']!=baseline['network_prefix_sha256']:raise ValueError('Cold native prefix differs')
        first={}
        for d in r['decisions']:
            if d.get('kind')=='decision' and 'context' in d:
                for uid in d['ue_ids']:first.setdefault(uid,d)
        if set(first)!={'ue_0','ue_4'}:continue
        if any(first[uid].get('context_names')!=CONTEXT_FEATURES for uid in first):
            excluded.append(dict(case=c['name'],reason='exploratory_feature_schema'));continue
        if any(first[uid]['slot_id']!=key[2]+40 for uid in first):raise ValueError('Unexpected first policy decision')
        ctx=[first[uid]['context'] for uid in ('ue_0','ue_4')]
        if key in contexts and any(abs(a-b)>1e-9 for a,b in zip(sum(ctx,[]),sum(contexts[key],[]))):
            raise ValueError('Paired public first states differ: '+c['name'])
        contexts[key]=ctx
        before=baseline['windows'][str(label_slots)]['aggregate']['states']['kpi:mec_offloaded_latency_ms']
        after=r['windows'][str(label_slots)]['aggregate']['states']['kpi:mec_offloaded_latency_ms']
        raw_before,raw_after=before,after
        if followup_labels:before,after=followed_cohort(baseline,label_slots),followed_cohort(r,label_slots)
        ids=lambda x:{(uid,t['task_key'],t['created_s'],t['offloaded']) for uid in ('ue_0','ue_4') for t in x['windows'][str(label_slots)]['per_ue'][uid]['tasks']}
        if ids(r)!=ids(baseline):raise ValueError('Offloaded task identities differ')
        if before.get('pending') or not before['generated'] or (not followup_labels and (before['value'] is None or before.get('failed') or not before['sample_count'])):
            excluded.append(dict(case=c['name'],reason='native_cohort_unresolved'));continue
        generated=after['generated']
        if not generated:continue
        end=c['slots'] if followup_labels else key[2]+label_slots
        tasks=[t for uid in ('ue_0','ue_4') for t in r['windows'][str(label_slots)]['per_ue'][uid]['tasks']]
        total=sum(t['terminal']['latency_ms'] for t in tasks if t.get('terminal',{}).get('kind')=='task_completed' and t['terminal']['recorded_slot']<=end)
        pending=[t for t in tasks if not t.get('terminal') or t['terminal']['recorded_slot']>end]
        pending_lower_bound=sum(max(100.,end*.5-t['created_s']*1000.) for t in pending)
        cost=(total+100.*after.get('failed',0)+pending_lower_bound)/generated
        reference_cost=before['value']
        if followup_labels:
            original=[t for uid in ('ue_0','ue_4') for t in baseline['windows'][str(label_slots)]['per_ue'][uid]['tasks']]
            reference_cost=(sum(t['terminal']['latency_ms'] for t in original if t.get('terminal',{}).get('kind')=='task_completed')+100.*before['failed'])/before['generated']
        if after.get('pending') and cost<reference_cost:
            excluded.append(dict(case=c['name'],reason='censored_label_cannot_establish_an_improvement'));continue
        target=(cost-reference_cost)/max(8.,reference_cost)
        vector=[first[uid]['action'] for uid in ('ue_0','ue_4')]
        row=dict(case=c['name'],seed=c['seed'],origin_slot=c['origin_slot'],worker=c['worker'],contexts=ctx,
            vector=vector,features=joint_feature_row(ctx,vector),target=target,
            before_ms=before['value'],after_ms=after['value'],observed_training_cost_ms=cost,
            sample_count=before['sample_count'],label_is_lower_bound=after.get('pending',0)>0,
            followup_labels=followup_labels,label_slots=label_slots,reference_training_cost_ms=reference_cost,
            raw_window_before_ms=raw_before['value'],raw_window_after_ms=raw_after['value'],
            baseline_failed=before.get('failed',0),candidate_failed=after.get('failed',0),
            off_case=baseline['case']['name'],prefix_sha256=r['network_prefix_sha256'])
        if vector!=[0,0]:rows.append(row)
        if key not in native:
            native.add(key);rows.append(dict(row,case=baseline['case']['name'],vector=[0,0],features=joint_feature_row(ctx,[0,0]),target=0.,
                after_ms=before['value'],observed_training_cost_ms=reference_cost,label_is_lower_bound=False))
    return rows,excluded

def main():
    fitter_source=Path(__file__).read_bytes()
    import fit_contextual,summarize_comparisons
    exporter_source=Path(fit_contextual.__file__).read_bytes();observation_source=Path(summarize_comparisons.__file__).read_bytes()
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--minimum-contexts',type=int,default=6)
    p.add_argument('--latency-only',action='store_true');p.add_argument('--action-only',action='store_true')
    p.add_argument('--balanced-seeds',action='store_true');p.add_argument('--followup-labels',action='store_true')
    p.add_argument('--dataset-file',type=Path);p.add_argument('--empirical-action-filter',action='store_true')
    p.add_argument('--label-slots',type=int,choices=[500,1000,2000,6000],default=500)
    p.add_argument('--include-training-diagnostics',action='store_true')
    p.add_argument('--minimum-predicted-gain',type=float,default=.02);a=p.parse_args()
    if not 0<=a.minimum_predicted_gain<=.5:raise ValueError('Invalid predicted-gain threshold')
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
    from sklearn.model_selection import LeaveOneGroupOut
    if a.dataset_file:
        frozen=json.loads(a.dataset_file.read_text());rows,excluded=frozen['rows'],frozen['excluded']
        if any(r.get('followup_labels',False)!=a.followup_labels for r in rows):raise ValueError('Frozen label definition differs from the requested training objective')
        if any(r.get('label_slots',500)!=a.label_slots for r in rows):raise ValueError('Frozen label horizon differs')
    else:rows,excluded=dataset(followup_labels=a.followup_labels,label_slots=a.label_slots,include_training_diagnostics=a.include_training_diagnostics)
    if a.latency_only:
        removed=[r for r in rows if any(i>=9 for i in r['vector'])]
        excluded.extend(dict(case=r['case'],reason='registered_latency_only_candidate') for r in removed)
        rows=[r for r in rows if all(i<9 for i in r['vector'])]
    contexts={(r['seed'],r['origin_slot']) for r in rows};seeds=sorted({r['seed'] for r in rows})
    planpath=OUT/'protocol/evaluation-selection.json'
    if planpath.exists():
        plan=json.loads(planpath.read_text());protected=set(plan['development_seeds'])|set(plan['final_heldout']['seeds'])
        if set(seeds)&protected:raise ValueError('Development/heldout seeds cannot enter training')
    if len(contexts)<a.minimum_contexts:raise ValueError('Not enough matched joint contexts')
    x=np.asarray([r['features'] for r in rows]);y=np.asarray([r['target'] for r in rows]);groups=np.asarray([r['seed'] for r in rows])
    if a.action_only:x[:,:-8]=0.  # A trained noncontextual control, not a manually selected fixed action.
    def weights(indices):
        from collections import Counter,defaultdict
        counts=Counter((rows[i]['seed'],rows[i]['origin_slot']) for i in indices);perseed=defaultdict(set)
        for seed,origin in counts:perseed[seed].add(origin)
        return np.asarray([1./(len(perseed[rows[i]['seed']])*counts[rows[i]['seed'],rows[i]['origin_slot']]) for i in indices])
    trials=[]
    for kind in ('rf','extra'):
        for depth in (3,5,7):
            for leaf in (1,2):
                params=dict(n_estimators=128,max_depth=depth,min_samples_leaf=leaf,max_features=.85,random_state=20261124,n_jobs=2)
                cls=RandomForestRegressor if kind=='rf' else ExtraTreesRegressor
                regrets=[];errors=[]
                for train,test in LeaveOneGroupOut().split(x,y,groups):
                    model=cls(**params).fit(x[train],y[train],sample_weight=weights(train) if a.balanced_seeds else None)
                    pred=model.predict(x[test]);errors.append(float(np.mean((pred-y[test])**2)))
                    grouped={}
                    catalogue=supported_vectors([rows[i] for i in train]) if a.empirical_action_filter else None
                    for i,index in enumerate(test):
                        r=rows[index];grouped.setdefault((r['seed'],r['origin_slot']),[]).append((r,pred[i]))
                    fold_regrets=[]
                    for choices in grouped.values():
                        eligible=[item for item in choices if catalogue is None or tuple(item[0]['vector'])in catalogue]
                        best=min(eligible,key=lambda item:0. if item[0]['vector']==[0,0] else item[1]+a.minimum_predicted_gain)
                        fold_regrets.append(best[0]['target']-min(r['target'] for r,_ in choices)+2.*max(0.,best[0]['target']))
                    regrets.append(float(np.mean(fold_regrets)))
                trials.append(dict(kind=kind,params=params,group_cv_regret=float(np.mean(regrets)),group_cv_mse=float(np.mean(errors))))
    selected=min(trials,key=lambda r:(r['group_cv_regret'],r['group_cv_mse']))
    cls=RandomForestRegressor if selected['kind']=='rf' else ExtraTreesRegressor
    model=cls(**selected['params']).fit(x,y,sample_weight=weights(range(len(rows))) if a.balanced_seeds else None);forest=export(model)
    assert max(abs(predict(forest,row)[0]-v) for row,v in zip(x.tolist(),model.predict(x)))<1e-9
    if a.action_only:
        assert all(f<0 or f>=len(JOINT_FEATURES)-8 for tree in forest for f in tree['feature'])
    measured={tuple(r['vector']) for r in rows}
    vectors=[list(v) for v in sorted(measured & supported_vectors(rows) if a.empirical_action_filter else measured)]
    dataset_text=json.dumps(dict(rows=rows,excluded=excluded),indent=2,allow_nan=False)+'\n'
    import sklearn
    metadata=dict(training_rows=len(rows),independent_contexts=len(contexts),training_seeds=seeds,
        estimator=type(model).__name__,fit_parameters=selected['params'],
        objective='network mean completed offloaded MEC latency; failures cost100ms; pending tasks retain observed-age lower bounds (at least100ms)',
        label=f'relative controlled native contrast over first{a.label_slots} slots'+('; actual terminal followup of the same eligible birth cohort' if a.followup_labels else ''),selection='leave-one-training-seed-out regret with regression penalty',
        label_slots=a.label_slots,training_diagnostic_records_included=a.include_training_diagnostics,
        created_utc=datetime.now(timezone.utc).isoformat(),no_heldout_or_development_rows=True,
        latency_only=a.latency_only,action_only=a.action_only,balanced_training_seeds_and_contexts=a.balanced_seeds,
        actual_terminal_followup_labels=a.followup_labels,
        empirical_action_filter=a.empirical_action_filter,
        minimum_predicted_gain=a.minimum_predicted_gain,
        action_filter_rule='Native and joint relative-priority action retained; alternatives require>2% native benefit and>1% benefit over the joint action on at least2 matched training seeds' if a.empirical_action_filter else None,
        cross_validation_seed_weights_equal=True,raw_record_order='sorted retained worker/case paths',
        dataset_sha256=hashlib.sha256(dataset_text.encode()).hexdigest(),
        fitter_sha256=hashlib.sha256(fitter_source).hexdigest(),
        exporter_sha256=hashlib.sha256(exporter_source).hexdigest(),
        observation_helper_sha256=hashlib.sha256(observation_source).hexdigest(),
        numpy_version=np.__version__,sklearn_version=sklearn.__version__)
    directory=OUT/'checkpoints'/a.version;directory.mkdir(exist_ok=False)
    (directory/'fitter.py').write_bytes(fitter_source);(directory/'exporter.py').write_bytes(exporter_source)
    (directory/'observations.py').write_bytes(observation_source)
    policy=dict(schema=SCHEMA,kind='joint_rf',actions=ACTIONS,action_vectors=vectors,feature_names=JOINT_FEATURES,forest=forest,
        startup_hold_ms=260,decision_ms=260,minimum_predicted_gain=a.minimum_predicted_gain,uncertainty_weight=0.,training_metadata=metadata)
    (directory/'policy.json').write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
    (directory/'dataset.json').write_text(dataset_text)
    (directory/'selection.json').write_text(json.dumps(dict(selected=selected,trials=trials,metadata=metadata),indent=2,allow_nan=False)+'\n')
    for guard in (0.,.2,.5):
        variant={**policy,'uncertainty_weight':guard}
        (directory/f'policy-guard-{guard:g}.json').write_text(json.dumps(variant,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(checkpoint=str(directory),metadata=metadata,selected=selected,action_vectors=vectors)),flush=True)

if __name__=='__main__':main()
