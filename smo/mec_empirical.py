"""Scoped empirical action-direction regression, separate from causal gains.

Only NumPy is required for fitting. No xApp training, simulator or executor is
imported. Rank/coverage diagnostics prevent ridge regularization from being
mistaken for identification of an unobserved control direction.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import math
import numpy as np

MODEL_SCHEMA='oran.mec-empirical-response.v1'
ALPHAS=(0.,.01,.1,1.,10.)


def _fit(rows,alpha):
    X=np.asarray([r['x']for r in rows],dtype=float);y=np.asarray([r['y']for r in rows],dtype=float)
    scales=np.asarray([3. if f.startswith('weight:')else 1. for f in rows[0]['features']])
    centers={};A=X.copy();b=y.copy()
    for stratum in sorted({r['stratum']for r in rows}):
        ids=[i for i,r in enumerate(rows)if r['stratum']==stratum]
        xm=X[ids].mean(axis=0);ym=float(y[ids].mean());centers[stratum]=dict(x=xm.tolist(),y=ym)
        A[ids]-=xm;b[ids]-=ym
    A/=scales
    U,s,Vt=np.linalg.svd(A,full_matrices=False)
    keep=s>max((float(s[0])if len(s)else 0)*1e-8,1e-10)
    U,s,Vt=U[:,keep],s[keep],Vt[keep]
    beta=Vt.T@((s/(s*s+alpha))*(U.T@b))if len(s)else np.zeros(X.shape[1])
    beta/=scales
    return dict(coefficients=beta.tolist(),scales=scales.tolist(),basis=Vt.tolist(),singular_values=s.tolist(),rank=len(s),
        centers=centers,alpha=alpha,feature_min=X.min(0).tolist(),feature_max=X.max(0).tolist(),n=len(rows))


def _predict(model,row):
    c=model['centers'].get(row['stratum'])
    if c is None:return None
    return c['y']+float(np.dot(np.asarray(row['x'])-c['x'],model['coefficients']))


def support_residual(model,direction):
    z=np.asarray(direction,dtype=float)/model['scales'];V=np.asarray(model['basis'],dtype=float)
    remainder=z-(V.T@(V@z)if V.size else 0)
    return float(np.linalg.norm(remainder)),float(np.linalg.norm(z))


def _validate(rows,alpha,*,nested=False):
    predictions=[]
    for group in sorted({r['trajectory_sha256']for r in rows}):
        train=[r for r in rows if r['trajectory_sha256']!=group]
        test=[r for r in rows if r['trajectory_sha256']==group]
        if len(train)<3:continue
        chosen=_choose_alpha(train)if nested else alpha
        model=_fit(train,chosen)
        for row in test:
            prediction=_predict(model,row)
            if prediction is None:continue
            center=model['centers'][row['stratum']];direction=np.asarray(row['x'])-center['x']
            residual,norm=support_residual(model,direction)
            actual=row['y']-center['y'];estimated=prediction-center['y']
            predictions.append(dict(job_ids=row['source_job_ids'],trajectory_sha256=group,stratum=row['stratum'],
                actual=row['y'],predicted=prediction,baseline=center['y'],absolute_error=abs(prediction-row['y']),
                baseline_error=abs(center['y']-row['y']),supported_direction=residual<=1e-7*max(1.,norm),
                direction_correct=(actual*estimated>0)if abs(actual)>1e-6 else None,alpha=chosen))
    return predictions


def _choose_alpha(rows):
    scored=[]
    for alpha in ALPHAS:
        p=_validate(rows,alpha)
        if p:scored.append((float(np.mean([r['absolute_error']for r in p])),alpha))
    return min(scored)[1]if scored else 1.


def fit_dataset(dataset):
    rows=dataset['rows']
    if len(rows)<4:raise ValueError('At least four unique executed conditions are required')
    if dataset.get('duplicate_response_discrepancies'):raise ValueError('Identical executions have inconsistent labels')
    if len({r['context_sha256']for r in rows})!=1:raise ValueError('Do not pool incompatible native environments')
    if len({tuple(r['features'])for r in rows})!=1:raise ValueError('Feature schemas differ')
    alpha=_choose_alpha(rows);model=_fit(rows,alpha);held_out=_validate(rows,alpha,nested=True)
    errors=[p['absolute_error']for p in held_out];naive=[p['baseline_error']for p in held_out]
    mae=float(np.mean(errors))if errors else None;baseline=float(np.mean(naive))if naive else None
    directions=[p['direction_correct']for p in held_out if p['direction_correct']is not None]
    usable=len(held_out)>=6 and model['rank']>0 and mae is not None and baseline is not None and mae<baseline
    result=dict(schema=MODEL_SCHEMA,evidence='observational_executed_control_regression',kpi='mec_offloaded_latency_ms',
        dataset_content_sha256=hashlib.sha256(json.dumps(dataset,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        method='stratum_fixed_effects_ridge_observed_SVD_directions',status='conditional'if usable else'exploratory_not_validated',
        automatic_selection=False,causal_effect_identified=False,independent_seeds=dataset['independent_seeds'],
        context=deepcopy(rows[0]['context']),context_sha256=rows[0]['context_sha256'],features=rows[0]['features'],
        horizon_ms=rows[0]['duration_ms'],window_slots=dataset['window_slots'],model=model,
        validation=dict(method='nested_leave_complete_execution_trajectory_out',mae_ms=mae,baseline_mae_ms=baseline,
            mean_error_reduction_fraction=1-mae/baseline if baseline and mae is not None else None,
            max_error_ms=max(errors)if errors else None,p90_error_ms=float(np.quantile(errors,.9))if errors else None,
            direction_accuracy=float(np.mean(directions))if directions else None,held_out_rows=len(held_out),
            supported_held_out_rows=sum(p['supported_direction']for p in held_out),predictions=held_out,
            interpretation='conditional_same_seed_time_strata_validation_not_independent_seed_generalization'),
        coverage=dict(unique_run_windows=len(rows),unique_trajectories=len({r['trajectory_sha256']for r in rows}),
            raw_jobs=dataset['raw_job_count'],usable_jobs_before_dedup=dataset['usable_before_dedup'],
            feature_dimension=len(rows[0]['features']),identified_rank=model['rank'],strata=len(model['centers'])),
        uncertainty='held_out_error_envelope_not_a_confidence_interval',
        notes=['Actual applied RCPs only; xApp identities are not regression features.',
            'Other controls stay native; unknown/extrapolated directions are refused.',
            'Window means approximate dynamic actions; individual task samples are not independent seeds.',
            'Coefficients in null-space directions are not identified; only supported joint action directions may be used.'])
    return result


def load_model(path):
    path=Path(path);data=json.loads(path.read_text())
    if data.get('schema')!=MODEL_SCHEMA or data.get('evidence')!='observational_executed_control_regression':raise ValueError('Unsupported empirical model')
    d=len(data['features']);m=data['model']
    if len(m['coefficients'])!=d or len(m['scales'])!=d or any(len(r)!=d for r in m['basis']):raise ValueError('Invalid response dimensions')
    if any(not math.isfinite(v)for v in m['coefficients']+m['scales'])or min(m['scales'])<=0:raise ValueError('Invalid response coefficients')
    if len(m['feature_min'])!=d or len(m['feature_max'])!=d or any(not math.isfinite(v)for v in m['feature_min']+m['feature_max']+[v for r in m['basis']for v in r]):raise ValueError('Invalid response support')
    V=np.asarray(m['basis'])
    if m['rank']!=len(m['basis'])or V.size and not np.allclose(V@V.T,np.eye(m['rank']),atol=1e-7):raise ValueError('Invalid response basis')
    if any(len(c['x'])!=d or any(not math.isfinite(v)for v in c['x']+[c['y']])for c in m['centers'].values()):raise ValueError('Invalid response centers')
    error=data['validation']['max_error_ms']
    if error is not None and(not math.isfinite(error)or error<0):raise ValueError('Invalid response error envelope')
    data['artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return data


def scheduler_features(state,cells,users):
    modes=state['scheduler_mode'];weights=state['ue_scheduler_weights']
    if set(modes)!=set(cells)or set(weights)!=set(users):raise ValueError('incomplete_scheduler_observation')
    if any(v not in('round_robin','latency_aware','qos_deadline')for v in modes.values()):raise ValueError('unknown_scheduler_mode')
    return [float(modes[c]==m)for c in cells for m in('latency_aware','qos_deadline')]+[float(weights[u])for u in users]


def estimate_change(artifact,base,target,*,horizon_ms):
    """Only return a joint supported direction; no fictitious per-UE derivative."""
    result=dict(delta=None,bounds=None,status='unavailable',issues=[],evidence=artifact['evidence'],
        model_sha256=artifact.get('artifact_sha256'),method=artifact['method'])
    if artifact['status']!='conditional':return dict(result,issues=['empirical_validation_failed'])
    context=artifact['context']
    for state in(base,target):
        if state['ue_drx_configs']!=context['native_drx']or state['ue_offload_fraction']!=context['native_offload']or any(
            v!='active'for v in state['cell_sleep_mode'].values())or set(state['cell_sleep_mode'])!=set(context['cells']):
            return dict(result,issues=['empirical_context_mismatch'])
    try:
        x=np.asarray(scheduler_features(base,context['cells'],context['users']))
        z=np.asarray(scheduler_features(target,context['cells'],context['users']))
    except ValueError:return dict(result,issues=['empirical_context_mismatch'])
    m=artifact['model'];delta=z-x;residual,norm=support_residual(m,delta)
    if residual>1e-7*max(1.,norm):return dict(result,issues=['empirical_unidentified_direction'],support_residual=residual)
    if not any(all(support_residual(m,v-np.asarray(c['x']))[0]<=1e-7*max(1.,np.linalg.norm(v-np.asarray(c['x'])))
        for v in(x,z))for c in m['centers'].values()):
        return dict(result,issues=['empirical_operating_point_outside_span'])
    low=np.asarray(m['feature_min']);high=np.asarray(m['feature_max'])
    if np.any(x<low-1e-8)or np.any(x>high+1e-8)or np.any(z<low-1e-8)or np.any(z>high+1e-8):
        return dict(result,issues=['empirical_outside_observed_range'])
    if horizon_ms!=artifact['horizon_ms']:return dict(result,issues=['empirical_horizon_mismatch'])
    predicted=float(delta@np.asarray(m['coefficients']))
    error=artifact['validation']['max_error_ms']
    # Two predictions form a contrast. This conservative descriptive envelope
    # is not a 95% CI or a probability of a network failure.
    return dict(result,delta=predicted,bounds=[predicted-2*error,predicted+2*error],status='conditional',
        issues=['empirical_observational_approximation','same_seed_validation_only'],support_residual=residual,
        validation_mae_ms=artifact['validation']['mae_ms'],error_envelope_ms=2*error,
        uncertainty=artifact['uncertainty'],identified_rank=m['rank'])


def context_compatible(artifact,config):
    context=artifact['context'];cfg=(config or {}).get('configs',{})
    return bool(config and config.get('source_sha256')==context.get('source_sha256')and
        cfg.get('phy',{}).get('rt_sys',{}).get('ue_services')==context['ue_services']and
        cfg.get('cell',{}).get('cell_radio_profiles')==context['cell_radio']and
        cfg.get('ue',{}).get('ues')==context['ue_config'])
