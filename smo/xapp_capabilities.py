"""Static xApp optimization directions, never current parameter actions.

Primary directions come from canonical released objectives. Optional secondary
directions require audited, model/scene/scope-matched frozen-policy ON/OFF
evidence. A parameter derivative cannot establish a policy capability direction.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


OBJECTIVE_DIRECTIONS = {'maximize':['increase'], 'minimize':['decrease'],
    'upper_bound':['decrease'], 'lower_bound':['increase'],
    'bidirectional':['increase','decrease'], 'maintain':['maintain']}


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def required_directions(row):
    """Read frozen interval semantics, not the latest proposed action or trend.

    A one-sided upper limit remains a decreasing capability requirement even
    when the latest value already satisfies it. For a bounded band, either
    direction is an initial candidate; staying inside is a later value check.
    """
    low,high=row.get('lower'),row.get('upper')
    domain=row.get('domain',[0.,None])
    lower_active=finite(low) and (low>domain[0] or low==domain[0] and not row.get('lower_inclusive',True))
    upper_active=finite(high) and (domain[1] is None or high<domain[1] or high==domain[1] and not row.get('upper_inclusive',True))
    if lower_active and not upper_active:
        return dict(directions=['increase'],basis='lower_bounded_intent')
    if upper_active and not lower_active:
        return dict(directions=['decrease'],basis='upper_bounded_intent')
    return dict(directions=['increase','decrease','maintain'],
        basis='bounded_band_requires_later_value_validation' if lower_active and upper_active else 'unconstrained_domain')


def build_capabilities(graph, catalog_path=None):
    apps={n['id']:n for n in graph['nodes'] if n['kind']=='xapp'}
    kpis={n['key'] for n in graph['nodes'] if n['kind']=='kpi'}
    # Small analytic graphs can omit KPI nodes; the real graph has explicit ones.
    kpis.update(e['target'][4:] for e in graph['edges'] if e['kind']=='influences')
    profiles={app:{} for app in apps}
    for app,node in apps.items():
        for kpi,objective in node.get('target_kpis',{}).items():
            directions=OBJECTIVE_DIRECTIONS.get(objective,[])
            if not directions:
                continue
            profiles[app][kpi]=dict(directions=directions[:],status='declared_objective',
                evidence=[dict(kind='released_objective',objective=objective,
                    source=node.get('contract_source'),model_sha256=node.get('model_sha256'))],
                effectiveness_guaranteed=False)
    source=None;rejected=[];applied=[]
    if catalog_path:
        path=Path(catalog_path);raw=path.read_bytes();data=json.loads(raw)
        if data.get('schema')!='oran.xapp-capabilities.v1' or not isinstance(data.get('estimates'),list):
            raise ValueError('Expected an audited xApp capability catalog')
        source=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())
        scenes={s.get('scenario_id') for e in graph['edges'] for s in
            (e.get('sensitivities') or ([e['sensitivity']] if e.get('sensitivity') else [])) if s.get('scenario_id')}
        for index,row in enumerate(data['estimates']):
            app=row.get('app_id');kpi=row.get('kpi');node=apps.get(app)
            reason=None
            if node is None or kpi not in kpis:reason='unknown_app_or_kpi'
            elif row.get('model_sha256')!=node.get('model_sha256') or not node.get('model_sha256'):reason='different_frozen_model'
            elif row.get('runtime_parameters')!=node.get('runtime_parameters',{}):reason='different_runtime_parameters'
            elif row.get('metric_scope')!=graph.get('metric_scope'):reason='different_metric_scope'
            elif len(scenes)!=1 or row.get('scenario_id') not in scenes:reason='different_or_unknown_scene'
            if reason:
                rejected.append(dict(index=index,app_id=app,kpi=kpi,reason=reason));continue
            samples=row.get('samples',[]);ci=row.get('confidence_interval');direction=row.get('direction')
            if row.get('kind')!='matched_frozen_policy_on_off' or direction not in ('increase','decrease'):
                raise ValueError('Secondary capability requires matched frozen-policy ON/OFF evidence')
            if len(samples)<2 or any(type(s.get('seed'))is not int or not finite(s.get('before'))or not finite(s.get('after'))for s in samples) or len({s['seed']for s in samples})!=len(samples):
                raise ValueError('Secondary capability requires at least two distinct paired seeds')
            mean=sum(s['after']-s['before']for s in samples)/len(samples)
            if not isinstance(ci,list) or len(ci)!=2 or not all(map(finite,ci))or not ci[0]<=mean<=ci[1]:
                raise ValueError('Invalid frozen-policy effect interval')
            if not (ci[0]>0 and direction=='increase' or ci[1]<0 and direction=='decrease'):
                rejected.append(dict(index=index,app_id=app,kpi=kpi,reason='policy_direction_not_resolved'));continue
            reference=row.get('source',{});evidence_path=Path(reference.get('path',''))
            if not evidence_path.is_absolute() or not evidence_path.is_file() or hashlib.sha256(evidence_path.read_bytes()).hexdigest()!=reference.get('sha256'):
                raise ValueError('Secondary capability source hash does not match')
            record=dict(kind=row['kind'],source=deepcopy(reference),model_sha256=row['model_sha256'],
                scenario_id=row['scenario_id'],metric_scope=row['metric_scope'],direction=direction,
                samples=deepcopy(samples),confidence_interval=ci[:])
            cell=profiles[app].get(kpi)
            if cell is None:
                profiles[app][kpi]=dict(directions=[direction],status='validated_secondary',evidence=[record],effectiveness_guaranteed=False)
            else:
                cell['evidence'].append(record)
                if direction not in cell['directions']:
                    cell.update(directions=[],status='conflicting_capability_evidence')
            applied.append(dict(index=index,app_id=app,kpi=kpi,direction=direction))
    return dict(schema='oran.xapp-capability-profiles.v1',profiles=profiles,
        secondary_source=source,applied_secondary=applied,rejected_secondary=rejected,
        current_actions_used=False,parameter_gain_signs_used=False,model_inference_invoked=False)
