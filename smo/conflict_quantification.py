"""Current method: read-only, proposal-conditioned first-order conflict review.

No model inference, random draw, control execution, pruning or ranking happens
here. Population gains are applied once to their measured common intervention,
never once per UE. Missing evidence is not a zero effect or a safe outcome.
"""
from copy import deepcopy
from itertools import combinations, product
import hashlib
import json
import math

from smo.runtime import number

METHOD = 'proposal_first_order_v1'
PARAMETERS = ('ue_offload_fraction','ue_drx_configs','scheduler_mode','ue_scheduler_weights','cell_sleep_mode')
SCHEDULER = frozenset(('scheduler_mode','ue_scheduler_weights'))


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def cancellation(a,b):
    """Opposition proportion AND dimensional cancelled amount; no epsilon score."""
    if not number(a) or not number(b):return dict(ratio=None,amount=None)
    if a*b>=0:return dict(ratio=0.,amount=0.)
    amount=min(abs(a),abs(b))
    return dict(ratio=2*amount/(abs(a)+abs(b)),amount=amount)


def interval(value,row):
    if not number(value) or row.get('empty'):return dict(status='unavailable',gap=None,margin=None)
    low,high=row.get('lower'),row.get('upper')
    distances=([value-low]if number(low)else[])+([high-value]if number(high)else[])
    bad=(number(low)and(value<low or value==low and not row.get('lower_inclusive',True)))or(
        number(high)and(value>high or value==high and not row.get('upper_inclusive',True)))
    return dict(status='outside'if bad else'inside',gap=max([0.]+[-d for d in distances]),
        margin=min(distances)if distances else None)


def memberships(frame,config=None):
    states=frame.get('states',{})
    by=lambda p:deepcopy(states.get('rcp:'+p,{}).get('by_entity',{}))
    cfg=(config or {}).get('configs',{})
    ues=cfg.get('ue',{}).get('ues',[])
    cells=[c['cell_id']for c in(config or {}).get('manifest',{}).get('cells',[])]
    tasks=cfg.get('phy',{}).get('rt_sys',{}).get('ue_services',{}).get('mec',{}).get('tasks_by_ue')
    return dict(cells=sorted(cells or by('scheduler_mode')),mec=sorted(tasks if tasks is not None else by('ue_offload_fraction')),
        drx=sorted(u for u,v in by('ue_drx_configs').items()if isinstance(v,dict)and v.get('enabled')),
        urllc=sorted(u['ue_id']for u in ues if u.get('slice_id')=='urllc'))


def _values(frame):
    return {p:deepcopy(frame.get('states',{}).get('rcp:'+p,{}).get('by_entity',{}))for p in PARAMETERS}


def _vector(base,actions):
    result=deepcopy(base)
    for a in actions:
        p=a['parameter'];entity=a.get('ue_id')or a.get('cell_id')
        result[p][entity]=deepcopy(a['proposed'])
    return result


def _group(row,p,groups,base):
    component=row.get('parameter_component')
    if component=='all_cells':return groups.get('cells',[])
    if component in ('mec_workload','mec_priority'):return groups.get('mec',[])
    if component=='drx_enabled_population':return groups.get('drx',[])
    if component=='urllc_priority':return groups.get('urllc',[])
    # Explicit member sets are supported for future measured interventions.
    return row.get('members',[])


def _common(values):
    if not values or any(v is None for v in values):return None
    return values[0]if all(v==values[0]for v in values)else None


def _context_issues(row,base,target):
    expected='latency_aware'if row.get('parameter_component')=='mec_priority'else(
        'qos_deadline'if row.get('parameter_component')=='urllc_priority'else'round_robin')
    issues=[]
    if row['parameter']!='scheduler_mode'and any(v!=expected for v in target['scheduler_mode'].values()):
        issues.append('operating_context_shift')
    if any(v!='active'for v in base['cell_sleep_mode'].values()):issues.append('operating_context_shift')
    if row['parameter']!='ue_offload_fraction'and any(v!=.5 for v in base['ue_offload_fraction'].values()):
        issues.append('operating_context_shift')
    if row['parameter']!='ue_scheduler_weights'and any(v!=1 for v in base['ue_scheduler_weights'].values()):
        issues.append('operating_context_shift')
    if any(v.get('enabled')and (v.get('inactivity_ms')!=8 or v.get('cycle_ms')!=40 or v.get('on_duration_ms')!=4)
           for v in base['ue_drx_configs'].values()if isinstance(v,dict)):
        issues.append('operating_context_shift')
    if sum(base[p]!=target[p]for p in PARAMETERS)>1:issues.append('additive_multi_parameter_approximation')
    return sorted(set(issues))


def response(graph,base,target,p,kpi,groups,horizon_ms,show_dashed=False):
    """One measured common-population response, conditional on the actual step."""
    result=dict(parameter=p,kpi=kpi,delta=None,bounds=None,status='unavailable',issues=[])
    changed={u for u in base[p].keys()|target[p].keys()if base[p].get(u)!=target[p].get(u)}
    if not changed:return dict(result,delta=0.,bounds=[0.,0.],status='unchanged')
    edge=next((e for e in graph['edges']if e['kind']=='influences'and e['source']=='rcp:'+p and e['target']=='kpi:'+kpi),None)
    rows=(edge.get('sensitivities')or([edge['sensitivity']]if edge.get('sensitivity')else[]))if edge else[]
    reasons=[];matches=[]
    for row in rows:
        if row.get('population_control_eligible')is not True:continue
        members=_group(row,p,groups,base)
        if not members or not changed<=set(members):
            reasons.append('population_scope_mismatch');continue
        component=row.get('scalar_component')
        if p=='ue_drx_configs':
            component=component or'inactivity_ms'
            if any(not isinstance(base[p].get(u),dict)or not isinstance(target[p].get(u),dict)or
                {k:v for k,v in base[p][u].items()if k!=component}!={k:v for k,v in target[p][u].items()if k!=component}
                for u in members):
                reasons.append('unmeasured_structured_change');continue
            old=_common([base[p][u].get(component)for u in members]);new=_common([target[p][u].get(component)for u in members])
        else:
            old=_common([base[p].get(u)for u in members]);new=_common([target[p].get(u)for u in members])
        if old is None or new is None:
            reasons.append('nonuniform_population_action');continue
        if row.get('kind')=='finite_contrast':
            start,end=row.get('from'),row.get('to')
            if isinstance(start,dict):start=start.get(component)
            if isinstance(end,dict):end=end.get(component)
            step=1. if old==start and new==end else-1. if old==end and new==start else None
            if step is None:reasons.append('unmeasured_categorical_transition');continue
        else:
            if not number(old)or not number(new):continue
            domain=row.get('fit_domain')or row.get('sampled_domain')
            if not domain or not domain[0]<=old<=domain[1]or not domain[0]<=new<=domain[1]:
                reasons.append('outside_measured_domain');continue
            step=new-old
        estimate=row.get('estimate');ci=row.get('confidence_interval')
        if not number(estimate)or not isinstance(ci,list)or len(ci)!=2 or not all(map(number,ci))or ci[0]>ci[1]:
            reasons.append('missing_gain_uncertainty');continue
        issues=_context_issues(row,base,target)
        resolved=ci[0]>0 or ci[1]<0 or estimate==0 and ci==[0,0]
        if not resolved:
            if not show_dashed:reasons.append('uncertain_gain_hidden');continue
            issues.append('gain_direction_uncertain')
        if row.get('scoring_window_ms')!=horizon_ms:issues.append('response_window_mismatch')
        if row.get('fit_diagnostics',{}).get('assessment')=='limited_first_order_approximation':
            issues.append('limited_linear_fit')
        matches.append(dict(result,delta=estimate*step,bounds=sorted(v*step for v in ci),
            status='conditional'if issues else'estimated',issues=sorted(set(issues)),
            edge_id=edge['id'],control_members=members,parameter_delta=step,
            gain=dict(estimate=estimate,confidence_interval=ci,contrast_id=row.get('contrast_id'),
                kind=row.get('kind'),unit=row.get('unit'),component=row.get('parameter_component'),
                scalar_component=component,scoring_window_ms=row.get('scoring_window_ms'),
                horizon_ms=row.get('horizon_ms'),context=row.get('context'),data_sha256=row.get('data_sha256'))))
    if len(matches)==1:return matches[0]
    return dict(result,issues=['ambiguous_calibration']if matches else sorted(set(reasons))or['gain_not_measured'])


def _sum(links):
    known=[l for l in links if number(l['delta'])]
    complete=len(known)==len(links)
    return dict(delta=math.fsum(l['delta']for l in known)if complete else None,
        known_delta=math.fsum(l['delta']for l in known),
        bounds=[math.fsum(l['bounds'][i]for l in known)for i in(0,1)]if complete else None,
        complete=complete,issues=sorted({s for l in links for s in l['issues']}))


def _units(app):
    """Mirror atomic scheduler arbitration; keep complete mode/UID-map requests."""
    units={}
    for a in app.get('actions',[]):
        p=a['parameter'];entity=a.get('ue_id')or a.get('cell_id')
        key=('scheduler_bundle',a['cell_id'])if p in SCHEDULER else(p,entity)
        unit=units.setdefault(key,dict(resource=list(key),app_id=app['app_id'],actions=[],value={}))
        unit['actions'].append(a)
        if p in SCHEDULER:
            if p=='scheduler_mode':unit['value'][p]=a['proposed']
            else:unit['value'].setdefault(p,{})[a['ue_id']]=a['proposed']
        else:unit['value']=a['proposed']
    return units


def quantify_catalog(graph,frame,catalog,preview,*,config=None,horizon_ms=250.,show_dashed=False,max_scenarios=256,empirical_model=None):
    """Quantify every supplied group without changing membership or ordering."""
    result=dict(schema='oran.conflict-quantification.v1',method=METHOD,
        snapshot={k:frame.get(k)for k in('slot','time_ms')},preview_slot=preview.get('slot'),
        report_slots=sorted({a['report_produced_at_slot']for a in preview.get('apps',[])if type(a.get('report_produced_at_slot'))is int}),
        assessment_horizon_ms=horizon_ms,show_dashed=show_dashed,status='ready',
        automatic_selection=False,ranking_performed=False,pruning_performed=False,deployment_enabled=False,
        mitigation='uniform_random_per_resource',probability_interpretation='instantaneous_frozen_proposals_only',
        uncertainty_interpretation='coefficient_interval_envelope_not_joint_confidence_or_SLA_probability',
        links=[],apps=[],groups=[],joint_effects=[])
    empirical_ok=False
    if empirical_model is not None:
        from smo.mec_empirical import context_compatible,estimate_change
        empirical_ok=context_compatible(empirical_model,config)
        result['empirical_response']={k:deepcopy(empirical_model.get(k))for k in('schema','evidence','status','artifact_sha256','coverage','independent_seeds')}
        result['empirical_response']['validation']={k:v for k,v in empirical_model['validation'].items()if k!='predictions'}
        result['empirical_response']['context_matches']=empirical_ok
    ric=(config or {}).get('configs',{}).get('ric',{})
    result['mitigation_window_ms']=ric.get('mitigation_window_ms',100)
    if ric.get('direct_conflict_mitigation','uniform_random')!='uniform_random':
        result.update(status='unavailable',reason='unsupported_mitigation_mode');return result
    result['input_sha256']=hashlib.sha256(encoded(dict(graph=graph,frame=frame,catalog=catalog,preview=preview,
        groups=memberships(frame,config),horizon_ms=horizon_ms,show_dashed=show_dashed,
        empirical_model_sha256=(empirical_model or{}).get('artifact_sha256'))).encode()).hexdigest()
    if preview.get('slot')!=frame.get('slot')or not preview.get('apps'):
        result.update(status='unavailable',reason='matching_decision_preview_unavailable');return result
    base=_values(frame)
    apps={a['app_id']:a for a in preview['apps']};required=catalog.get('required_kpis',[])
    candidate_ids={a for g in catalog.get('groups',[])for a in g['app_ids']}
    # Consistent common input and exact parameter readings are required.
    observations={a.get('observation_slot')for a in apps.values()if type(a.get('observation_slot'))is int}
    if len(observations)>1:
        result.update(status='unavailable',reason='inconsistent_public_observation');return result
    supplemented=[]
    for a in apps.values():
        for action in a.get('actions',[]):
            p=action.get('parameter');entity=action.get('ue_id')or action.get('cell_id')
            if p not in base or action.get('current')is None or entity in base[p]and base[p][entity]is not None and base[p][entity]!=action['current']:
                result.update(status='unavailable',reason='parameter_observation_mismatch');return result
            if entity not in base[p]or base[p][entity]is None:
                base[p][entity]=deepcopy(action['current'])
                supplemented.append(dict(parameter=p,entity=entity,source='captured_public_report'))
    complete_frame=dict(states={'rcp:'+p:dict(by_entity=v)for p,v in base.items()})
    groups=memberships(complete_frame,config)
    result['supplemented_parameter_readings']=supplemented
    allowed={(e['source'],e['target'][4:])for e in graph['edges']if e['kind']=='controls'and e.get('deployment')!='native_locked'}
    available={i:apps[i]for i in candidate_ids if i in apps and apps[i].get('status')=='proposed'and not apps[i].get('action_contract_failures')
        and apps[i].get('actions')and all((i,a['parameter'])in allowed for a in apps[i]['actions'])
        and len({(a['parameter'],a.get('ue_id')or a.get('cell_id'))for a in apps[i]['actions']})==len(apps[i]['actions'])}
    available={i:a for i,a in available.items()if all(u['resource'][0]!='scheduler_bundle'or
        set(u['value'])==SCHEDULER for u in _units(a).values())}
    def complete_effect(controlled,target,key):
        changed={p for p in PARAMETERS if base[p]!=target[p]}
        if controlled['complete']or key!='mec_offloaded_latency_ms'or not empirical_ok or not changed<=SCHEDULER:
            return controlled
        empirical=estimate_change(empirical_model,base,target,horizon_ms=horizon_ms)
        if empirical['delta']is None:
            return dict(controlled,issues=sorted(set(controlled['issues']+empirical['issues'])))
        return dict(empirical,complete=True,known_delta=empirical['delta'],
            controlled_gain_unavailable_reasons=controlled['issues'])
    app_links={};unit_maps={i:_units(a)for i,a in available.items()}
    for ident in sorted(candidate_ids):
        a=available.get(ident)
        links=[]
        if a:
            target=_vector(base,a['actions'])
            for p in sorted({x['parameter']for x in a['actions']}):
                control=next((e for e in graph['edges']if e['kind']=='controls'and e['source']==ident and e['target']=='rcp:'+p and e.get('deployment')!='native_locked'),None)
                if control is None:continue
                for key in required:
                    link=dict(app_id=ident,control_edge_id=control['id'],**response(graph,base,target,p,key,groups,horizon_ms,show_dashed))
                    link['current']=deepcopy(base[p]);link['proposed']=deepcopy(target[p]);links.append(link)
        app_links[ident]=links;result['links'].extend(links)
        effects=[]
        if a:
            for key in required:
                effect=complete_effect(_sum([l for l in links if l['kpi']==key]),target,key)
                effects.append(dict(kpi=key,**effect))
                if effect.get('evidence'):result['joint_effects'].append(dict(app_id=ident,kpi=key,**effect))
        result['apps'].append(dict(app_id=ident,status='available'if a else'unavailable',
            reason=None if a else apps.get(ident,{}).get('reason')or'no_fresh_proposal',
            effects=effects))
    for group in catalog.get('groups',[]):
        ids=group['app_ids'];resources={};missing=[i for i in ids if i not in available]
        for ident in ids:
            for key,unit in unit_maps.get(ident,{}).items():resources.setdefault(key,[]).append(unit)
        direct=[];compatible=0;options=[];unsupported=False
        for key,units in sorted(resources.items()):
            variants={}
            for unit in units:
                signature=encoded(unit['value'])
                if signature not in variants:variants[signature]=dict(actions=unit['actions'],probability=0.)
                variants[signature]['probability']+=1/len(units)
            if len(units)>1:
                if len(variants)>1:
                    supported=key[0]in('ue_offload_fraction','scheduler_bundle');unsupported|=not supported
                    direct.append(dict(resource=list(key),apps=[u['app_id']for u in units],distinct_values=len(variants),
                        candidates=[dict(app_id=u['app_id'],value=u['value'],probability=1/len(units)if supported else None)for u in units],
                        mitigation_supported=supported,action_disagreement=True))
                else:compatible+=1
            options.append(list(variants.values()))
        indirect=[]
        for a,b in combinations(ids,2):
            # Shared RCPs are handled by the arbiter, not additive cancellation.
            shared={l['parameter']for l in app_links[a]}&{l['parameter']for l in app_links[b]}
            for key in required:
                left=[l for l in app_links[a]if l['kpi']==key and l['parameter']not in shared]
                right=[l for l in app_links[b]if l['kpi']==key and l['parameter']not in shared]
                if not left or not right:continue
                x,y=_sum(left),_sum(right);c=cancellation(x['delta'],y['delta'])
                robust=bool(x['bounds']and y['bounds']and(x['bounds'][1]<0<y['bounds'][0]or y['bounds'][1]<0<x['bounds'][0]))
                indirect.append(dict(apps=[a,b],kpi=key,**c,left_delta=x['delta'],right_delta=y['delta'],
                    opposition_robust_to_gain_intervals=robust,
                    status='unavailable'if c['ratio']is None else'conditional'if x['issues']or y['issues']else'estimated',
                    issues=sorted(set(x['issues']+y['issues'])),
                    parameters=sorted({l['parameter']for l in left+right}),
                    edge_ids=sorted({e for l in left+right for e in(l.get('edge_id'),l.get('control_edge_id'))if e})))
        scenario_count=math.prod(len(o)for o in options);exhaustive=scenario_count<=max_scenarios
        outcomes={k:[]for k in required};unknown={k:bool(missing or not exhaustive or unsupported)for k in required}
        issues={k:set()for k in required};ranges={r['kpi']:r for r in catalog.get('ranges',[])};used_empirical=set()
        if exhaustive:
            for selected in product(*options):
                target=_vector(base,[a for s in selected for a in s['actions']]);probability=math.prod(s['probability']for s in selected)
                for key in required:
                    effects=[response(graph,base,target,p,key,groups,horizon_ms,show_dashed)for p in PARAMETERS if base[p]!=target[p]]
                    total=complete_effect(_sum(effects),target,key);issues[key].update(total['issues'])
                    if total.get('evidence'):used_empirical.add(key)
                    current=frame.get('states',{}).get('kpi:'+key,{}).get('value')
                    if not total['complete']or not number(current):unknown[key]=True;continue
                    value=current+total['delta'];bounds=[current+d for d in total['bounds']]
                    outcomes[key].append(dict(value=value,bounds=bounds,probability=probability,**interval(value,ranges[key])))
        kpis=[]
        for key in required:
            cases=outcomes[key];complete=not unknown[key]and bool(cases)
            kpis.append(dict(kpi=key,current=frame.get('states',{}).get('kpi:'+key,{}).get('value'),
                range=deepcopy(ranges[key]),status='conditional'if complete and issues[key]else'estimated'if complete else'unavailable',
                predicted_min=min(c['value']for c in cases)if complete else None,
                predicted_max=max(c['value']for c in cases)if complete else None,
                gain_envelope=[min(c['bounds'][0]for c in cases),max(c['bounds'][1]for c in cases)]if complete else None,
                violation_probability=sum(c['probability']for c in cases if c['status']=='outside')if complete else None,
                worst_gap=max(c['gap']for c in cases)if complete else None,
                minimum_margin=min(c['margin']for c in cases)if complete else None,
                evidence='observational_executed_control_regression'if key in used_empirical else'controlled_parameter_gains',
                validation_mae_ms=empirical_model['validation']['mae_ms']if key in used_empirical else None,
                issues=sorted(issues[key]|({'proposal_unavailable'}if missing else set())|({'scenario_limit'}if not exhaustive else set())|
                    ({'unsupported_resource_arbitration'}if unsupported else set()))))
        result['groups'].append(dict(group_id=group['id'],app_ids=ids,direct=direct,compatible_shared_resources=compatible,
            indirect=indirect,scenario_count=scenario_count,scenario_enumeration_complete=exhaustive,
            missing_proposal_apps=missing,kpis=kpis,
            summary=dict(direct_disagreements=len(direct),opposing_indirect_pairs=sum((r['ratio']or 0)>0 for r in indirect),
                max_cancellation_ratio=max((r['ratio']for r in indirect if r['ratio']is not None),default=None),
                unresolved_indirect_pairs=sum(r['ratio']is None for r in indirect),
                quantified_kpis=sum(k['status']!='unavailable'for k in kpis),required_kpis=len(required),
                proposals_available=len(ids)-len(missing),proposals_required=len(ids),direct_evaluation_complete=not missing),
            interpretation='instantaneous_local_response_not_simulated_or_guaranteed_attainment'))
    result['status']='partial'if any(g['summary']['quantified_kpis']<len(required)for g in result['groups'])else'ready'
    return result
