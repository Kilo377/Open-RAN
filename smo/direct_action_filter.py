"""Reject confirmed opposite writes to the same actuator, before risk scoring.

Reads captured shadow actions only. No model execution, arbitration, gain signs,
KPI forecasts, ranking or deployment. Unknown directions never mean opposite.
"""
from copy import deepcopy
from itertools import combinations

from smo.runtime import number

METHOD='opposed_direct_actions_v1'
SCALARS={'ue_offload_fraction','ue_scheduler_weights'}
TIMERS={'cycle_ms','on_duration_ms','inactivity_ms','wakeup_ms'}


def filter_catalog(catalog,graph,context):
    result=deepcopy(catalog)
    preview=context.get('preview') or {};frame=context.get('frame') or {}
    info=dict(method=METHOD,status='ready',decision_slot=preview.get('slot'),
        requested_slot=context.get('requested_slot'),lag_ms=context.get('lag_ms'),
        removed_primary_count=0,removed_reserve_count=0,excluded_group_count=0,
        rejected_pairs=[],unknown_apps=[],unresolved_shared_actions=[],
        action_values_used=False,model_inference_invoked=False,observed_outcomes_used=False,
        conflict_scores_used=False,arbitration_performed=False,deployment_enabled=False)
    result['direct_action_filter']=info
    if preview.get('status')=='unavailable' or not isinstance(preview.get('slot'),int) or preview.get('slot')!=frame.get('slot'):
        info.update(status='unavailable',reason='matching_decision_preview_unavailable')
        return result
    controls={}
    for e in graph.get('edges',[]):
        if e['kind']=='controls' and e.get('deployment')!='native_locked':
            controls.setdefault(e['source'],set()).add(e['target'].removeprefix('rcp:'))
    ids={a for g in catalog.get('groups',[])+catalog.get('reserve_groups',[])for a in g['app_ids']}
    apps={a['app_id']:a for a in preview.get('apps',[])};actions={}
    for ident in sorted(ids):
        app=apps.get(ident);reason=None
        if not app:reason='proposal_missing'
        elif app.get('action_contract_failures'):reason='invalid_proposal'
        elif app.get('status')=='no_action' and app.get('inference_invoked')is True and not app.get('actions'):
            actions[ident]={};continue
        elif app.get('status')!='proposed' or not app.get('actions'):reason='proposal_unavailable'
        if reason:
            info['unknown_apps'].append(dict(app_id=ident,reason=reason));continue
        mapped={};seen=set()
        for a in app['actions']:
            p=a.get('parameter');uid=a.get('ue_id');cid=a.get('cell_id');entity=uid or cid
            key=(p,entity)
            observed=frame.get('states',{}).get('rcp:'+str(p),{}).get('by_entity',{}).get(entity)
            if (p not in controls.get(ident,set()) or not entity or key in seen or a.get('current')is None
                or a.get('proposed')is None or observed is not None and observed!=a['current']):
                reason='invalid_or_mismatched_action';break
            seen.add(key);mapped[key]=a
        if reason:info['unknown_apps'].append(dict(app_id=ident,reason=reason))
        else:actions[ident]=mapped
    observation_slots={apps[a].get('observation_slot')for a in actions if apps[a].get('observation_slot')is not None}
    if len(observation_slots)>1:
        info.update(status='unavailable',reason='inconsistent_public_observation');return result
    info['action_values_used']=bool(actions)
    for left,right in combinations(sorted(actions),2):
        evidence=[]
        for key in sorted(actions[left].keys() & actions[right].keys()):
            a,b=actions[left][key],actions[right][key];p,entity=key
            if a['current']!=b['current'] or a.get('cell_id')!=b.get('cell_id'):
                info['unresolved_shared_actions'].append(dict(apps=[left,right],parameter=p,entity=entity,reason='inconsistent_action_baseline_or_membership'));continue
            if a['proposed']==b['proposed']:continue
            if p in SCALARS:
                fields=[(None,a['current'],a['proposed'],b['proposed'])]
            elif p=='ue_drx_configs' and all(isinstance(v,dict)for v in(a['current'],a['proposed'],b['proposed'])):
                fields=[(k,a['current'].get(k),a['proposed'].get(k),b['proposed'].get(k))for k in sorted(TIMERS)]
                if any(a['proposed'].get(k)!=b['proposed'].get(k)for k in (a['proposed'].keys()|b['proposed'].keys())-TIMERS):
                    info['unresolved_shared_actions'].append(dict(apps=[left,right],parameter=p,entity=entity,reason='unordered_structured_component'))
            else:
                fields=[]
            checked=False
            for component,current,first,second in fields:
                if not all(number(v)for v in(current,first,second)):continue
                checked=True
                da,db=first-current,second-current
                # Explicit signed coordinate movement, not KPI/capability direction.
                if da>0 and db<0 or da<0 and db>0:
                    evidence.append(dict(parameter=p,component=component,entity=entity,cell_id=a.get('cell_id'),
                        ue_id=a.get('ue_id'),current=deepcopy(current),
                        actions=[dict(app_id=left,proposed=first,delta=da,direction='increase'if da>0 else'decrease'),
                            dict(app_id=right,proposed=second,delta=db,direction='increase'if db>0 else'decrease')]))
            if not checked:
                info['unresolved_shared_actions'].append(dict(apps=[left,right],parameter=p,entity=entity,reason='no_ordered_numeric_direction'))
        if evidence:info['rejected_pairs'].append(dict(apps=[left,right],evidence=evidence))
    excluded=[]
    for pool,count in [('groups','removed_primary_count'),('reserve_groups','removed_reserve_count')]:
        kept=[]
        for g in result.get(pool,[]):
            matches=[pair for pair in info['rejected_pairs']if set(pair['apps'])<=set(g['app_ids'])]
            if not matches:kept.append(g);continue
            rejected=deepcopy(g);rejected['action_exclusion']=dict(reason='opposed_direct_control_actions',
                original_pool='primary'if pool=='groups'else'reserve',decision_slot=preview['slot'],pairs=deepcopy(matches))
            rejected.setdefault('initial_selection',{}).update(pool='excluded',reason='opposed_direct_control_actions')
            excluded.append(rejected);info[count]+=1
        if pool in result:result[pool]=kept
    result['excluded_groups']=excluded
    info['excluded_group_count']=len(excluded)
    result['group_count']=result['eligible_group_count']=len(result.get('groups',[]))
    result['reserve_group_count']=len(result.get('reserve_groups',[]))
    if result.get('initial_selection'):
        s=result['initial_selection']
        s['pre_action_filter_primary_group_count']=s['primary_group_count']
        s['pre_action_filter_reserve_group_count']=s['reserve_group_count']
        s.update(primary_group_count=result['group_count'],reserve_group_count=result['reserve_group_count'],
            excluded_group_count=len(excluded),current_actions_used=info['action_values_used'],action_values_used=info['action_values_used'])
    return result
