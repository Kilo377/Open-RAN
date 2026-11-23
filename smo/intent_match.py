"""Legacy action-space previews, retained for offline experiment audits.

The active GraphService uses graph_combinations and never invokes this matcher.
No learner, controller or executor is imported here either.
"""
from copy import deepcopy
import math

from smo.intent_language import finite


MEC = ('ue_0', 'ue_4')
URLLC = ('ue_2', 'ue_6')
USERS = ('ue_0', 'ue_1', 'ue_2', 'ue_4', 'ue_6', 'ue_8')


def profiles(app):
    """Released action families; group semantics stay explicit, never averaged."""
    key = app['key']
    def part(parameter, value, group='all_cells', members=(), scope='All cells'):
        return dict(parameter=parameter, value=value, group=group, members=list(members), scope=scope)
    def scheduler(mode, weights, group, members, scope):
        mode_part = part('scheduler_mode', mode, 'eligible_cells' if key == 'x12_mec' else 'all_cells',
                         scope='MEC-serving cells' if key == 'x12_mec' else 'All cells')
        priority = part('ue_scheduler_weights', weights, group, members, scope)
        priority['context_approximation'] = key == 'x12_mec'
        result = [mode_part, priority]
        if key == 'x12_urllc':
            result.append(part('ue_scheduler_weights', 1., 'other_traffic_reset', tuple(u for u in USERS if u not in members), 'Other traffic priorities reset to 1'))
        return result
    if key in ('x01', 'x13'):
        members = MEC if key == 'x01' else tuple(app.get('target_ues') or ('ue_4',))
        values = (0., .5, 1.) if key == 'x01' else (0., .5)
        scope = 'MEC workload' if key == 'x01' else 'Budget-controlled MEC workload'
        drx=app.get('drx_profiles') if key=='x01' and app.get('runtime_parameters',{}).get('deploy_drx') else []
        if drx:
            return [dict(name='DRX '+('/'.join(str(round(profile[k])) for k in ('cycle_ms','on_duration_ms','inactivity_ms')))+' ms · Offload '+str(round(v*100))+'%',
                         parts=[part('ue_offload_fraction',v,'mec_workload',members,scope),
                                part('ue_drx_configs',dict(profile),'mec_drx_profile',members,scope)]) for profile in drx for v in values]
        return [dict(name='Offloading probability '+str(round(v*100))+'%',
                     parts=[part('ue_offload_fraction', v, 'mec_workload', members, scope)]) for v in values]
    if key in ('x12_mec', 'x12_urllc'):
        mec = key == 'x12_mec'; mode = 'latency_aware' if mec else 'qos_deadline'
        weights = (.5, 1., 2.) if mec else (1., 2., 4.)
        group, members, scope = ('mec_priority', MEC, 'MEC workload') if mec else ('urllc_priority', URLLC, 'URLLC traffic')
        return [dict(name='Native round robin', parts=scheduler('round_robin', 1., group, members, scope))] + [
            dict(name=('MEC' if mec else 'URLLC')+' priority '+str(w),
                 parts=scheduler(mode, w, group, members, scope)) for w in weights]
    if key == 'x08':
        reset = part('ue_scheduler_weights', 1., 'all_users_reset', USERS, 'All traffic priorities reset to 1')
        return [dict(name='Native round robin', parts=[part('scheduler_mode', 'round_robin'), reset]),
                dict(name='Neutral latency-aware policy', parts=[part('scheduler_mode', 'latency_aware'), reset])] + [
            dict(name='Efficiency-first priority '+str(w), parts=[part('scheduler_mode', 'latency_aware'), reset,
                part('ue_scheduler_weights', w, 'efficiency_priority', (), 'Public efficiency-selected users')]) for w in (1.5, 2.)]
    if key == 'x02':
        return [dict(name='All cells active', parts=[part('cell_sleep_mode', 'active')]),
                dict(name='Deep sleep for empty cells', parts=[part('cell_sleep_mode', 'deep_sleep', 'empty_cells', scope='Empty cells only')]),
                dict(name='Micro-sleep occupied cells', parts=[part('cell_sleep_mode', 'micro_sleep', 'occupied_cells', scope='Occupied cells; empty cells use deep sleep')]),
                dict(name='Light-sleep occupied cells', parts=[part('cell_sleep_mode', 'light_sleep', 'occupied_cells', scope='Occupied cells; empty cells use deep sleep')])]
    return []


def _current(part, frame):
    state = frame.get('states', {}).get('rcp:'+part['parameter'], {})
    values = state.get('by_entity', {})
    if part['group'] == 'efficiency_priority':
        return None  # Public efficiency selection requires the actual observation.
    chosen = [values.get(uid) for uid in part['members']] if part['members'] else list(values.values())
    if chosen and all(v is not None and v == chosen[0] for v in chosen):
        return chosen[0]
    return state.get('value') if not part['members'] else None


def _rows(graph, parameter, kpi=None):
    for edge in graph['edges']:
        if edge['kind'] == 'influences' and edge['source'] == 'rcp:'+parameter and (kpi is None or edge['target'] == 'kpi:'+kpi):
            for row in edge.get('sensitivities', [edge['sensitivity']] if edge.get('sensitivity') else []):
                yield edge, row


def _effect(edge, row, part, current):
    if not finite(row.get('estimate')):
        return None
    factor = None
    if row['kind'] == 'range_linear_fit':
        if row.get('parameter_component') != part['group'] or not finite(current) or not finite(part['value']):
            return None
        low, high = row['fit_domain']
        if not low <= current <= high or not low <= part['value'] <= high:
            return None
        factor = part['value'] - current
    elif row['kind'] == 'finite_contrast':
        if current == row['from'] and part['value'] == row['to']:
            factor = 1.
        elif current == row['to'] and part['value'] == row['from']:
            factor = -1.
    if factor is None:
        return None
    value = row['estimate'] * factor
    ci = row.get('confidence_interval')
    ci = sorted(x * factor for x in ci) if ci is not None else None
    resolved = ci is not None and (ci[0] > 0 or ci[1] < 0)
    constant = row.get('fit_diagnostics', {}).get('assessment') == 'constant_response_observed' or row.get('zero_in_both') is True
    partial = part['group'] in ('occupied_cells', 'empty_cells', 'eligible_cells') or part.get('context_approximation', False) or part['parameter'] == 'ue_offload_fraction' and set(part['members']) != set(MEC)
    return dict(kpi=row['kpi'], edge_id=edge['id'], contrast_id=row.get('contrast_id'),
                delta=value, delta_interval=ci,
                direction='increase' if value > 0 else 'decrease' if value < 0 else 'no_change',
                evidence='no_observed_change' if constant else 'resolved_average' if resolved else 'uncertain_direction',
                coefficient=row['estimate'], coefficient_unit=row['unit'],
                coefficient_interval=row.get('confidence_interval'),
                nonlinear=row.get('fit_diagnostics', {}).get('assessment') == 'limited_first_order_approximation',
                scope_approximation=partial, context=row['context'],
                data_sha256=row['data_sha256'], sample_count=row['sample_count'])


def _compatible(row, part, profile):
    if part['parameter']=='ue_drx_configs':
        return row['kind']=='finite_contrast' and row.get('parameter_component')==part['group']
    if row['kind'] != 'range_linear_fit' or part['parameter'] != 'ue_scheduler_weights':
        return True
    mode = next((p['value'] for p in profile['parts'] if p['parameter'] == 'scheduler_mode'), None)
    expected = 'latency_aware' if part['group'] == 'mec_priority' else 'qos_deadline' if part['group'] == 'urllc_priority' else None
    return mode == expected


def _owns_target(app, target):
    # Reverse KPI ownership is structural. A policy's maximize/minimize label
    # cannot establish compatibility with a numerical, possibly bounded band.
    return target['kpi'] in app.get('target_kpis', {})


def match_intent(graph, frame, interpreted):
    """Return capability previews; signs concern individual tested paths, not a bundled guarantee."""
    targets = interpreted['ranges']; needs = [t for t in targets if not t['empty']]
    nodes = {n['id']: n for n in graph['nodes']}
    energy=next((n for n in graph['nodes']if n.get('key')=='x01' and n['kind']=='xapp'),{})
    native_drx=next(iter(energy.get('drx_profiles')or[]),None)
    current_drx=_current(dict(parameter='ue_drx_configs',group='mec_drx_profile',members=MEC),frame) if native_drx else None
    response = dict(schema='oran.intent-review.v1', ranges=deepcopy(targets), questions=interpreted['questions'],
                    snapshot=dict(slot=frame.get('slot'), time_ms=frame.get('time_ms')),
                    candidates=[], excluded_paths=[], deployment_enabled=False,
                    note='Capability previews, not frozen-policy decisions. Coupled actions and interval attainment require validation.')
    for edge in graph['edges']:
        if edge['kind'] == 'controls' and edge.get('deployment') == 'native_locked':
            response['excluded_paths'].append(dict(app=edge['source'], parameter=edge['target'], reason='Native-locked in the released policy'))
    if interpreted['questions'] or any(t['empty'] for t in targets) or not needs:
        return response
    for app in (n for n in graph['nodes'] if n['kind'] == 'xapp'):
        controls = {e['target'][4:]: e for e in graph['edges'] if e['kind'] == 'controls' and e['source'] == app['id'] and e.get('deployment') != 'native_locked'}
        variants = []
        for index, profile in enumerate(profiles(app)):
            actions, effects, context_changed = [], [], False
            proposed_drx=next((p['value']for p in profile['parts']if p['parameter']=='ue_drx_configs'),native_drx)
            drx_context_differs=native_drx is not None and (current_drx!=native_drx or proposed_drx!=native_drx)
            for p in profile['parts']:
                if p['parameter'] not in controls:
                    continue
                before = _current(p, frame)
                if before == p['value']:
                    continue
                action = dict(parameter=p['parameter'], label=nodes['rcp:'+p['parameter']]['label'],
                              before=before, after=p['value'], scope=p['scope'], group=p['group'], control_edge=controls[p['parameter']]['id'])
                if p['group'] == 'efficiency_priority':
                    action['note'] = 'Weights are selected from public efficiency reports; service-class gains do not calibrate this vector.'
                if p['parameter']=='ue_drx_configs':
                    action['note']='Joint cycle/on-duration/inactivity profile for the MEC group. The native population inactivity-only fit does not calibrate this action.'
                actions.append(action)
                if p['parameter'] == 'scheduler_mode':
                    context_changed = True
                for edge, row in _rows(graph, p['parameter']):
                    if not _compatible(row, p, profile):
                        continue
                    eff = _effect(edge, row, p, before)
                    if eff:
                        eff['parameter'] = p['parameter']
                        if drx_context_differs:
                            eff['scope_approximation']=True;eff['context_approximation']=True
                        effects.append(eff)
            if not actions:
                continue
            matched, goal_effects = [], []
            for target in needs:
                relevant = [e for e in effects if e['kpi'] == target['kpi']]
                helpful = [e for e in relevant if e['direction'] == target['needed_direction']]
                opposed = [e for e in relevant if e['direction'] not in (target['needed_direction'], 'no_change')]
                # No invented sign when no compatible action calibration exists.
                declared = _owns_target(app, target)
                if helpful or declared:
                    match = dict(kpi=target['kpi'], label=target['label'], direction=target['needed_direction'],
                                 status=('range_review' if relevant else 'contract_only') if declared else 'conditional_path',
                                 primary_target=declared, paths=helpful or relevant,
                                 opposing_paths=opposed)
                    matched.append(match); goal_effects.extend(helpful or relevant)
            if not matched:
                continue
            side_effects = []
            matched_keys = {t['kpi'] for t in matched}
            for eff in effects:
                if eff['kpi'] in matched_keys:
                    continue
                row = deepcopy(eff); row['label'] = nodes['kpi:'+row['kpi']]['label']
                guard = next((t for t in targets if t['kpi'] == row['kpi']), None)
                row['requested_range'] = guard
                row['range_risk'] = False
                if guard and finite(guard['current']):
                    row['range_risk'] = ((row['direction'] == 'increase' and guard['upper'] is not None and guard['current'] >= guard['upper']) or
                                         (row['direction'] == 'decrease' and guard['current'] <= guard['lower']))
                side_effects.append(row)
            side_effects.sort(key=lambda e: (not bool(e['requested_range']), e['evidence'] == 'no_observed_change', not nodes['kpi:'+e['kpi']].get('primary'), e['label']))
            edge_ids = {a['control_edge'] for a in actions}
            edge_ids.update(e['edge_id'] for e in goal_effects)
            for target in matched:
                edge_ids.update(e['edge_id'] for e in target['opposing_paths'])
                # Contract-only routes can still be reviewed, without a fitted sign.
                for action in actions:
                    candidate = 'influence:'+action['parameter']+':'+target['kpi']
                    if candidate in {e['id'] for e in graph['edges']}:
                        edge_ids.add(candidate)
            notes = []
            if context_changed and any(a['parameter'] == 'ue_scheduler_weights' for a in actions):
                notes.append('Scheduling mode and priority change together. Individual path gains do not establish the net bundle effect.')
            if any(e['scope_approximation'] for e in effects):
                notes.append('The action scope or scheduling context differs from the full calibrated group; gain magnitude is unavailable for this scope.')
            if app['key'] == 'x02' and any(a['after'] != 'active' for a in actions):
                notes.append('Cell occupancy must be checked. Empty-cell deep sleep and light sleep have no calibration in this study.')
            if app['key'] == 'x08':
                notes.append('Efficiency-selected weights are not the MEC/URLLC group perturbations; policy performance is not predicted here.')
            if any(not m['primary_target'] for m in matched):
                notes.append('This matches a side-effect capability; the xApp policy may pursue a different primary objective.')
            resolved = sum(e['evidence']=='resolved_average' and not e['scope_approximation'] and e['direction']==m['direction'] for m in matched for e in m['paths'])
            primary_resolved = sum(m['primary_target'] and e['evidence']=='resolved_average' and not e['scope_approximation'] and e['direction']==m['direction'] for m in matched for e in m['paths'])
            variants.append(dict(id=app['id']+':preview:'+str(index), name=profile['name'], actions=actions,
                                 matches=matched, side_effects=side_effects, notes=notes,
                                 path_edge_ids=sorted(edge_ids),
                                 _rank=(any(m['primary_target'] for m in matched), primary_resolved, resolved,
                                        -sum(e['range_risk'] for e in side_effects), -sum(len(m['opposing_paths']) for m in matched))))
        if variants:
            variants.sort(key=lambda v:v['_rank'], reverse=True)
            rank = variants[0]['_rank']
            for v in variants:
                v.pop('_rank')
            app_state = frame.get('states', {}).get(app['id'], {})
            response['candidates'].append(dict(app_id=app['id'], paper_id=app['paper_id'], label=app['label'],
                                              enabled=app_state.get('enabled'), variants=variants, _rank=rank))
    response['candidates'].sort(key=lambda c:c['_rank'], reverse=True)
    for c in response['candidates']:
        c.pop('_rank')
    return response
