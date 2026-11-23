"""Read-only xApp combination review from graph structure alone.

No action space, frozen model, learner, simulator or executor is imported.
An edge can identify shared control or coupling, never an incompatible pair of
actual values or the direction of a joint KPI response without live proposals.
"""
from copy import deepcopy
from itertools import combinations
import math


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _empty(row):
    low, high = row.get('lower'), row.get('upper')
    if any(value is not None and not _finite(value) for value in (low, high)):
        raise ValueError('Graph review requires resolved numerical KPI bounds.')
    return bool(row.get('empty')) or low is not None and high is not None and (
        low > high or low == high and not (row.get('lower_inclusive') and row.get('upper_inclusive')))


def _resolved(row, population):
    # Population regressions must explicitly be eligible common interventions.
    # A significant vector member is not evidence of a common population gain.
    if population and row.get('population_control_eligible') is not True:
        return False
    if row.get('zero_in_both') or row.get('fit_diagnostics', {}).get('assessment') == 'constant_response_observed':
        return False
    ci = row.get('confidence_interval')
    return isinstance(ci, (list, tuple)) and len(ci) == 2 and all(map(_finite, ci)) and ci[0] <= ci[1] and (ci[0] > 0 or ci[1] < 0)


def influence_evidence(edge, graph):
    """Overview evidence across contexts; it does not select any contrast."""
    rows = edge.get('sensitivities') or ([edge['sensitivity']] if edge.get('sensitivity') else [])
    good = [row for row in rows if _resolved(row, graph.get('metric_scope') == 'ue_population.v1')]
    measured = [row for row in rows if _finite(row.get('estimate')) and
                (graph.get('metric_scope') != 'ue_population.v1' or row.get('population_control_eligible') is True)]
    declared = bool(edge.get('objectives'))
    # A confidence interval describes a gain, not the existence of the path.
    # Finite measured weights and declared objective paths remain in topology.
    return dict(dashed=not bool(measured or declared), conditional_evidence=bool(rows),
        measured_evidence=bool(measured),declared_objective_path=declared,
        gain_direction_resolved=bool(good),observed_zero=_known_zero(edge),
        resolved_contrasts=[row.get('contrast_id') for row in good],
        tested_contrasts=[row.get('contrast_id') for row in rows],
        action_selected=False)


def _known_zero(edge):
    rows = edge.get('sensitivities') or ([edge['sensitivity']] if edge.get('sensitivity') else [])
    return bool(rows) and all(row.get('zero_in_both') or
        row.get('estimate') == 0 and row.get('confidence_interval') == [0, 0] or
        row.get('fit_diagnostics', {}).get('assessment') == 'constant_response_observed'
        for row in rows)


def _members(edge, app):
    """Only explicit write scopes can establish disjointness.

    Scheduler controllers can reset priorities outside target_ues. A generic
    target label therefore cannot prove their write scopes are disjoint.
    """
    explicit = edge.get('write_scope')
    if isinstance(explicit, dict) and isinstance(explicit.get('members'), list):
        return set(explicit['members'])
    detail = edge.get('declared_values')
    if isinstance(detail, dict) and isinstance(detail.get('target_ues'), list):
        return set(detail['target_ues'])
    if edge['target'] in ('rcp:ue_offload_fraction', 'rcp:ue_drx_configs') and isinstance(app.get('target_ues'), list):
        return set(app['target_ues'])
    return None


def _overlap(left, right, apps):
    a, b = _members(left, apps[left['source']]), _members(right, apps[right['source']])
    if a is not None and b is not None:
        shared = sorted(a & b)
        return ('known_overlap', shared) if shared else ('disjoint', [])
    if a == set() or b == set():
        return 'disjoint', []
    return 'possible_overlap', []


def _summarize(conflicts):
    summaries = {}
    for conflict in conflicts:
        identity = (conflict['type'], tuple(conflict['apps']))
        if identity not in summaries:
            summaries[identity] = dict(type=conflict['type'], apps=conflict['apps'],
                status=conflict['status'], parameters=[], kpis=[], edge_ids=[], paths=[],
                dashed=False, conditional_evidence=False, scope_status=conflict.get('scope_status'),
                overlap_members=[], action_values_known=False, actual_conflict_verified=False)
        summary = summaries[identity]
        for field in ('parameters', 'kpis', 'edge_ids', 'paths', 'overlap_members'):
            for value in conflict.get(field, []):
                if value not in summary[field]:
                    summary[field].append(value)
        summary['dashed'] |= conflict['dashed']
        summary['conditional_evidence'] |= conflict['conditional_evidence']
        if conflict.get('scope_status') == 'possible_overlap':
            summary['scope_status'] = 'possible_overlap'
    return list(summaries.values())


def enumerate_combinations(graph, frame, plan_or_ranges, *, show_dashed=False):
    """Enumerate all structurally relevant groups covering every requested KPI.

    ``plan_or_ranges`` is a resolved plan dict or a list of merged range rows.
    Empty intersections stop enumeration. Numerical range attainment is never
    inferred from graph reachability, coefficient signs, or controller labels.
    """
    plan = plan_or_ranges if isinstance(plan_or_ranges, dict) else dict(ranges=plan_or_ranges)
    ranges = plan.get('ranges', [])
    if not isinstance(ranges, list):
        raise ValueError('Expected a list of resolved KPI ranges.')
    required = list(dict.fromkeys(row['kpi'] for row in ranges))
    result = dict(schema='oran.graph-combinations.v1', snapshot={k:frame.get(k) for k in ('slot','time_ms')},
        analysis_basis='knowledge_graph', runtime_proposals_used=False, action_values_known=False,
        automatic_selection=False, selected_by_default=False, deployment_enabled=False,
        selection_algorithm=None, ranking_performed=False,
        show_dashed=show_dashed, required_kpis=required, ranges=deepcopy(ranges), candidates=[],
        excluded_apps=[], missing_kpis=required[:], groups=[], group_count=0, eligible_group_count=0,
        attainment_verified=False,
        note='Graph structure identifies potential shared-control and cross-parameter coupling risks. '
             'No action, model output, contrast, conflict winner or deployment is selected. '
             'Compatible values and joint KPI attainment remain unverified.')
    if plan.get('snapshot', {}).get('slot', frame.get('slot')) != frame.get('slot'):
        result['unavailable'] = 'The graph review snapshot differs from the intent observation.'
        return result
    if plan.get('questions'):
        result['unavailable'] = 'Clarification is required before reviewing this intent.'
        return result
    if plan.get('planning_blocked') or any(issue.get('blocking') for issue in plan.get('constraint_issues', [])):
        result['planning_blocked'] = True
        result['constraint_issues'] = deepcopy(plan.get('constraint_issues', []))
        result['unavailable'] = 'Resolve the marked active constraints before reviewing combinations.'
        if any(_empty(row) for row in ranges):
            result['empty_intersection'] = True
        return result
    if any(_empty(row) for row in ranges):
        result['empty_intersection'] = True
        return result
    if not required:
        return result

    apps = {node['id']:node for node in graph['nodes'] if node['kind']=='xapp'}
    controls = {app_id:[] for app_id in apps}
    for edge in graph['edges']:
        if edge['kind']=='controls' and edge.get('deployment')!='native_locked' and edge['source'] in apps:
            controls[edge['source']].append(edge)
    influences = {(edge['source'],edge['target']):edge for edge in graph['edges'] if edge['kind']=='influences'}
    evidence = {edge['id']:influence_evidence(edge,graph) for edge in influences.values()}
    for app_id, app in apps.items():
        owns = [key for key in required if key in app.get('target_kpis',{})]
        if not controls[app_id]:
            if owns:
                result['excluded_apps'].append(dict(app_id=app_id, reason='no_unlocked_control_edges'))
            continue
        paths = []
        for control in controls[app_id]:
            for key in required:
                edge = influences.get((control['target'], 'kpi:'+key))
                if not edge:
                    continue
                detail = evidence[edge['id']]
                # A declared objective still needs a real control/influence
                # path. Uncalibrated contract paths are retained as declarations,
                # not as measured numerical response evidence.
                declared_path = key in owns
                if _known_zero(edge) and not declared_path:
                    continue
                if not show_dashed and not detail['measured_evidence'] and not declared_path:
                    continue
                if detail['dashed'] and not show_dashed and not declared_path:
                    continue
                paths.append(dict(kpi=key, parameter=control['target'][4:],
                    control_edge_id=control['id'], influence_edge_id=edge['id'],
                    declared_target=key in owns, **detail))
        related = [key for key in required if any(p['kpi']==key for p in paths)]
        if not related:
            continue
        state = frame.get('states',{}).get(app_id,{})
        result['candidates'].append(dict(app_id=app_id,paper_id=app.get('paper_id'),label=app.get('label'),
            target_kpis=deepcopy(app.get('target_kpis',{})),owns_kpis=owns,related_kpis=related,
            relation_paths=paths,enabled=state.get('enabled'),
            controls=[edge['id'] for edge in controls[app_id]]))
    contributors = {candidate['app_id']:candidate['related_kpis'] for candidate in result['candidates']}
    owners = {candidate['app_id']:candidate['owns_kpis'] for candidate in result['candidates']}
    result['coverage_basis'] = 'reverse_graph_paths'
    result['kpi_backtrace'] = {key:[dict(app_id=c['app_id'], label=c['label'],
        paths=deepcopy([p for p in c['relation_paths'] if p['kpi']==key]))
        for c in result['candidates'] if key in c['related_kpis']] for key in required}
    result['missing_kpis'] = [key for key in required if not any(key in keys for keys in contributors.values())]
    if result['missing_kpis']:
        return result

    def path(app_id, control, influence, role):
        return dict(role=role, app_id=app_id, parameter=control['target'][4:],
            kpi=influence['target'][4:],control_edge_id=control['id'],influence_edge_id=influence['id'],
            **evidence[influence['id']])

    def pair_risks(a, b):
        risks = []
        for left in controls[a]:
            for right in controls[b]:
                if left['target'] != right['target']:
                    continue
                scope, shared = _overlap(left,right,apps)
                if scope == 'disjoint':
                    continue
                risks.append(dict(type='direct',apps=[a,b],parameters=[left['target'][4:]],kpis=[],
                    edge_ids=[left['id'],right['id']],paths=[],status='potential_unknown_values',
                    scope_status=scope,overlap_members=shared,dashed=False,conditional_evidence=False,
                    action_values_known=False,actual_conflict_verified=False))
        source, target = a, b
        for key in required:
            for source_control in controls[source]:
                source_edge = influences.get((source_control['target'],'kpi:'+key))
                if not source_edge or _known_zero(source_edge):
                    continue
                for target_control in controls[target]:
                    if target_control['target'] == source_control['target']:
                        continue
                    target_edge = influences.get((target_control['target'],'kpi:'+key))
                    if not target_edge or _known_zero(target_edge):
                        continue
                    paths=[path(source,source_control,source_edge,'source'),path(target,target_control,target_edge,'target')]
                    risks.append(dict(type='indirect',apps=[source,target],
                        parameters=[source_control['target'][4:],target_control['target'][4:]],kpis=[key],
                        edge_ids=[source_control['id'],source_edge['id'],target_control['id'],target_edge['id']],
                        paths=paths,status='structural_coupling',dashed=any(p['dashed'] for p in paths),
                        conditional_evidence=any(p['conditional_evidence'] for p in paths),
                        action_values_known=False,actual_conflict_verified=False))
        return risks

    pair_cache = {pair:pair_risks(*pair) for pair in combinations(contributors,2)}
    for size in range(1,len(contributors)+1):
        for group in combinations(contributors,size):
            coverage = {key:[app for app in group if key in contributors[app]] for key in required}
            if not all(coverage.values()):
                continue
            raw = [deepcopy(risk) for pair in combinations(group,2) for risk in pair_cache[pair]]
            for index, risk in enumerate(raw):
                risk['id'] = 'graph-conflict:'+str(index)
            visible = [risk for risk in raw if show_dashed or risk['type']=='direct' or not risk['dashed']]
            result['groups'].append(dict(id='graph-group:'+'|'.join(group),app_ids=list(group),
                ownership={app:owners[app][:] for app in group},coverage=coverage,
                coverage_paths={key:[dict(app_id=row['app_id'],paths=deepcopy(row['paths']))
                    for row in result['kpi_backtrace'][key] if row['app_id'] in group] for key in required},
                intent_coverage=dict(required_kpis=required[:],covered_kpis=required[:],missing_kpis=[],
                    required_count=len(required),covered_count=len(coverage),
                    contributors=deepcopy(coverage),complete=True,eligible=True,
                    basis='reverse_graph_paths',attainment_verified=False),
                raw_conflicts=raw,conflicts=visible,conflict_summaries=_summarize(visible),
                raw_conflict_count=len(raw),visible_conflict_count=len(visible),
                analysis_basis='knowledge_graph',actual_conflict_verified=False))
    result['group_count'] = result['eligible_group_count'] = len(result['groups'])
    return result
