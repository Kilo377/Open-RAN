"""Initial candidate pools from graph influence, independent of conflict scoring.

The graph screen contains no proposals or observed deployment results. Moving
a group to reserve is a search heuristic. An optional second stage excludes
confirmed opposed direct writes from captured decisions; it never runs models.
Original IDs and evidence remain available for audit, not automatic re-admission.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from smo.graph_combinations import enumerate_combinations


METHODS = ('reverse_graph', 'influence_coverage_v1', 'semantic_influence_coverage_v2')


@dataclass(frozen=True)
class SelectionConfig:
    coverage_threshold: float = .05
    marginal_tolerance: float = .05
    preserve_distinct_controls: bool = True
    preserve_shared_policies: bool = True
    capability_catalog: str | None = None

    def __post_init__(self):
        for key in ('coverage_threshold', 'marginal_tolerance'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(key+' must be a finite number in [0, 1]')
        for key in ('preserve_distinct_controls', 'preserve_shared_policies'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(key+' must be boolean')
        if self.capability_catalog is not None and (not isinstance(self.capability_catalog,str) or not Path(self.capability_catalog).is_absolute()):
            raise ValueError('capability_catalog must be an absolute operator-owned path')


def options(settings=None):
    """Validate a service/manifest configuration before any intent is saved."""
    settings = deepcopy(settings) if settings is not None else {}
    if not isinstance(settings, dict):
        raise ValueError('candidate_selection must be an object')
    method = settings.pop('method', 'reverse_graph')
    if method not in METHODS:
        raise ValueError('Unknown initial candidate selection method: '+str(method))
    config = SelectionConfig(**settings)
    return dict(method=method, **asdict(config))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def influence_capabilities(graph, catalog):
    """Largest absolute *already normalized* path weight, per app and KPI.

    No summation across shared paths and no coefficient-sign/action inference.
    Missing calibration is distinct from an observed zero. Confidence intervals
    are retained as diagnostics; crossing zero never deletes a path here.
    """
    edges = {e['id']:e for e in graph['edges']}
    required = catalog['required_kpis']
    population = graph.get('metric_scope') == 'ue_population.v1'
    matrix = {}
    for candidate in catalog['candidates']:
        cells = {}
        for kpi in required:
            paths = []
            for path in candidate['relation_paths']:
                if path['kpi'] != kpi:
                    continue
                edge = edges[path['influence_edge_id']]
                rows = edge.get('sensitivities') or ([edge['sensitivity']] if edge.get('sensitivity') else [])
                valid = [r for r in rows if (not population or r.get('population_control_eligible') is True)
                    and _finite(r.get('estimate'))
                    and _finite(r.get('importance', {}).get('signed_reference_importance'))]
                values = [abs(r['importance']['signed_reference_importance']) for r in valid]
                paths.append(dict(parameter=path['parameter'], control_edge_id=path['control_edge_id'],
                    influence_edge_id=path['influence_edge_id'], value=max(values) if values else None,
                    unquantified=not values or len(valid) != len(rows),
                    contrasts=[dict(contrast_id=r.get('contrast_id'),
                        magnitude=abs(r['importance']['signed_reference_importance']),
                        confidence_interval=deepcopy(r.get('confidence_interval')),
                        normalization_basis=r['importance'].get('normalization_basis')) for r in valid]))
            values = [p['value'] for p in paths if p['value'] is not None]
            cells[kpi] = dict(value=max(values) if values else 0., paths=paths,
                unquantified=any(p['unquantified'] for p in paths))
        matrix[candidate['app_id']] = cells
    scales = {k:max((cells[k]['value'] for cells in matrix.values()), default=0.) for k in required}
    for cells in matrix.values():
        for kpi, cell in cells.items():
            scale = scales[kpi]
            cell['relative_strength'] = cell['value']/scale if scale else None
            for path in cell['paths']:
                path['relative_strength'] = path['value']/scale if scale and path['value'] is not None else None
    return matrix, scales


def filter_catalog(graph, catalog, *, method='reverse_graph', config=None):
    """Split a structural catalog into primary/reserve pools without mutations."""
    if method not in METHODS:
        raise ValueError('Unknown initial candidate selection method: '+str(method))
    config = config if isinstance(config, SelectionConfig) else SelectionConfig(**(config or {}))
    result = deepcopy(catalog)
    if method == 'reverse_graph':
        # Compatibility is intentional: old callers receive the exact catalog.
        return result
    matrix, scales = influence_capabilities(graph, catalog)
    semantic = method == 'semantic_influence_coverage_v2'
    requirements = {}; profiles = None; matching = {}; support_scales = scales
    if semantic:
        from smo.xapp_capabilities import build_capabilities, required_directions
        requirements = {r['kpi']:required_directions(r) for r in catalog['ranges']}
        profiles = build_capabilities(graph, config.capability_catalog)
        matching = {a:{k:bool(set(profiles['profiles'].get(a,{}).get(k,{}).get('directions',[])) & set(requirements[k]['directions']))
            and bool(cells[k]['paths']) for k in catalog['required_kpis']} for a,cells in matrix.items()}
        support_scales = {k:max((cells[k]['value'] for a,cells in matrix.items() if matching[a][k]),default=0.)
            for k in catalog['required_kpis']}
    required = catalog['required_kpis']
    groups = catalog['groups']
    by_members = {frozenset(g['app_ids']):g for g in groups}

    def coverage(members):
        if semantic:
            return {k:max((matrix[a][k]['value']/support_scales[k] if support_scales[k] else 0.
                for a in members if matching[a][k]),default=0.) for k in required}
        return {k:max((matrix[a][k]['relative_strength'] or 0. for a in members),default=0.) for k in required}

    def unquantified(members):
        if semantic:
            return [k for k in required if any(matching[a][k] for a in members) and
                (not support_scales[k] or any(matrix[a][k]['unquantified'] for a in members if matching[a][k]))]
        return [k for k in required if not scales[k] or any(matrix[a][k]['unquantified'] for a in members)]

    def directions(members):
        return {k:dict(required=requirements[k]['directions'][:],
            supporting_apps=sorted(a for a in members if matching[a][k]),
            unconfirmed_apps=sorted(a for a in members if matrix[a][k]['paths'] and
                not profiles['profiles'].get(a,{}).get(k,{}).get('directions')))
            for k in required}

    def protected_strategies(app, remaining):
        # Different trained policies writing the same meaningful control stay
        # distinct; no arbitration or compatibility assumption is made.
        meaningful = {p['parameter'] for k in required for p in matrix[app][k]['paths']
            if p['relative_strength'] is not None and p['relative_strength'] > 0
            and p['relative_strength'] >= config.coverage_threshold}
        other = {p['parameter'] for a in remaining for k in required for p in matrix[a][k]['paths']}
        return ((config.preserve_shared_policies and bool(meaningful & other)) or
                (config.preserve_distinct_controls and bool(meaningful - other)))

    annotations = {}
    # Ascending subsets give every redundant group a retained subset witness.
    ordered = sorted(groups, key=lambda g:(len(g['app_ids']), tuple(sorted(g['app_ids']))))
    for group in ordered:
        members = frozenset(group['app_ids'])
        capability = coverage(members)
        missing_calibration = unquantified(members)
        weak = [k for k in required if capability[k] < config.coverage_threshold]
        info = dict(pool='primary', reason='meaningful_coverage_and_policy_diversity',
            coverage=capability, weak_kpis=weak, unquantified_kpis=missing_calibration,
            witness_group_id=None, removed_app_ids=[], maximum_coverage_loss=None,
            attainment_verified=False)
        missing_direction = []
        if semantic:
            info['direction_coverage'] = directions(members)
            missing_direction = [k for k,v in info['direction_coverage'].items() if not v['supporting_apps']]
            info['missing_direction_kpis'] = missing_direction
        if missing_direction:
            info.update(pool='reserve',reason='unconfirmed_capability_direction' if any(
                info['direction_coverage'][k]['unconfirmed_apps'] for k in missing_direction) else 'unsupported_capability_direction')
        elif missing_calibration:
            info['reason'] = 'unquantified_evidence_preserved'
        elif weak:
            info.update(pool='reserve', reason='weak_influence_coverage')
        else:
            for app in sorted(members):
                remaining = members - {app}
                subset = by_members.get(remaining)
                if subset is None:
                    continue
                prior = annotations[subset['id']]
                if prior['pool'] != 'primary' or prior['unquantified_kpis']:
                    continue
                loss = max((capability[k]-prior['coverage'][k] for k in required), default=0.)
                if loss > config.marginal_tolerance or protected_strategies(app, remaining):
                    continue
                info.update(pool='reserve', reason='low_marginal_contribution',
                    witness_group_id=subset['id'], removed_app_ids=[app], maximum_coverage_loss=loss)
                break
            # A reserve immediate subset can still have a primary descendant.
            if info['pool'] == 'primary':
                for subset in ordered:
                    sm = frozenset(subset['app_ids'])
                    if not sm < members:
                        continue
                    prior = annotations.get(subset['id'])
                    if not prior or prior['pool'] != 'primary' or prior['unquantified_kpis']:
                        continue
                    removed = members-sm
                    loss = max((capability[k]-prior['coverage'][k] for k in required), default=0.)
                    if loss <= config.marginal_tolerance and not any(protected_strategies(a,sm) for a in removed):
                        info.update(pool='reserve', reason='low_marginal_contribution',
                            witness_group_id=subset['id'], removed_app_ids=sorted(removed), maximum_coverage_loss=loss)
                        break
        annotations[group['id']] = info
    result['groups'] = []
    result['reserve_groups'] = []
    for group in groups:
        row = deepcopy(group)
        row['initial_selection'] = annotations[group['id']]
        result['groups' if row['initial_selection']['pool'] == 'primary' else 'reserve_groups'].append(row)
    # If the proxy removes every group, restore exhaustive evaluation. An
    # influence screen must not make an otherwise reviewable intent disappear.
    # An unsupported direction is not allowed back into primary through fallback.
    fallback = not semantic and bool(groups) and not result['groups']
    if fallback:
        result['groups'] = result['reserve_groups']
        result['reserve_groups'] = []
        for row in result['groups']:
            row['initial_selection'].update(pool='primary', reason='empty_primary_fallback')
    result['group_count'] = result['eligible_group_count'] = len(result['groups'])
    result['structural_group_count'] = len(groups)
    result['reserve_group_count'] = len(result['reserve_groups'])
    result['selection_algorithm'] = method
    result['initial_selection'] = dict(schema='oran.initial-candidate-selection.v1', method=method,
        config=asdict(config), input_group_count=len(groups), primary_group_count=len(result['groups']),
        reserve_group_count=len(result['reserve_groups']), exhaustive_fallback=fallback,
        pruned_fraction=len(result['reserve_groups'])/len(groups) if groups else 0.,
        capability_aggregation='maximum_absolute_normalized_path_weight',
        per_kpi_scales=support_scales, app_capabilities=matrix,
        graph_sha256=hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        action_values_used=False, observed_outcomes_used=False, conflict_scores_used=False,
        intent_attainment_predicted=False, final_strategy_selected=False,
        note='Primary/reserve is a graph influence heuristic, not proof of feasibility or impossibility. '
             'Bounds, action direction, magnitude and joint effects are not predicted. All reserve groups are recoverable.')
    if semantic:
        result['initial_selection'].update(direction_requirements=requirements,
            xapp_direction_capabilities=profiles, direction_coverage_basis='released_objectives_and_audited_policy_capabilities',
            capability_direction_used=True, current_actions_used=False, model_inference_invoked=False,
            gain_signs_used_for_capability=False,
            historical_capability_evidence_used=bool(profiles['applied_secondary']))
        result['direction_missing_kpis']=[k for k in required if not any(matching[a][k] for a in matrix)]
    return result


def select_combinations(graph, frame, plan_or_ranges, *, show_dashed=False, settings=None, decision_context=None):
    """Replaceable initial-selection entry point for service and experiments."""
    selected = options(settings)
    method = selected.pop('method')
    catalog = enumerate_combinations(graph, frame, plan_or_ranges, show_dashed=show_dashed)
    result=filter_catalog(graph, catalog, method=method, config=selected)
    if decision_context is not None:
        from smo.direct_action_filter import filter_catalog as filter_direct_actions
        result=filter_direct_actions(result,graph,decision_context)
    return result
