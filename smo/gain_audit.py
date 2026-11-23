"""Read-only numerical gain coverage, distinct from topology and attainment.

A finite uncertain or observed-zero coefficient is a measured gain. A declared
objective alone is not. This audit never invents edges, transfers estimates
between scenarios, changes screening policy, or checks deployed outcomes.
"""
from copy import deepcopy
import math

from smo.graph_audit import audit_graph
from smo.graph_combinations import influence_evidence
from smo.influence_units import projection


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _rows(edge):
    return edge.get('sensitivities') or ([edge['sensitivity']] if edge.get('sensitivity') else [])


def _row_audit(row, edge, kpi_unit, *, population):
    if not isinstance(row, dict):
        return dict(contrast_id=None, valid=False, reasons=['invalid_estimate_record'], evidence='unavailable')
    reasons=[]
    if not _finite(row.get('estimate')):reasons.append('non_finite_scalar_gain')
    parameter=edge.get('source') if isinstance(edge.get('source'),str) else ''
    kpi=edge.get('target') if isinstance(edge.get('target'),str) else ''
    if row.get('parameter')!=parameter[4:]:reasons.append('parameter_mismatch')
    if row.get('kpi') is not None and row['kpi']!=kpi[4:]:reasons.append('kpi_mismatch')
    if population and row.get('population_control_eligible') is not True:
        reasons.append('population_evidence_unavailable')
    expected=projection(row,kpi_unit)
    supplied=row.get('projection_units')
    if not expected['valid']:
        reasons.append(expected.get('reason','invalid_gain_unit'))
    elif not isinstance(supplied,dict) or not supplied.get('valid'):
        reasons.append('missing_or_invalid_unit_contract')
    elif any(supplied.get(k)!=expected.get(k) for k in ('source','fields','kpi_unit','parameter_unit')):
        reasons.append('stale_unit_contract')
    ci=row.get('confidence_interval')
    valid_ci=isinstance(ci,(list,tuple)) and len(ci)==2 and all(map(_finite,ci)) and ci[0]<=ci[1]
    if valid_ci and _finite(row.get('estimate')) and not ci[0]<=row['estimate']<=ci[1]:
        reasons.append('estimate_outside_confidence_interval')
    zero=_finite(row.get('estimate')) and row['estimate']==0 and valid_ci and ci[0]==ci[1]==0
    if row.get('zero_in_both') and _finite(row.get('estimate')) and row['estimate']!=0:
        reasons.append('inconsistent_observed_zero')
    evidence='observed_zero' if zero else 'uncertainty_unavailable' if not valid_ci else (
        'direction_resolved' if ci[0]>0 or ci[1]<0 else 'uncertain_direction')
    return dict(contrast_id=row.get('contrast_id'),valid=not reasons,reasons=reasons,evidence=evidence,
        estimate=row.get('estimate') if _finite(row.get('estimate')) else None,
        unit=row.get('unit'),confidence_interval=deepcopy(ci),confidence_level=row.get('confidence_level'),
        parameter_component=row.get('parameter_component'),scalar_component=row.get('scalar_component'),
        members=deepcopy(row.get('members')),fit_domain=deepcopy(row.get('fit_domain') or row.get('sampled_domain')),
        from_value=deepcopy(row.get('from')),to_value=deepcopy(row.get('to')))


def audit_gains(graph, *, all_kpis=False):
    """Check every P–K edge and numeric target path without changing the KG.

    The default visible scope matches the UI's primary KPI view. All-edge
    diagnostics also include auxiliary KPIs. Control edges describe writable
    parameters and therefore do not carry statistical KPI coefficients.
    """
    nodes={n['id']:n for n in graph.get('nodes',[])}
    population=graph.get('metric_scope')=='ue_population.v1'
    links=[];numeric_edges=[];topology_edges=[]
    for edge in graph.get('edges',[]):
        source,target=nodes.get(edge.get('source')),nodes.get(edge.get('target'))
        if edge.get('kind')=='controls':
            if source and target:topology_edges.append(edge)
            continue
        if edge.get('kind')!='influences':continue
        endpoint_valid=bool(source and source.get('kind')=='rcp' and target and target.get('kind')=='kpi')
        rows=[_row_audit(r,edge,target.get('unit') if target else None,population=population) for r in _rows(edge)]
        valid=endpoint_valid and bool(rows) and all(r['valid'] for r in rows)
        scope=bool(target and (all_kpis or target.get('primary')))
        # An uncalibrated declared path can still be visible in the old graph;
        # include it in this audit rather than counting a contract as a gain.
        malformed=any(not isinstance(r,dict) for r in _rows(edge))
        visible=scope and (not endpoint_valid or malformed or not influence_evidence(edge,graph)['dashed'])
        reasons=([] if endpoint_valid else ['invalid_parameter_kpi_endpoints'])+([] if rows else ['gain_unavailable'])
        reasons+=sorted({reason for row in rows for reason in row['reasons']})
        links.append(dict(edge_id=edge.get('id'),source=edge.get('source'),target=edge.get('target'),
            visible=visible,valid=valid,reasons=reasons,rows=rows))
        if valid:numeric_edges.append(edge)
    numeric_graph={**graph,'edges':topology_edges+numeric_edges}
    topology=audit_graph(numeric_graph,all_kpis=all_kpis)
    visible_links=[e for e in links if e['visible']]
    primary={n['id'] for n in nodes.values() if n.get('kind')=='kpi' and n.get('primary')}
    no_primary=[n['id'] for n in nodes.values() if n.get('kind')=='rcp' and
        not any(e['source']==n['id'] and e['target'] in primary for e in numeric_edges)]
    rows=[r for e in links for r in e['rows']]
    return dict(schema='oran.gain-coverage.v1',all_kpis=all_kpis,
        influence_links=len(links),visible_influence_links=len(visible_links),
        link_count=len(links),valid_link_count=sum(e['valid'] for e in links),
        valid_gain_links=sum(e['valid'] for e in links),invalid_gain_links=sum(not e['valid'] for e in links),
        invalid_edge_ids=[e['edge_id'] for e in links if not e['valid']],
        visible_invalid_edge_ids=[e['edge_id'] for e in visible_links if not e['valid']],
        estimate_count=len(rows),valid_estimate_count=sum(r['valid'] for r in rows),
        uncertain_estimate_count=sum(r['valid'] and r['evidence']=='uncertain_direction' for r in rows),
        observed_zero_estimate_count=sum(r['valid'] and r['evidence']=='observed_zero' for r in rows),
        unknown_uncertainty_count=sum(r['valid'] and r['evidence']=='uncertainty_unavailable' for r in rows),
        all_link_gains_complete=all(e['valid'] for e in links),
        visible_link_gains_complete=all(e['valid'] for e in visible_links),
        missing_target_paths=topology['missing_target_paths'],isolated_nodes=topology['isolated_nodes'],
        target_paths=topology['target_paths'],topology_complete=topology['topology_complete'],
        rcps_without_primary_kpi_gains=no_primary,edges=links,
        control_edges_are_write_contracts=True,uncertain_gains_are_not_missing=True,
        observed_zero_is_not_nonzero_influence=True,action_applicability_checked=False,
        attainment_verified=False,deployment_results_used=False,graph_modified=False)
