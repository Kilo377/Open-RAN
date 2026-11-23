"""Zero-anchored observed intent improvement, independent of conflict risk.

Pure derived views: archived intervals, measurements and pass/fail statuses
are never rewritten. One batch baseline supplies common KPI scales.
"""
from copy import deepcopy
import math

from smo.runtime import number
from smo.intent_quality import VALID

METHOD = 'baseline_relative_intent_v2'


def normalized(value,scale):
    if not number(value) or not number(scale) or scale<=0:return None
    result=value/scale*100
    return result if number(result)else None


def reference_scale(kpi, reference_values, reference_scales=None, basis='before_window'):
    explicit = reference_scales or {}
    value = reference_values.get(kpi)
    if kpi in explicit:
        scale = explicit[kpi]
        return dict(scale=scale if number(scale) and scale > 0 else None,
            scale_basis='explicit_reference_scale' if number(scale) and scale > 0 else 'invalid_explicit_scale',
            reference_value=value)
    if number(value) and value != 0:
        return dict(scale=abs(value), scale_basis=basis+'_magnitude', reference_value=value)
    return dict(scale=None, scale_basis='explicit_scale_required', reference_value=value)


def boundaries(row):
    domain = row.get('domain', [0, None])
    low, high = row.get('lower'), row.get('upper')
    point = number(low) and number(high) and low == high
    result = []
    if number(low) and (point or low != domain[0] or not row.get('lower_inclusive', True)):
        result.append(('lower', low))
    if number(high) and (point or high != domain[1] or not row.get('upper_inclusive', True)):
        result.append(('upper', high))
    return result


def margin(row, reading):
    if not reading or reading.get('status') not in VALID or not number(reading.get('value')):
        return dict(value=None, reason='measurement_unavailable')
    if row.get('empty') or row['kpi'] == 'ue_energy_j':
        return dict(value=None, reason='invalid_intent')
    m = reading.get('measurement') or {}
    if m.get('reason'):
        return dict(value=None, reason=m['reason'])
    low = m.get('minimum') if m.get('constraint_uses_extrema') else reading['value']
    high = m.get('maximum') if m.get('constraint_uses_extrema') else reading['value']
    if not number(low) or not number(high) or low > high:
        return dict(value=None, reason='measurement_unavailable')
    domain = row.get('domain', [0, None])
    if number(domain[0]) and low < domain[0] or number(domain[1]) and high > domain[1]:
        return dict(value=None, reason='measurement_outside_physical_domain')
    limits = boundaries(row)
    # A range equal to the entire physical domain is a neutral requirement,
    # not a missing measurement or an implicit preference for one endpoint.
    value=min((low-bound if side == 'lower' else bound-high for side,bound in limits),default=0.)
    if not number(value):return dict(value=None,reason='nonfinite_interval_margin')
    return dict(value=value,
        reason=None, strict_boundary_violation=bool(reading.get('strict_boundary_violation')),
        satisfied=reading['status'] == 'satisfied')


def evaluate(comparison, *, reference_values=None, reference_scales=None, ranges=None,
             reference_basis='before_window'):
    rows = comparison.get('rows', [])
    targets = ranges if ranges is not None else rows
    keys = [r['kpi'] for r in targets]
    if len(set(keys)) != len(keys):
        raise ValueError('Result evaluation needs unique requested KPIs')
    indexed = {r['kpi']: r for r in rows}
    reference = reference_values if reference_values is not None else {
        r['kpi']: (r.get('before') or {}).get('value') for r in rows}
    metrics = []
    for target in targets:
        k = target['kpi']; row = indexed.get(k, {})
        # A failure ratio already is dimensionless. Its changes are percentage
        # points, never relative percentages divided by an observed zero.
        if k == 'urllc_failure_fraction':
            unit=target.get('unit','ratio')
            norm=dict(scale=1. if unit in ('ratio','fraction') else 100. if unit=='%' else None,
                scale_basis='ratio_percentage_points' if unit in ('ratio','fraction','%') else 'incompatible_ratio_unit',
                reference_value=reference.get(k))
        else:
            norm = reference_scale(k, reference, reference_scales, reference_basis)
        scale = norm['scale']
        limits=boundaries(target)
        result = dict(kpi=k, unit=target.get('unit'), **norm, active_boundaries=limits,
            score_unit='percentage_points' if k=='urllc_failure_fraction' else 'relative_percent',
            weight=1.,weight_basis='equal_kpi_value',
            mode='directional_change' if len(limits)==1 else 'interval_distance_reduction',
            improvement_score=None,change=None,regression=None,target_progress_percent=None)
        for phase in ('before', 'after'):
            reading = row.get(phase)
            m = margin(target, reading)
            distance=max(0.,-m['value']) if m['value'] is not None else None
            result[phase] = dict(**m, distance=distance,
                normalized_violation=normalized(distance,scale),
                margin_score=normalized(m['value'],scale),
                score=0. if phase=='before' and m['value'] is not None and scale is not None else None,
                status=(reading or {}).get('status', 'unavailable'))
            if m['value'] is not None and scale is None:
                result[phase]['reason'] = norm['scale_basis'] if k=='urllc_failure_fraction' else 'explicit_scale_required'
            elif m['value'] is not None and result[phase]['normalized_violation'] is None:
                result[phase].update(score=None,reason='nonfinite_normalized_violation')
        a,b=result['before'],result['after']
        complete=all(p['value'] is not None and p['normalized_violation'] is not None for p in (a,b)) and scale is not None
        if complete:
            # Half-lines have a natural direction and retain useful benefit
            # after crossing their boundary. Genuine bands/points reward only
            # reduced distance, never an unrequested preference for the centre.
            delta=b['value']-a['value'] if len(limits)==1 else a['distance']-b['distance']
            strict_regression=bool(a.get('satisfied') and b.get('strict_boundary_violation'))
            gain=normalized(delta,scale)
            if gain is None:
                b['reason']='nonfinite_normalized_improvement'
                metrics.append(result)
                continue
            result.update(improvement_score=gain,change=gain,
                strict_boundary_regression=strict_regression,regression=gain<0 or strict_regression)
            b['score']=gain
            if a['distance']>0:
                result['target_progress_percent']=normalized(a['distance']-b['distance'],a['distance'])
        metrics.append(result)

    def aggregate(phase):
        valid = [m for m in metrics if m[phase]['score'] is not None]
        complete = bool(metrics) and len(valid) == len(metrics)
        gains=[m[phase]['score']for m in valid]
        regressed=phase=='after' and complete and any(m['regression'] for m in metrics)
        score=(min(gains) if regressed else math.fsum(g/len(gains)for g in gains)) if complete else None
        weakest=min(gains)if complete else None
        # Never infer attainment from the sign or round away strict endpoints.
        return dict(score=score, complete=complete, valid=len(valid), required=len(metrics),
            all_satisfied=all(m[phase].get('satisfied') is True for m in metrics) if complete else None,
            bottleneck_kpis=[m['kpi'] for m in metrics if m[phase]['score'] == weakest] if complete else [],
            reason=None if complete else 'incomplete_requested_kpis')

    before, after = aggregate('before'), aggregate('after')
    change=after['score']
    complete=after['complete']
    regressed=[m['kpi']for m in metrics if m['regression']is True]
    no_regression=not bool(regressed)if complete else None
    mean_improvement=math.fsum(m['improvement_score']/len(metrics)for m in metrics)if complete else None
    def violation(phase):
        values=[m[phase]['normalized_violation']for m in metrics]
        valid=bool(values) and all(v is not None for v in values)
        return dict(mean=math.fsum(v/len(values)for v in values)if valid else None,
            max=max(values)if valid else None,complete=valid)
    violation_before,violation_after=violation('before'),violation('after')
    state='insufficient_data' if not complete else 'regressed'if regressed else 'improved'if any(m['improvement_score']>0 for m in metrics)else'unchanged'
    return dict(schema='oran.observed-intent-result.v2', method=METHOD,
        reference_basis=reference_basis, reference_values=deepcopy(reference), metrics=metrics,
        before=before, after=after, change=change, higher_is_better=True,
        no_regression=no_regression,admissible=no_regression,regression_kpis=regressed,
        mean_improvement=mean_improvement,state=state,
        violation=dict(before=violation_before,after=violation_after,
            change=violation_before['mean']-violation_after['mean']if violation_before['complete']and violation_after['complete']else None),
        aggregation='equal_mean_if_no_regression_else_worst_regression',
        hard_veto_boundary_is_discontinuous=True,unchanged_is_zero=True,
        physical_domain_bounds_are_objectives=False, zero_baseline_fallback_invented=False,
        risk_inputs_used=False, graph_gains_used=False, ranking_performed=False,
        interpretation='zero_anchored_improvement_with_noncompensable_regression_and_separate_interval_violation',
        statistical_significance='not_estimated')


def enrich_batch(batch):
    """Use the retained BEFORE reference, never after data or member order."""
    from smo.intent_quality import enrich_comparison
    result = deepcopy(batch)
    reference = result.get('baseline_reference', {})
    for member in result.get('members', []):
        if member.get('comparison'):
            member['comparison'] = enrich_comparison(member['comparison'],
                evaluation_reference=reference,
                evaluation_scales=result.get('result_evaluation_scales'),
                evaluation_ranges=result.get('ranges'), reference_basis='batch_baseline_reference')
    result['result_evaluation_method'] = METHOD
    result['result_evaluation_reference'] = deepcopy(reference)
    return result
