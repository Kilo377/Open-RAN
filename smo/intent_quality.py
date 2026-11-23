"""Intent-bound before/after measurements; no weighted KPI utility."""
from copy import deepcopy
from smo.runtime import number

VALID = frozenset(('satisfied','below_range','above_range','outside_range'))


def quantify(row, before, after):
    """Use actual matched-window readings, never proposals or predicted gains.

    Bounds stay frozen at intent parsing. Pure directional requests have no
    finite amount to complete; their benefit is a signed change from the actual
    deployment baseline. Numeric targets additionally expose boundary progress.
    """
    b, y = before.get('value'), (after or {}).get('value')
    valid = before.get('status') in VALID and (after or {}).get('status') in VALID
    low, high = row.get('lower'), row.get('upper')
    domain = row.get('domain', [0., None])
    low_active = number(low) and (low != domain[0] or not row.get('lower_inclusive', True))
    high_active = number(high) and (high != domain[1] or not row.get('upper_inclusive', True))
    direction = 1 if low_active and not high_active else -1 if high_active and not low_active else None
    # Frozen symbolic requests distinguish "reduce" from "reduce by 2%".
    anchors = row.get('anchors') or [row]
    requested = [r for a in anchors for r in a.get('requested', [])]
    side = 'lower' if direction == 1 else 'upper'
    directional = direction is not None and bool(requested) and all(
        r.get(side) == 'current' and not r.get(side+'_inclusive', True)
        and r.get('upper' if direction == 1 else 'lower') in
        ((None, 'maximum') if direction == 1 else (None, 'minimum', 0, 0.))
        for r in requested)
    # Older archives retain bounds/current but not symbolic request metadata.
    if direction is not None and not requested:
        directional = all(number(a.get('current')) and a.get(side) == a['current']
            and not a.get(side+'_inclusive', True) for a in anchors)
    mode = 'directional_change' if directional else 'target_progress'
    result = dict(mode=mode, complete=valid, direction=direction, delta=None,
        relative_change_percent=None, favorable_delta=None, favorable_change_percent=None,
        target_boundary=None, required_change=None, target_progress_percent=None,
        before_margin=None, after_margin=None, after_normalized_margin=None,
        satisfied=(after or {}).get('status') == 'satisfied' if (after or {}).get('status') in VALID else None,
        evidence='observed_window_change', statistical_significance='not_estimated')
    if not valid:return result
    delta = y-b
    result.update(delta=delta, relative_change_percent=delta/abs(b)*100 if b else None)
    def margin(reading):
        m = reading.get('measurement', {})
        lo = m.get('minimum', reading['value']) if m.get('constraint_uses_extrema') else reading['value']
        hi = m.get('maximum', reading['value']) if m.get('constraint_uses_extrema') else reading['value']
        return min([lo-low] if high is None else [lo-low, high-hi])
    result.update(before_margin=margin(before), after_margin=margin(after))
    result['after_normalized_margin'] = result['after_margin']/abs(b) if b else None
    if direction is not None:
        result.update(favorable_delta=direction*delta,
            favorable_change_percent=direction*delta/abs(b)*100 if b else None)
    if directional:return result
    # Progress is meaningful only when the baseline must move to a boundary.
    def extreme(reading, key):
        m=reading.get('measurement', {})
        return m.get(key,reading['value']) if m.get('constraint_uses_extrema') else reading['value']
    bmin,bmax=extreme(before,'minimum'),extreme(before,'maximum')
    target = low if bmin < low else high if high is not None and bmax > high else None
    if bmin < low and high is not None and bmax > high:target=None
    if target is not None:
        key='minimum' if bmin < low else 'maximum'
        start,end=extreme(before,key),extreme(after,key)
        result.update(target_boundary=target, required_change=target-start,
            target_progress_percent=(end-start)/(target-start)*100,
            progress_statistic='window_'+key if before.get('measurement',{}).get('constraint_uses_extrema') or after.get('measurement',{}).get('constraint_uses_extrema') else 'window_value')
    else:
        result['mode'] = 'maintain_range' if before['status'] == 'satisfied' else 'strict_boundary'
    return result


def enrich_comparison(comparison, *, evaluation_reference=None, evaluation_scales=None,
                      evaluation_ranges=None, reference_basis='before_window'):
    """Add derived quantities to an archived result without changing its data."""
    result = deepcopy(comparison)
    for row in result.get('rows', []):
        row['quantification'] = quantify(row, row['before'], row.get('after'))
    result['quantification_version'] = 1
    from smo.result_evaluation import evaluate
    result['result_evaluation'] = evaluate(result, reference_values=evaluation_reference,
        reference_scales=evaluation_scales, ranges=evaluation_ranges, reference_basis=reference_basis)
    return result


def normalization(row, before):
    value=before.get('value')
    if number(value) and abs(value)>0:
        return dict(scale=abs(value),basis='before_window_magnitude')
    bounds=[abs(row[k]) for k in ('lower','upper') if number(row.get(k)) and abs(row[k])>0]
    if bounds:return dict(scale=max(bounds),basis='nonzero_requested_bound')
    resolution=before.get('resolution')
    if number(resolution) and resolution>0:return dict(scale=resolution,basis='measurement_resolution')
    return dict(scale=None,basis='normalization_unavailable')


def assess(row, frame, scale=None):
    metric=frame.get('states',{}).get('kpi:'+row['kpi'],{}) if frame else {}
    value=metric.get('value');low=row['lower'];high=row['upper']
    lower=metric.get('minimum',value) if metric.get('constraint_uses_extrema') else value
    upper=metric.get('maximum',value) if metric.get('constraint_uses_extrema') else value
    gap=None;boundary=False
    if row['kpi']=='ue_energy_j':status='legacy_kpi'
    elif row.get('empty'):status='empty_range'
    elif not number(value) or metric.get('reason') or not number(lower) or not number(upper):status='unavailable'
    else:
        below=lower<low or lower==low and not row['lower_inclusive']
        above=high is not None and (upper>high or upper==high and not row['upper_inclusive'])
        status='outside_range' if below and above else 'below_range' if below else 'above_range' if above else 'satisfied'
        gap=max(0.,low-lower,upper-high if high is not None else 0.)
        boundary=(below or above) and gap==0
    norm=gap/scale['scale'] if gap is not None and scale and number(scale.get('scale')) and scale['scale']>0 else None
    return dict(value=value if number(value) else None,status=status,absolute_gap=gap,
        normalized_gap=norm,strict_boundary_violation=boundary,
        measurement={k:deepcopy(v) for k,v in metric.items() if k!='value'})


def score(ranges,frame,scales):
    rows=[dict(kpi=r['kpi'],**assess(r,frame,scales.get(r['kpi']))) for r in ranges]
    valid=sum(r['status']in VALID for r in rows);passed=sum(r['status']=='satisfied'for r in rows)
    normalized=all(r['normalized_gap'] is not None for r in rows)
    return dict(satisfied=passed,required=len(rows),valid=valid,all_satisfied=bool(rows)and passed==len(rows),
        worst_normalized_gap=max((r['normalized_gap']for r in rows),default=0.) if normalized and rows else None,
        rows=rows)


def quality(before,after):
    a,b=before.get('worst_normalized_gap'),after.get('worst_normalized_gap')if after else None
    complete=bool(after)and before['valid']==before['required']and after['valid']==after['required']and a is not None and b is not None
    change=(a-b)*100 if complete else None
    if not complete:state='insufficient_data'
    elif after['all_satisfied']:state='maintained_feasibility' if before['all_satisfied']else'reached_intent'
    elif before['all_satisfied']:state='violated_intent'
    else:state='closer'if change>0 else'further'if change<0 else'unchanged'
    return dict(state=state,complete=complete,before_worst_normalized_gap=a,after_worst_normalized_gap=b,
        improvement_percentage_points=change,ranking_performed=False,
        basis='all_requested_targets_then_worst_normalized_interval_gap',
        oversatisfaction_reward=False)
