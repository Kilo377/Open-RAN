"""Canonical projection coordinates and shared KPI scales; no fitted weights.

Raw graph gains are preserved. Only the small projection view is converted.
Unknown/incompatible units are unavailable rather than guessed from KPI names.
"""
from copy import deepcopy

from smo.runtime import number
from smo.result_evaluation import reference_scale

# Dimension, SI multiplier. Distinguish coordinate semantics from physical units.
UNITS = {
    'W':('power',1.), 'mW':('power',.001), 'kW':('power',1000.),
    'J':('energy',1.), 'mJ':('energy',.001),
    's':('time',1.), 'ms':('time',.001), 'us':('time',.000001),
    'bit/s':('bit_rate',1.), 'kbit/s':('bit_rate',1000.), 'Mbit/s':('bit_rate',1e6), 'Gbit/s':('bit_rate',1e9),
    'FLOP/s':('compute_rate',1.), 'GFLOP/s':('compute_rate',1e9),
    'events/s':('event_rate',1.), 'events/ms':('event_rate',1000.),
    'byte':('data',1.), 'kbyte':('data',1000.),
    'ratio':('fraction',1.), 'fraction':('fraction',1.), '%':('fraction',.01),
    'weight':('scheduler_weight',1.), 'contrast':('contrast',1.),
}
SOURCE_FIELDS = ('unit','x_unit','kind','parameter','scalar_component','estimate',
    'confidence_interval','fit_domain','sampled_domain')


def factor(source, target):
    if source not in UNITS or target not in UNITS:
        raise ValueError('unknown_unit')
    a,b = UNITS[source],UNITS[target]
    if a[0] != b[0]:raise ValueError('incompatible_unit_dimensions')
    return a[1]/b[1]


def projection(row, kpi_unit):
    source = {k:deepcopy(row[k]) for k in SOURCE_FIELDS if k in row}
    result = dict(schema='oran.projection-units.v1', source=source, valid=False)
    try:
        if not isinstance(row.get('unit'),str):raise ValueError('missing_gain_unit')
        parts = row['unit'].split(' / ')
        if len(parts) not in (1,2):raise ValueError('invalid_gain_unit_expression')
        output, denominator = parts[0], parts[1] if len(parts)==2 else None
        out_factor = factor(output,kpi_unit)
        if row['kind']=='finite_contrast':
            if denominator not in (None,'contrast'):raise ValueError('contrast_is_not_a_derivative')
            coordinate, x_factor, gain_factor = 'contrast',1.,out_factor
        else:
            p=row['parameter']
            coordinate = {'ue_offload_fraction':'fraction','ue_scheduler_weights':'weight'}.get(p)
            if p=='ue_drx_configs' and row.get('scalar_component','inactivity_ms') in ('inactivity_ms','cycle_ms'):
                coordinate='ms'
            if not coordinate:raise ValueError('unknown_parameter_coordinate')
            if denominator is None:raise ValueError('missing_parameter_unit')
            x_factor=factor(denominator,coordinate)
            if row.get('x_unit') is not None and factor(row['x_unit'],denominator)!=1:
                raise ValueError('coefficient_and_fit_coordinates_disagree')
            gain_factor=out_factor/x_factor
        if not number(row.get('estimate')):raise ValueError('non_scalar_gain')
        fields=dict(estimate=row['estimate']*gain_factor,
            unit=kpi_unit+' / '+coordinate, x_unit=coordinate)
        if row.get('confidence_interval') is not None:
            fields['confidence_interval']=[v*gain_factor for v in row['confidence_interval']]
        if row['kind']!='finite_contrast':
            for key in ('fit_domain','sampled_domain'):
                if row.get(key) is not None:fields[key]=[v*x_factor for v in row[key]]
        result.update(valid=True, fields=fields, kpi_unit=kpi_unit, parameter_unit=coordinate,
            output_factor=out_factor, coordinate_factor=x_factor, gain_factor=gain_factor,
            reference_drive_unit=kpi_unit, normalized_drive_unit='dimensionless')
    except (ValueError,KeyError,TypeError) as error:
        result['reason']=str(error)
    return result


def align_graph(graph):
    result=deepcopy(graph)
    units={n['id']:n.get('unit') for n in result.get('nodes',[]) if n['kind']=='kpi'}
    for edge in result.get('edges',[]):
        if edge['kind']!='influences':continue
        for row in edge.get('sensitivities') or ([edge['sensitivity']] if edge.get('sensitivity') else []):
            row['projection_units']=projection(row,units.get(edge['target']))
        if edge.get('sensitivity'):
            edge['sensitivity']['projection_units']=projection(edge['sensitivity'],units.get(edge['target']))
    result['influence_unit_alignment']=dict(schema='oran.influence-units.v1',
        raw_gains_modified=False, value_weights='equal', normalized_unit='dimensionless')
    return result


def frozen_intent_reference(row, frame):
    """Choose one immutable reference independently of active-intent ordering.

    Modern anchors carry their original network snapshot and intent ID. The
    numeric tie-break keeps legacy/synthetic anchors deterministic when that
    provenance is absent; it does not rewrite or average their measurements.
    Keep this ordering aligned with SmoGraph.frozenIntentReference.
    """
    def order(anchor):
        snapshot=anchor.get('snapshot')
        if not isinstance(snapshot,dict):snapshot={}
        return (snapshot['time_ms'] if number(snapshot.get('time_ms')) else float('inf'),
            snapshot['slot'] if number(snapshot.get('slot')) else float('inf'),
            str(anchor.get('intent_id') or ''),anchor['current'])
    anchors=[a for a in row.get('anchors') or [] if isinstance(a,dict) and number(a.get('current'))]
    if anchors:return min(anchors,key=order)['current']
    if number(row.get('current')):return row['current']
    current=frame.get('states',{}).get('kpi:'+row['kpi'],{}).get('value')
    return current if number(current) else None


def normalization(ranges, frame, *, batch=None):
    reference={r['kpi']:frozen_intent_reference(r,frame) for r in ranges}
    basis='frozen_intent_reference';explicit={}
    pending = batch is not None and 'baseline_reference'not in batch
    if batch is not None and not pending:
        reference=deepcopy(batch.get('baseline_reference',{}))
        basis='batch_baseline_reference';explicit=batch.get('result_evaluation_scales',{})
    elif batch is not None:
        explicit=batch.get('result_evaluation_scales',{})
    def scale(row):
        if row['kpi']=='urllc_failure_fraction':
            # A failure fraction is already dimensionless. Use percentage
            # points even at a zero baseline, just as observed evaluation
            # does; dividing by the current failure fraction is not needed.
            unit=row.get('unit','ratio')
            return dict(scale=1. if unit in ('ratio','fraction') else 100. if unit=='%' else None,
                scale_basis='ratio_percentage_points' if unit in ('ratio','fraction','%') else 'incompatible_ratio_unit',
                reference_value=reference.get(row['kpi']))
        return reference_scale(row['kpi'],reference,explicit,basis)
    return dict(schema='oran.influence-normalization.v1', value_weights='equal',
        reference_basis=basis, batch_before_reference_pending=pending, observed_after_kpis_used=False,
        reference_selection='earliest_finite_anchor_by_snapshot_then_id_then_value',
        kpis=[dict(kpi=r['kpi'], **scale(r),
            weight=1.,weight_basis='equal_kpi_value') for r in ranges])
