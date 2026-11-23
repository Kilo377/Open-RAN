"""Opt-in current-observation coordinates; historical intent bounds stay raw.

This module is deliberately separate from influence_units.normalization.
It reads only the supplied public decision frame and never a deployment
batch, historical anchor value, proposed outcome or feedback record.
"""
from copy import deepcopy
import math


POLICY = 'current_observation_v1'
POLICY_VERSION = 1
FRAME_FIELDS = ('source_id', 'source_sha256', 'run_id', 'graph_sha256',
                'evaluation_origin_id', 'window_ms', 'evaluation_window')
MEASUREMENT_FIELDS = ('source', 'sampled_at_slot', 'window_start_slot', 'window_slots',
                      'window', 'aggregation', 'sample_count', 'reason', 'status',
                      'unit', 'canonical_unit', 'measurement')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _pick(value, keys):
    return {k:deepcopy(value[k]) for k in keys if k in value}


def normalization(ranges, frame):
    """Return the existing projection-normalization shape with explicit policy.

    A probability's physical range defines its scale even when its current
    reading is missing; such a row still reports observation unavailable.
    Other zero/missing readings supply no relative-percentage denominator.
    Neither ranges nor anchors are rewritten or used as fallback values.
    """
    snapshot = {k:frame.get(k) if _finite(frame.get(k)) else None for k in ('slot', 'time_ms')}
    rows = []
    states = frame.get('states') if isinstance(frame.get('states'), dict) else {}
    for row in ranges:
        key = row['kpi']
        state = states.get('kpi:'+key)
        present = isinstance(state, dict)
        state = state if present else {}
        current = state.get('value')
        observed = _finite(current)
        if key == 'urllc_failure_fraction':
            unit = row.get('unit', 'ratio')
            scale = 1. if unit in ('ratio', 'fraction') else 100. if unit == '%' else None
            basis = 'ratio_percentage_points' if scale is not None else 'incompatible_ratio_unit'
        elif not observed:
            scale, basis = None, 'current_observation_unavailable'
        elif current == 0:
            scale, basis = None, 'explicit_scale_required'
        else:
            scale, basis = abs(current), 'current_observation_magnitude'
        reason = basis if scale is None else None if observed else 'current_observation_unavailable'
        rows.append(dict(kpi=key, scale=scale, scale_basis=basis,
            reference_value=current if observed else None, weight=1., weight_basis='equal_kpi_value',
            status='available' if reason is None else 'unavailable', reason=reason,
            scale_available=scale is not None, reference_observation_available=observed,
            observation_provenance=dict(snapshot=deepcopy(snapshot), state_present=present,
                metadata=_pick(state, MEASUREMENT_FIELDS))))
    return dict(schema='oran.influence-normalization.v1', policy=POLICY, policy_version=POLICY_VERSION,
        value_weights='equal', reference_basis='current_observed_frame',
        reference_selection='current_frame_only', reference_snapshot=snapshot,
        reference_frame_provenance=_pick(frame, FRAME_FIELDS),
        batch_before_reference_pending=False, observed_after_kpis_used=False,
        deployment_batch_used=False, intent_anchors_used_for_scale=False,
        interval_endpoints_modified=False, kpis=rows)
